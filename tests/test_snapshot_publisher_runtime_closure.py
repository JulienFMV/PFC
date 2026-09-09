from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_snapshot_publisher_runtime_closure import (
    build_runtime_dependency_closure,
    validate_runtime_dependency_closure,
)

_DISTRIBUTIONS = {
    "cffi": "2.1.0",
    "cryptography": "47.0.0",
    "numpy": "2.0.2",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "pycparser": "3.0",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.2",
    "PyYAML": "6.0.3",
    "six": "1.17.0",
    "tzdata": "2026.3",
}


def test_closure_is_built_only_from_locked_record_verified_wheels(
    tmp_path: Path,
) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(tmp_path)
    closure = tmp_path / "closure"
    receipt = tmp_path / "receipt.json"

    result = build_runtime_dependency_closure(
        wheelhouse,
        closure,
        receipt,
        lock_path=lock_path,
    )
    replay = validate_runtime_dependency_closure(
        wheelhouse,
        closure,
        receipt,
        lock_path=lock_path,
    )

    assert result["schema_version"] == (
        "fmv_lt_snapshot_publisher_dependency_closure.v2"
    )
    assert result["wheel_provenance"] == (
        "VERIFIED_ARCHIVES_AND_RECORD_AGAINST_UV_LOCK"
    )
    assert result["distribution_count"] == len(_DISTRIBUTIONS)
    assert replay["receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert json.dumps(
        json.loads(receipt.read_bytes()), sort_keys=True, separators=(",", ":")
    ).encode("ascii") == receipt.read_bytes()


def test_closure_rejects_same_name_and_version_with_other_wheel_bytes(
    tmp_path: Path,
) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(tmp_path)
    wheel = next(wheelhouse.glob("numpy-*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"post-lock mutation")

    with pytest.raises(ValueError, match="differs from uv.lock"):
        build_runtime_dependency_closure(
            wheelhouse,
            tmp_path / "closure",
            tmp_path / "receipt.json",
            lock_path=lock_path,
        )


@pytest.mark.parametrize("attack", ["bad_record_hash", "unrecorded_member"])
def test_closure_rejects_invalid_wheel_record(tmp_path: Path, attack: str) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(
        tmp_path,
        wheel_attacks={"numpy": attack},
    )

    with pytest.raises(ValueError, match="RECORD"):
        build_runtime_dependency_closure(
            wheelhouse,
            tmp_path / "closure",
            tmp_path / "receipt.json",
            lock_path=lock_path,
        )


def test_closure_rejects_wrong_target_tag(tmp_path: Path) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(
        tmp_path,
        wheel_tags={"numpy": "cp311-cp311-manylinux_2_28_x86_64"},
    )

    with pytest.raises(ValueError, match="compatible locked wheel is unavailable: numpy"):
        build_runtime_dependency_closure(
            wheelhouse,
            tmp_path / "closure",
            tmp_path / "receipt.json",
            lock_path=lock_path,
        )


def test_closure_rejects_case_insensitive_cross_wheel_collision(
    tmp_path: Path,
) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(
        tmp_path,
        additional_files={
            "cffi": {"Shared.py": b"FIRST = True\n"},
            "cryptography": {"shared.py": b"SECOND = True\n"},
        },
    )

    with pytest.raises(ValueError, match="installation path collision"):
        build_runtime_dependency_closure(
            wheelhouse,
            tmp_path / "closure",
            tmp_path / "receipt.json",
            lock_path=lock_path,
        )


def test_closure_builder_does_not_delete_racing_unowned_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(tmp_path)
    closure = tmp_path / "closure"
    receipt = tmp_path / "receipt.json"
    original_mkdir = Path.mkdir

    def racing_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path == closure:
            original_mkdir(path)
            (path / "owned-by-other-builder.txt").write_text(
                "do not delete", encoding="utf-8"
            )
            raise FileExistsError("injected concurrent builder")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)

    with pytest.raises(FileExistsError, match="concurrent builder"):
        build_runtime_dependency_closure(
            wheelhouse,
            closure,
            receipt,
            lock_path=lock_path,
        )

    assert (closure / "owned-by-other-builder.txt").read_text(
        encoding="utf-8"
    ) == "do not delete"


def test_closure_replay_rejects_forged_receipt(tmp_path: Path) -> None:
    wheelhouse, lock_path = _write_test_wheelhouse(tmp_path)
    closure = tmp_path / "closure"
    receipt = tmp_path / "receipt.json"
    build_runtime_dependency_closure(
        wheelhouse,
        closure,
        receipt,
        lock_path=lock_path,
    )
    forged = json.loads(receipt.read_bytes())
    forged["wheels"][0]["record_sha256"] = "0" * 64
    receipt.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="wheel evidence diverges"):
        validate_runtime_dependency_closure(
            wheelhouse,
            closure,
            receipt,
            lock_path=lock_path,
        )


def _write_test_wheelhouse(
    root: Path,
    *,
    wheel_attacks: dict[str, str] | None = None,
    wheel_tags: dict[str, str] | None = None,
    additional_files: dict[str, dict[str, bytes]] | None = None,
) -> tuple[Path, Path]:
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir()
    entries: list[tuple[str, str, str, int]] = []
    for name, version in _DISTRIBUTIONS.items():
        normalized = name.lower().replace("-", "_")
        tag = (wheel_tags or {}).get(name, "py3-none-any")
        wheel = wheelhouse / f"{normalized}-{version}-{tag}.whl"
        files = {
            f"{normalized}_publisher_fixture.py": (
                f"NAME = {name!r}\nVERSION = {version!r}\n"
            ).encode("ascii"),
            **(additional_files or {}).get(name, {}),
        }
        _write_test_wheel(
            wheel,
            name=name,
            version=version,
            tag=tag,
            files=files,
            attack=(wheel_attacks or {}).get(name),
        )
        payload = wheel.read_bytes()
        entries.append(
            (wheel.name, hashlib.sha256(payload).hexdigest(), name, len(payload))
        )
    lock_path = root / "uv.lock"
    lines = ['version = 1', 'revision = 3', 'requires-python = ">=3.11"', ""]
    for filename, digest, name, size in entries:
        lines.extend(
            [
                "[[package]]",
                f'name = "{name}"',
                f'version = "{_DISTRIBUTIONS[name]}"',
                "wheels = [",
                (
                    "  { url = \"https://example.invalid/"
                    f"{filename}\", hash = \"sha256:{digest}\", size = {size} }},"
                ),
                "]",
                "",
            ]
        )
    lock_path.write_text("\n".join(lines), encoding="utf-8")
    return wheelhouse, lock_path


def _write_test_wheel(
    path: Path,
    *,
    name: str,
    version: str,
    tag: str,
    files: dict[str, bytes],
    attack: str | None,
) -> None:
    dist_info = f"{name.lower().replace('-', '_')}-{version}.dist-info"
    payloads = {
        **files,
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
        ).encode("ascii"),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: pfc-test\n"
            "Root-Is-Purelib: true\n"
            f"Tag: {tag}\n"
        ).encode("ascii"),
    }
    if attack == "unrecorded_member":
        payloads["unrecorded.py"] = b"UNRECORDED = True\n"
    record_path = f"{dist_info}/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for index, (member, payload) in enumerate(sorted(payloads.items())):
        if attack == "unrecorded_member" and member == "unrecorded.py":
            continue
        digest = hashlib.sha256(payload).digest()
        if attack == "bad_record_hash" and index == 0:
            digest = b"\x00" * 32
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        writer.writerow((member, f"sha256={encoded}", str(len(payload))))
    writer.writerow((record_path, "", ""))
    payloads[record_path] = record.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in sorted(payloads.items()):
            archive.writestr(member, payload)
