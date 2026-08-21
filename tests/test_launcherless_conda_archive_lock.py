from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import build_launcherless_conda_archive_lock as archive_lock


def _add_tar_file(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(payload))


def _write_test_conda_archive(
    path: Path,
    *,
    name: str,
    version: str,
    build: str,
    payload_path: str,
    payload: bytes,
) -> None:
    index = {
        "name": name,
        "version": version,
        "build": build,
        "build_number": 0,
        "subdir": "win-64",
    }
    paths = {
        "paths_version": 1,
        "paths": [
            {
                "_path": payload_path,
                "path_type": "hardlink",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_in_bytes": len(payload),
            }
        ],
    }
    with tarfile.open(path, "w:bz2") as package:
        _add_tar_file(
            package,
            "info/index.json",
            json.dumps(index, sort_keys=True).encode("utf-8"),
        )
        _add_tar_file(
            package,
            "info/paths.json",
            json.dumps(paths, sort_keys=True).encode("utf-8"),
        )
        _add_tar_file(package, payload_path, payload)


def _fixture(
    tmp_path: Path, *, package_count: int = 2
) -> tuple[Path, Path, list[dict[str, object]]]:
    prefix = tmp_path / "runtime"
    metadata_root = prefix / "conda-meta"
    cache = tmp_path / "cache"
    metadata_root.mkdir(parents=True)
    cache.mkdir()
    (metadata_root / "history").write_text(
        "==> 2026-07-28 00:00:00 <==\n"
        "# cmd: conda create --offline --copy --file exact.txt\n"
        "# conda version: 24.1.2\n",
        encoding="utf-8",
    )
    records: list[dict[str, object]] = []
    for index in range(package_count):
        name = f"package{index}"
        version = f"1.0.{index}"
        build = f"h{index}_0"
        filename = f"{name}-{version}-{build}.tar.bz2"
        installed_payload = f"{name}\n".encode()
        _write_test_conda_archive(
            cache / filename,
            name=name,
            version=version,
            build=build,
            payload_path=f"Lib/{name}.py",
            payload=installed_payload,
        )
        payload = (cache / filename).read_bytes()
        record: dict[str, object] = {
            "name": name,
            "version": version,
            "build": build,
            "build_number": 0,
            "subdir": "win-64",
            "channel": "https://example.invalid/conda/win-64",
            "fn": filename,
            "url": f"https://example.invalid/conda/win-64/{filename}",
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "depends": ["python >=3.11"],
            "constrains": [],
            "files": [f"Lib/{name}.py"],
        }
        (metadata_root / f"{name}-{version}-{build}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)
    return prefix, cache, records


def test_metadata_only_conda_archive_has_empty_verified_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata-only-1.0-test_0.tar.bz2"
    index = {
        "name": "metadata-only",
        "version": "1.0",
        "build": "test_0",
        "build_number": 0,
        "subdir": "win-64",
    }
    paths = {"paths_version": 1, "paths": []}
    with tarfile.open(path, "w:bz2") as package:
        _add_tar_file(
            package,
            "info/index.json",
            json.dumps(index, sort_keys=True).encode("utf-8"),
        )
        _add_tar_file(
            package,
            "info/paths.json",
            json.dumps(paths, sort_keys=True).encode("utf-8"),
        )

    result = archive_lock._conda_archive_payload_contract(
        path.read_bytes(),
        filename=path.name,
        expected_identity=index,
    )

    assert result["archive_payload_file_count"] == 0
    assert result["archive_payload_tree_sha256"] == hashlib.sha256().hexdigest()


def test_windows_binary_prefix_payload_is_not_rewritten(tmp_path: Path) -> None:
    marker = b"C:\\short\\_h_env"
    payload = b"binary-header\0" + marker + b"\0binary-tail"

    candidates = archive_lock._prefix_replacement_candidates(
        payload,
        placeholder=marker.decode("utf-8"),
        file_mode="binary",
        prefix=tmp_path / "a-prefix-longer-than-the-placeholder",
    )

    assert candidates == (payload,)


def test_windows_pyzzer_binary_prefix_payload_is_fail_closed(
    tmp_path: Path,
) -> None:
    marker = b"C:\\short\\_h_env"
    payload = b"#!" + marker + b"\\python.exe\nPK\x05\x06"

    with pytest.raises(ValueError, match="pyzzer.*unsupported"):
        archive_lock._prefix_replacement_candidates(
            payload,
            placeholder=marker.decode("utf-8"),
            file_mode="binary",
            prefix=tmp_path / "runtime",
        )


@pytest.mark.parametrize(
    "payload_path",
    [
        "Scripts/.package-pre-link.bat",
        "Scripts/.package_pre_link.cmd",
        "Scripts/.package-post-link.bat",
        "Scripts/.package_post_link.cmd",
        "Scripts/.package-pre-unlink.bat",
        "Scripts/.package_pre_unlink.cmd",
        "Scripts/.package-post-unlink.bat",
        "Scripts/.package_post_unlink.cmd",
    ],
)
def test_conda_archive_rejects_all_lifecycle_script_families(
    tmp_path: Path,
    payload_path: str,
) -> None:
    path = tmp_path / "lifecycle-1.0-test_0.tar.bz2"
    identity = {
        "name": "lifecycle",
        "version": "1.0",
        "build": "test_0",
        "build_number": 0,
        "subdir": "win-64",
    }
    _write_test_conda_archive(
        path,
        name="lifecycle",
        version="1.0",
        build="test_0",
        payload_path=payload_path,
        payload=b"forbidden lifecycle script\n",
    )

    with pytest.raises(ValueError, match="lifecycle execution payload"):
        archive_lock._conda_archive_payload_contract(
            path.read_bytes(),
            filename=path.name,
            expected_identity=identity,
        )


def test_archive_lock_is_deterministic_and_replays_exact_bytes(tmp_path: Path) -> None:
    prefix, cache, _ = _fixture(tmp_path)
    first = tmp_path / "lock-a.json"
    second = tmp_path / "lock-b.json"

    result = archive_lock.build_conda_archive_lock(
        runtime_prefix=prefix,
        package_cache_roots=(cache,),
        output=first,
    )
    archive_lock.build_conda_archive_lock(
        runtime_prefix=prefix,
        package_cache_roots=(cache,),
        output=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert result["package_count"] == 2
    assert result["network_required"] is False
    assert result["admin_required"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
    assert result["explicit_spec_lines"][0] == "@EXPLICIT"
    assert all(
        line.startswith("file:///") and "#" in line
        for line in result["explicit_spec_lines"][1:]
    )
    replay = archive_lock.verify_conda_archive_lock(
        lock_path=first,
        expected_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
    )
    assert replay == result


def test_archive_lock_rejects_changed_archive_bytes(tmp_path: Path) -> None:
    prefix, cache, records = _fixture(tmp_path, package_count=1)
    (cache / str(records[0]["fn"])).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="archive bytes diverge"):
        archive_lock.conda_archive_lock_payload(
            runtime_prefix=prefix,
            package_cache_roots=(cache,),
        )


def test_archive_lock_rejects_missing_or_ambiguous_archive(tmp_path: Path) -> None:
    prefix, cache, records = _fixture(tmp_path, package_count=1)
    filename = str(records[0]["fn"])
    archive = cache / filename
    payload = archive.read_bytes()
    archive.unlink()
    with pytest.raises(ValueError, match="resolution is not unique"):
        archive_lock.conda_archive_lock_payload(
            runtime_prefix=prefix,
            package_cache_roots=(cache,),
        )

    archive.write_bytes(payload)
    second_cache = tmp_path / "cache-two"
    second_cache.mkdir()
    (second_cache / filename).write_bytes(payload)
    with pytest.raises(ValueError, match="resolution is not unique"):
        archive_lock.conda_archive_lock_payload(
            runtime_prefix=prefix,
            package_cache_roots=(cache, second_cache),
        )


def test_archive_lock_rejects_duplicate_metadata_keys(tmp_path: Path) -> None:
    prefix, cache, records = _fixture(tmp_path, package_count=1)
    record = records[0]
    path = next((prefix / "conda-meta").glob("*.json"))
    path.write_text(
        "{"
        f'"name":"{record["name"]}",'
        f'"name":"{record["name"]}"'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        archive_lock.conda_archive_lock_payload(
            runtime_prefix=prefix,
            package_cache_roots=(cache,),
        )


def test_archive_lock_rejects_tampered_lock_before_replay(tmp_path: Path) -> None:
    prefix, cache, _ = _fixture(tmp_path, package_count=1)
    lock = tmp_path / "lock.json"
    archive_lock.build_conda_archive_lock(
        runtime_prefix=prefix,
        package_cache_roots=(cache,),
        output=lock,
    )

    with pytest.raises(ValueError, match="SHA-256 diverges"):
        archive_lock.verify_conda_archive_lock(
            lock_path=lock,
            expected_sha256="0" * 64,
        )


def test_archive_lock_rejects_noncanonical_bytes_with_matching_hash(
    tmp_path: Path,
) -> None:
    prefix, cache, _ = _fixture(tmp_path, package_count=1)
    canonical = tmp_path / "canonical.json"
    archive_lock.build_conda_archive_lock(
        runtime_prefix=prefix,
        package_cache_roots=(cache,),
        output=canonical,
    )
    value = json.loads(canonical.read_bytes())
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical JSONL"):
        archive_lock.verify_conda_archive_lock(
            lock_path=noncanonical,
            expected_sha256=hashlib.sha256(noncanonical.read_bytes()).hexdigest(),
        )


def test_archive_lock_refuses_existing_output(tmp_path: Path) -> None:
    prefix, cache, _ = _fixture(tmp_path, package_count=1)
    output = tmp_path / "lock.json"
    output.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="must not already exist"):
        archive_lock.build_conda_archive_lock(
            runtime_prefix=prefix,
            package_cache_roots=(cache,),
            output=output,
        )
    assert output.read_text(encoding="utf-8") == "preserve"


def test_explicit_spec_requires_verified_lock_and_is_new_name(tmp_path: Path) -> None:
    prefix, cache, _ = _fixture(tmp_path, package_count=2)
    lock = tmp_path / "lock.json"
    spec = tmp_path / "explicit.txt"
    archive_lock.build_conda_archive_lock(
        runtime_prefix=prefix,
        package_cache_roots=(cache,),
        output=lock,
    )
    lock_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest()

    result, record = archive_lock.materialize_conda_explicit_spec(
        lock_path=lock,
        expected_sha256=lock_sha256,
        output=spec,
    )

    assert spec.read_text(encoding="ascii").splitlines() == result[
        "explicit_spec_lines"
    ]
    assert record["sha256"] == hashlib.sha256(spec.read_bytes()).hexdigest()
    assert record["line_count"] == 3
    with pytest.raises(ValueError, match="must not already exist"):
        archive_lock.materialize_conda_explicit_spec(
            lock_path=lock,
            expected_sha256=lock_sha256,
            output=spec,
        )


def _explicit_target(
    tmp_path: Path, lock: dict[str, object], spec: Path
) -> tuple[Path, list[dict[str, object]]]:
    prefix = tmp_path / "replayed-runtime"
    metadata_root = prefix / "conda-meta"
    metadata_root.mkdir(parents=True)
    files: list[dict[str, object]] = []
    history = (
        "==> 2026-07-28 00:00:00 <==\n"
        f"# cmd: conda create --offline --copy --prefix {prefix} "
        f"--file {spec} --yes --json\n"
        "# conda version: 24.1.2\n"
    )
    for package in lock["packages"]:
        relative = f"Lib/{package['name']}.py"
        target = prefix / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"{package['name']}\n".encode())
        record = {
            "name": package["name"],
            "version": package["version"],
            "build": package["build"],
            "build_number": package["build_number"],
            "subdir": package["subdir"],
            "channel": "<unknown>",
            "fn": package["filename"],
            "url": package["archive_uri"],
            "md5": package["md5"],
            "sha256": package["sha256"],
            "size": package["size"],
            "depends": package["depends"],
            "constrains": package["constrains"],
            "files": [relative],
        }
        metadata_name = (
            f"{package['name']}-{package['version']}-{package['build']}.json"
        )
        metadata_path = metadata_root / metadata_name
        metadata_path.write_text(json.dumps(record), encoding="utf-8")
        files.extend(
            [
                {"path": relative},
                {"path": f"conda-meta/{metadata_name}"},
            ]
        )
        history += f"+<unknown>/{package['subdir']}::{package['name']}\n"
        history += f"# spec [url={package['archive_uri']},md5={package['md5']}]\n"
    history_path = metadata_root / "history"
    history_path.write_text(history, encoding="utf-8")
    files.append({"path": "conda-meta/history"})
    files.sort(key=lambda item: str(item["path"]))
    return prefix, files


def test_prefix_audit_binds_lock_spec_history_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, cache, _ = _fixture(tmp_path, package_count=2)
    lock_path = tmp_path / "lock.json"
    spec = tmp_path / "explicit.txt"
    lock = archive_lock.build_conda_archive_lock(
        runtime_prefix=source,
        package_cache_roots=(cache,),
        output=lock_path,
    )
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    archive_lock.materialize_conda_explicit_spec(
        lock_path=lock_path,
        expected_sha256=lock_sha256,
        output=spec,
    )
    prefix, files = _explicit_target(tmp_path, lock, spec)
    manifest_path = tmp_path / "python-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        archive_lock,
        "validate_python_runtime_manifest",
        lambda **_: {
            "files": files,
            "tree_sha256": "1" * 64,
            "file_count": len(files),
            "python_executable_sha256": "2" * 64,
            "python_dll_sha256": "3" * 64,
        },
    )
    repo_build_receipt = tmp_path / "repo-build-receipt.json"
    repo_build_receipt.write_text("{}\n", encoding="utf-8")
    repo_build_identity = {
        "path": str(repo_build_receipt),
        "sha256": "b" * 64,
        "schema_version": "fmv_lt_repo_local_conda_prefix_build.v1",
        "status": "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION",
        "command_sha256": "c" * 64,
        "external_guard_sha256": "d" * 64,
        "external_guard_unchanged": True,
    }
    monkeypatch.setattr(
        archive_lock,
        "validate_repo_local_conda_prefix_build_receipt",
        lambda **_: repo_build_identity,
    )

    receipt = archive_lock.audit_conda_prefix_build(
        lock_path=lock_path,
        expected_lock_sha256=lock_sha256,
        explicit_spec_path=spec,
        expected_explicit_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
        runtime_prefix=prefix,
        python_runtime_manifest=manifest_path,
        expected_python_runtime_manifest_sha256=manifest_sha256,
        repo_local_build_receipt=repo_build_receipt,
        expected_repo_local_build_receipt_sha256="b" * 64,
        output=tmp_path / "prefix-receipt.json",
    )

    assert receipt["status"] == (
        "PASS_LOCAL_EXPLICIT_ARCHIVE_REPLAY_NOT_PRODUCTION"
    )
    assert receipt["package_count"] == 2
    assert receipt["file_count"] == 5
    assert receipt["build_policy"]["solver_used"] is False
    assert receipt["build_policy"]["target_python_executed_before_manifest"] is False
    assert receipt["production_authorization"] is False

    (prefix / "Lib" / "package0.py").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="differ from the locked archive"):
        archive_lock.audit_conda_prefix_build(
            lock_path=lock_path,
            expected_lock_sha256=lock_sha256,
            explicit_spec_path=spec,
            expected_explicit_spec_sha256=hashlib.sha256(
                spec.read_bytes()
            ).hexdigest(),
            runtime_prefix=prefix,
            python_runtime_manifest=manifest_path,
            expected_python_runtime_manifest_sha256=manifest_sha256,
            repo_local_build_receipt=repo_build_receipt,
            expected_repo_local_build_receipt_sha256="b" * 64,
            output=tmp_path / "tampered-prefix-receipt.json",
        )


def test_prefix_audit_rejects_non_explicit_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, cache, _ = _fixture(tmp_path, package_count=1)
    lock_path = tmp_path / "lock.json"
    spec = tmp_path / "explicit.txt"
    lock = archive_lock.build_conda_archive_lock(
        runtime_prefix=source,
        package_cache_roots=(cache,),
        output=lock_path,
    )
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    archive_lock.materialize_conda_explicit_spec(
        lock_path=lock_path,
        expected_sha256=lock_sha256,
        output=spec,
    )
    prefix, files = _explicit_target(tmp_path, lock, spec)
    (prefix / "conda-meta" / "history").write_text(
        "# cmd: conda create python=3.11\n# conda version: 24.1.2\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "python-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        archive_lock,
        "validate_python_runtime_manifest",
        lambda **_: {
            "files": files,
            "tree_sha256": "1" * 64,
            "file_count": len(files),
            "python_executable_sha256": "2" * 64,
            "python_dll_sha256": "3" * 64,
        },
    )
    repo_build_receipt = tmp_path / "repo-build-receipt.json"
    repo_build_receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        archive_lock,
        "validate_repo_local_conda_prefix_build_receipt",
        lambda **_: {
            "path": str(repo_build_receipt),
            "sha256": "b" * 64,
            "schema_version": "fmv_lt_repo_local_conda_prefix_build.v1",
            "status": "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION",
            "command_sha256": "c" * 64,
            "external_guard_sha256": "d" * 64,
            "external_guard_unchanged": True,
        },
    )

    with pytest.raises(ValueError, match="does not prove explicit offline copy"):
        archive_lock.audit_conda_prefix_build(
            lock_path=lock_path,
            expected_lock_sha256=lock_sha256,
            explicit_spec_path=spec,
            expected_explicit_spec_sha256=hashlib.sha256(
                spec.read_bytes()
            ).hexdigest(),
            runtime_prefix=prefix,
            python_runtime_manifest=manifest_path,
            expected_python_runtime_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            repo_local_build_receipt=repo_build_receipt,
            expected_repo_local_build_receipt_sha256="b" * 64,
            output=tmp_path / "prefix-receipt.json",
        )


def test_prefix_audit_records_dependency_metadata_as_non_authority(
    tmp_path: Path,
) -> None:
    source, cache, _ = _fixture(tmp_path, package_count=2)
    lock = archive_lock.conda_archive_lock_payload(
        runtime_prefix=source,
        package_cache_roots=(cache,),
    )
    prefix = tmp_path / "explicit-prefix"
    metadata = prefix / "conda-meta"
    metadata.mkdir(parents=True)
    package = lock["packages"][0]
    first_relative = f"Lib/{package['name']}.py"
    first_target = prefix / Path(first_relative)
    first_target.parent.mkdir(parents=True, exist_ok=True)
    first_target.write_bytes(f"{package['name']}\n".encode())
    record = {
        "name": package["name"],
        "version": package["version"],
        "build": package["build"],
        "build_number": package["build_number"],
        "subdir": package["subdir"],
        "channel": "<unknown>",
        "fn": package["filename"],
        "url": package["archive_uri"],
        "md5": package["md5"],
        "sha256": package["sha256"],
        "size": package["size"],
        "depends": ["unlocked >=1"],
        "constrains": package["constrains"],
        "files": [first_relative],
    }
    metadata_path = metadata / (
        f"{package['name']}-{package['version']}-{package['build']}.json"
    )
    metadata_path.write_text(json.dumps(record), encoding="utf-8")
    other = lock["packages"][1]
    other_relative = f"Lib/{other['name']}.py"
    (prefix / Path(other_relative)).write_bytes(f"{other['name']}\n".encode())
    other_record = {
        "name": other["name"],
        "version": other["version"],
        "build": other["build"],
        "build_number": other["build_number"],
        "subdir": other["subdir"],
        "channel": "<unknown>",
        "fn": other["filename"],
        "url": other["archive_uri"],
        "md5": other["md5"],
        "sha256": other["sha256"],
        "size": other["size"],
        "depends": other["depends"],
        "constrains": other["constrains"],
        "files": [other_relative],
    }
    other_path = metadata / (
        f"{other['name']}-{other['version']}-{other['build']}.json"
    )
    other_path.write_text(json.dumps(other_record), encoding="utf-8")

    records, _ = archive_lock._validated_explicit_prefix_records(
        prefix=prefix,
        locked_packages=lock["packages"],
    )
    selected = next(
        item for item in records if item["filename"] == package["filename"]
    )
    assert selected["target_record_depends"] == ["unlocked >=1"]
    assert selected["source_record_depends"] == package["depends"]
    assert selected["dependency_metadata_matches_source_record"] is False
