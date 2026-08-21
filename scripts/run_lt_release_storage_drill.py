#!/usr/bin/env python3
"""Run a non-promotional LT release storage drill on a target volume."""

from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.pipeline.storage_drill import (  # noqa: E402
    JOB_OWNER_ENV,
    StorageDrillError,
    _enter_windows_kill_on_close_job,
    _terminate_worker_tree,
    run_storage_drill,
)
from pfc_shaping.pipeline.atomic_promotion import PromotionError  # noqa: E402


SUPERVISION_TOKEN_ENV = "PFC_STORAGE_DRILL_SUPERVISION_TOKEN"


def main(argv: list[str] | None = None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drill-root",
        required=True,
        type=Path,
        help="existing target-volume directory; only its isolated drill namespace is used",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--alias-root",
        action="append",
        default=[],
        type=Path,
        help="optional drive/UNC/FQDN alias resolving to the same target directory",
    )
    parser.add_argument(
        "--require-alias-class",
        action="append",
        default=[],
        choices=("drive", "unc_short", "unc_fqdn"),
        help="alias class required for an IT qualification run",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--global-timeout-seconds", type=float)
    parser.add_argument("--_run-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_supervision-token", help=argparse.SUPPRESS)
    args = parser.parse_args(selected_argv)
    if not args._run_child:
        return _run_supervised_cli(selected_argv, args)
    expected_token = os.environ.get(SUPERVISION_TOKEN_ENV, "")
    if not expected_token or not hmac.compare_digest(
        str(args._supervision_token or ""),
        expected_token,
    ):
        print(
            json.dumps(
                {
                    "schema_version": "lt_release_storage_drill_error.v1",
                    "status": "FAIL",
                    "promotion_ready": False,
                    "production_authorization": False,
                    "error": "internal child mode requires a live parent handshake",
                },
                sort_keys=True,
            )
        )
        return 2
    if os.environ.get(JOB_OWNER_ENV) == "1":
        _enter_windows_kill_on_close_job()
    try:
        result = run_storage_drill(
            drill_root=args.drill_root,
            run_id=args.run_id,
            alias_roots=args.alias_root,
            required_alias_classes=args.require_alias_class,
            timeout_seconds=args.timeout_seconds,
        )
    except (StorageDrillError, PromotionError, OSError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "lt_release_storage_drill_error.v1",
                    "status": "FAIL",
                    "promotion_ready": False,
                    "production_authorization": False,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def _run_supervised_cli(argv: list[str], args: argparse.Namespace) -> int:
    global_timeout = args.global_timeout_seconds
    if global_timeout is None:
        global_timeout = (float(args.timeout_seconds) + 2) * 7 + 30
    if not 10 <= global_timeout <= 3600:
        print(
            json.dumps(
                {
                    "schema_version": "lt_release_storage_drill_error.v1",
                    "status": "FAIL",
                    "promotion_ready": False,
                    "production_authorization": False,
                    "error": "global timeout must be between 10 and 3600 seconds",
                },
                sort_keys=True,
            )
        )
        return 2
    token = secrets.token_hex(32)
    child_argv = [value for value in argv if value != "--_run-child"] + [
        "--_run-child",
        "--_supervision-token",
        token,
    ]
    environment = os.environ.copy()
    environment[SUPERVISION_TOKEN_ENV] = token
    if os.name == "nt":
        environment[JOB_OWNER_ENV] = "1"
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if os.name == "nt":
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *child_argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    timed_out = False
    stdout = b""
    stderr = b""
    communication_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        try:
            stdout, stderr = child.communicate(timeout=global_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException as exc:
            communication_error = exc
    finally:
        try:
            _terminate_worker_tree(child)
        except BaseException as exc:
            cleanup_error = exc
    cancellation = next(
        (
            error
            for error in (communication_error, cleanup_error)
            if isinstance(error, (KeyboardInterrupt, SystemExit))
        ),
        None,
    )
    if cancellation is not None:
        secondary = cleanup_error if cancellation is communication_error else communication_error
        if secondary is not None:
            cancellation.add_note(f"secondary supervision failure: {secondary}")
        raise cancellation
    if cleanup_error is not None:
        print(
            json.dumps(
                {
                    "schema_version": "lt_release_storage_drill_error.v1",
                    "status": "FAIL",
                    "promotion_ready": False,
                    "production_authorization": False,
                    "process_cleanup_status": "FAIL",
                    "error": f"storage drill process cleanup failed: {cleanup_error}",
                },
                sort_keys=True,
            )
        )
        return 2
    if communication_error is not None:
        print(
            json.dumps(
                {
                    "schema_version": "lt_release_storage_drill_error.v1",
                    "status": "FAIL",
                    "promotion_ready": False,
                    "production_authorization": False,
                    "process_cleanup_status": "PASS",
                    "error": f"storage drill communication failed: {communication_error}",
                },
                sort_keys=True,
            )
        )
        return 2
    if timed_out:
        print(
            json.dumps(
                {
                    "schema_version": "lt_release_storage_drill_error.v1",
                    "status": "TIMEOUT",
                    "promotion_ready": False,
                    "production_authorization": False,
                    "process_cleanup_status": "PASS",
                    "error": "full storage drill exceeded its supervised deadline",
                },
                sort_keys=True,
            )
        )
        return 2
    if stderr:
        print(stderr.decode(errors="replace"), file=sys.stderr, end="")
    if stdout:
        print(stdout.decode(errors="replace"), end="")
        return int(child.returncode or 0)
    print(
        json.dumps(
            {
                "schema_version": "lt_release_storage_drill_error.v1",
                "status": "FAIL",
                "promotion_ready": False,
                "production_authorization": False,
                "error": f"storage drill child exited {child.returncode} without JSON output",
            },
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
