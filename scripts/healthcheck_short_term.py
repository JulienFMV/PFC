#!/usr/bin/env python3
"""
Automated short-term healthcheck for the PFC stack.

Checks:
1. AST syntax validation on critical modules/scripts.
2. FutureBoost evaluation script.
3. FutureBoost campaign script (baseline only by default).
4. Optional production dry run (skip LEAR backtest).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "healthcheck_short_term.json"
DEFAULT_LOG = ROOT / "healthcheck_short_term.log"

CRITICAL_FILES = [
    "pfc_shaping/data/ingest_entso.py",
    "pfc_shaping/model/futureboost_experimental.py",
    "run_pfc_production.py",
    "scripts/export_pricefm_ch_dataset.py",
    "scripts/eval_futureboost_experimental.py",
    "scripts/eval_futureboost_campaign.py",
    "scripts/run_pricefm_ch_poc.py",
]


def _run_command(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout: int = 900,
) -> dict[str, object]:
    start = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "seconds": round(time.time() - start, 2),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "command": cmd,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "seconds": round(time.time() - start, 2),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "command": cmd,
        }


def _ast_checks() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for rel in CRITICAL_FILES:
        path = ROOT / rel
        item = {"file": rel, "ok": False}
        try:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=rel)
            item["ok"] = True
        except Exception as exc:
            item["error"] = str(exc)
        rows.append(item)
    return {
        "ok": all(bool(r["ok"]) for r in rows),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short-term healthchecks for PFC.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for ppa_env checks.")
    parser.add_argument(
        "--pricefm-python",
        default=r"C:\Users\jbattaglia\.conda\pricefm_tf\python.exe",
        help="Python executable for PriceFM generation path.",
    )
    parser.add_argument("--with-production", action="store_true", help="Run production pipeline dry run with skip backtest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    base_env = os.environ.copy()
    path_prefix = ";".join(
        [
            r"C:\Users\jbattaglia\.conda\ppa_env",
            r"C:\Users\jbattaglia\.conda\ppa_env\Library\bin",
            r"C:\Users\jbattaglia\.conda\ppa_env\Scripts",
        ]
    )
    base_env["PATH"] = f"{path_prefix};{base_env.get('PATH', '')}"
    base_env["PYTHONIOENCODING"] = "utf-8"

    payload: dict[str, object] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {},
    }

    payload["checks"]["ast"] = _ast_checks()
    payload["checks"]["eval_futureboost_experimental"] = _run_command(
        [args.python, "scripts/eval_futureboost_experimental.py"],
        env=base_env,
        timeout=300,
    )
    payload["checks"]["eval_futureboost_campaign_baseline"] = _run_command(
        [
            args.python,
            "scripts/eval_futureboost_campaign.py",
            "--mode",
            "baseline",
            "--output",
            "pfc_shaping/output/futureboost_campaign_healthcheck.json",
        ],
        env=base_env,
        timeout=300,
    )

    if args.with_production:
        prod_env = base_env.copy()
        prod_env["PFC_ENABLE_PRICEFM_EXPERIMENT"] = "1"
        prod_env["PFC_APPLY_PRICEFM_EXPERIMENT_TO_PFC"] = "1"
        prod_env["PFC_GENERATE_PRICEFM_EXPERIMENT"] = "1"
        prod_env["PFC_PRICEFM_PYTHON"] = args.pricefm_python
        prod_env["PFC_PRICEFM_BLEND_MODE"] = "futureboost"
        prod_env["PFC_FUTUREBOOST_USE_CIN"] = "0"
        prod_env["PFC_FUTUREBOOST_USE_QRA"] = "1"
        prod_env["PFC_LEAR_BACKTEST_MODE"] = "skip"
        payload["checks"]["run_pfc_production_skip"] = _run_command(
            [args.python, "run_pfc_production.py"],
            env=prod_env,
            timeout=1800,
        )

    all_ok = True
    for check in payload["checks"].values():
        all_ok = all_ok and bool(check.get("ok", False))
    payload["ok"] = all_ok

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.log.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"output:{args.output.resolve()}")
    print(f"ok:{payload['ok']}")
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

