#!/usr/bin/env python3
"""Capture a caller-held manifest before first execution of a Python prefix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.build_launcherless_local_runtime import build_python_runtime_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_python_runtime_manifest(
        runtime_prefix=args.runtime_prefix,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"INTEGRITY_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
