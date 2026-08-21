"""Checkout wrapper for the sealed market-time-regime auditor.

The authoritative operator route is the installed module under ``python -I
-B -m``.  This wrapper only provides a deterministic, origin-checked failure
from a source checkout; it cannot bypass sealed-runtime admission.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _bootstrap() -> object:
    script = Path(__file__).resolve(strict=True)
    root = script.parents[1]
    cwd = Path.cwd().resolve(strict=True)
    if os.path.normcase(str(cwd)) != os.path.normcase(str(root)):
        raise ValueError("checkout wrapper cwd must equal its repository root")
    sys.path.insert(0, str(root))
    from pfc_shaping.cli import audit_ch_market_time_regime as installed_cli

    expected = root / "pfc_shaping" / "cli" / "audit_ch_market_time_regime.py"
    observed = Path(installed_cli.__file__).resolve(strict=True)
    if os.path.normcase(str(observed)) != os.path.normcase(str(expected)):
        raise ValueError("checkout wrapper imported a noncanonical CLI module")
    return installed_cli


def main(argv: list[str] | None = None) -> int:
    try:
        installed_cli = _bootstrap()
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_CHECKOUT_WRAPPER_ORIGIN",
                    "error": str(exc),
                    "execution_authorized": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    return installed_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
