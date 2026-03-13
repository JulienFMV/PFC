"""
check_ssl_bundle.py
-------------------
Quick SSL diagnostics for corporate CA bundle configuration.

Examples:
  python -m pfc_shaping.tools.check_ssl_bundle --from-config
  python -m pfc_shaping.tools.check_ssl_bundle --ca-bundle "H:\\path\\corp_root.pem"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import yaml

DEFAULT_URLS = [
    "https://api.energy-charts.info/price?bzn=CH&start=2026-03-01&end=2026-03-02",
    "https://www.smard.de/app/chart_data/259/DE/index_hour.json",
    "https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv",
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_bundle_from_config() -> str | None:
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ssl_cfg = cfg.get("ssl", {}) or {}
    bundle = ssl_cfg.get("ca_bundle")
    if not bundle:
        return None
    p = Path(bundle)
    if p.is_absolute():
        return str(p)
    return str((CONFIG_PATH.parent / p).resolve())


def _check_one(url: str, verify: bool | str, timeout: int) -> tuple[bool, str]:
    try:
        r = requests.get(url, timeout=timeout, verify=verify)
        return True, f"HTTP {r.status_code}"
    except requests.exceptions.SSLError as e:
        return False, f"SSL_ERROR: {e}"
    except requests.exceptions.RequestException as e:
        return False, f"REQUEST_ERROR: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SSL connectivity for PFC data sources")
    parser.add_argument("--ca-bundle", help="Path to PEM/CRT bundle for requests verify")
    parser.add_argument("--from-config", action="store_true", help="Use ssl.ca_bundle from config.yaml")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--url", action="append", dest="urls", help="Optional custom URL (can repeat)")
    args = parser.parse_args()

    verify: bool | str = True
    bundle = args.ca_bundle
    if args.from_config:
        bundle = _load_bundle_from_config()

    if bundle:
        bundle_path = Path(bundle)
        if not bundle_path.exists():
            print(f"[ERROR] CA bundle not found: {bundle_path}")
            return 2
        verify = str(bundle_path)
        print(f"[INFO] Using CA bundle: {bundle_path}")
    else:
        print("[INFO] Using default system trust store (no custom CA bundle)")

    urls = args.urls or DEFAULT_URLS
    print(f"[INFO] Testing {len(urls)} endpoint(s)")

    ok_count = 0
    for url in urls:
        ok, msg = _check_one(url, verify=verify, timeout=args.timeout)
        status = "OK" if ok else "KO"
        print(f"[{status}] {url}")
        print(f"      {msg}")
        if ok:
            ok_count += 1

    print(f"[SUMMARY] {ok_count}/{len(urls)} successful")
    return 0 if ok_count == len(urls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
