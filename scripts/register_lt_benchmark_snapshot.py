"""Append one explicitly supplied local benchmark snapshot; no network or schedule."""
import argparse
import json
from pathlib import Path
import pandas as pd
from pfc_shaping.validation.lt_benchmark_snapshots import append_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise ValueError('canonical workspace required')
    entry_path = args.entry.resolve(strict=True)
    if not entry_path.is_relative_to(root/'build'):
        raise ValueError('entry must remain below build')
    path = append_snapshot(root, args.registry, json.loads(entry_path.read_text()), now=pd.Timestamp.now(tz='UTC'))
    print(json.dumps(dict(status='LOCAL_SNAPSHOT_REGISTERED_NO_AUTHORITY', path=str(path))))


if __name__ == '__main__':
    main()
