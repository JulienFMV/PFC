"""Build and register one real daily D304 snapshot from explicitly frozen local inputs.

No network access, source discovery, backdated valuation or scheduled process.
The input request binds recipe plus EEX/CH_HISTORY/OMPEX/LSEG artifacts using the
same four-source schema as lt_benchmark_snapshots. EEX is normalized full history,
CH_HISTORY is complete monthly signed targets, external inputs are hourly Parquet.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface, split_solver_quote_maps
from pfc_shaping.pipeline.monthly_curve_authority import select_wholly_undelivered_forward_prices, delivery_months_from_prices, solve_monthly_level_authority
from pfc_shaping.pipeline.production_phases import _solver_delivery_quarter_hour_grid
from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from pfc_shaping.validation.lt_benchmark_snapshots import bound_file, utc, read_registry, append_snapshot
from scripts.run_lt_hourly_stability import training_thresholds
from scripts.run_lt_signed_composition import ROOT, SOURCE, sha, write_json, assembler, save_curve


def preflight(root, request, registry, now):
    if set(request) != {'recipe', 'inputs'} or set(request['inputs']) != {'EEX','CH_HISTORY','OMPEX','LSEG'}:
        raise ValueError('exact daily request schema required')
    recipe_path = bound_file(root, request['recipe'])
    recipe = json.loads(recipe_path.read_text())
    if recipe['model'] != 'signed-equal' or recipe['authority'] != dict(AUTHORITIES) or recipe['external_model_input'] is not False:
        raise ValueError('D304 negative-authority recipe required')
    records, _ = read_registry(root, registry, now=now)
    if len(records) >= 20:
        raise ValueError('twenty-day pilot cap reached')
    if records and (request['recipe'] != records[0]['entry']['recipe'] or
        utc(now).tz_convert('Europe/Zurich').date() <= utc(records[-1]['entry']['valuation_at_utc']).tz_convert('Europe/Zurich').date()):
        raise ValueError('frozen recipe and genuinely later Swiss capture day required')
    paths = {}
    for role, source in request['inputs'].items():
        if set(source) != {'artifact','observed_at_utc','issue_at_utc','vendor_availability_authenticated'}:
            raise ValueError('exact source observation schema required')
        if utc(source['observed_at_utc']) > utc(now) or source['vendor_availability_authenticated'] is not False:
            raise ValueError('local pre-cutoff source observation required')
        if role != 'CH_HISTORY' and utc(source['observed_at_utc']).tz_convert('Europe/Zurich').date() != utc(now).tz_convert('Europe/Zurich').date():
            raise ValueError('daily market/benchmark sources must be re-observed on the capture day')
        if source['issue_at_utc'] is not None and utc(source['issue_at_utc']) > utc(source['observed_at_utc']):
            raise ValueError('source issued after local observation')
        paths[role] = bound_file(root, source['artifact'])
    for name, digest in recipe['source_code_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('frozen model code changed')
    return recipe, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if Path.cwd() != ROOT or not out.is_relative_to(ROOT/'build') or out == ROOT/'build' or out.exists():
        raise ValueError('fresh canonical build output required')
    request_path = args.request.resolve(strict=True)
    if not request_path.is_relative_to(ROOT/'build'):
        raise ValueError('local frozen request required')
    request = json.loads(request_path.read_text())
    origin = pd.Timestamp.now(tz='UTC')
    recipe, paths = preflight(ROOT, request, args.registry, origin)
    target = pd.read_parquet(paths['CH_HISTORY'])
    thresholds = training_thresholds(target, origin)
    history = pd.read_parquet(paths['EEX'])
    if pd.Timestamp(history.date.max()).date() > origin.tz_convert('Europe/Zurich').date():
        raise ValueError('future EEX quotation')
    surface = select_latest_quote_surface(history)
    quotes = split_solver_quote_maps(surface)
    own = dict(quotes['BASE'])
    own.update({key+'-Peak':value for key,value in quotes['PEAK'].items()})
    eligible = select_wholly_undelivered_forward_prices(own, valuation_timestamp=origin, timezone='Europe/Zurich')
    months = delivery_months_from_prices(eligible)
    settings = dict(recipe['monthly_settings'], eex_history_path=str(paths['EEX']))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'plan.json', dict(valuation_at_utc=origin.isoformat(), request=request, thresholds=thresholds,
        request_sha256=sha(request_path), runner_sha256=sha(Path(__file__)), authority=dict(AUTHORITIES)))
    solved = solve_monthly_level_authority(market='CH', delivery_months=months, own_base_prices=own,
        all_market_base_prices={}, eex_history=history, run_timestamp=origin, settings=settings,
        timezone='Europe/Zurich', source_hashes={'current_eex_history':sha(paths['EEX'])}, original_forward_prices=own, allow_unverified_inputs=True)
    write_json(out/'solver.json', dict(monthly_manifest=solved.manifest, assembler_base_prices=solved.assembler_base_prices,
        quoted_keys=sorted(solved.quoted_keys), authority=dict(AUTHORITIES)))
    grid = _solver_delivery_quarter_hour_grid(months, timezone='Europe/Zurich')
    hourly = pd.date_range(grid[0], grid[-1], freq='h')
    raw, _ = calendar_cell_reference(target.signed_target, hourly)
    raw = pd.Series(raw, index=hourly)
    assembly = assembler(HydroAlignedShapeHourlyMLP.load(SOURCE/'fitted-models/hourly.pkl'))
    curve = assembly.build(base_prices=solved.assembler_base_prices, quoted_keys=set(solved.quoted_keys), delivery_index=grid,
        reference_date=origin, country='CH', signed_hourly_shape=center_signed_hourly_shape(raw))
    receipt = save_curve(out/'D304', curve, solved.assembler_base_prices, surface, assembly)
    raw.to_frame('raw').to_parquet(out/'D304/raw.parquet')
    committed = pd.Timestamp.now(tz='UTC')
    # Recheck exact input bytes after construction before registration.
    for source in request['inputs'].values(): bound_file(ROOT, source['artifact'])
    entry = dict(schema='fmv-benchmark-snapshot.v1', valuation_at_utc=origin.isoformat(), candidate_committed_at_utc=committed.isoformat(),
        registered_at_utc=pd.Timestamp.now(tz='UTC').isoformat(), recipe=request['recipe'],
        candidate=dict(path=(out/'D304/curve.parquet').relative_to(ROOT).as_posix(),sha256=sha(out/'D304/curve.parquet')),
        inputs=request['inputs'], authority=dict(AUTHORITIES), evidence_class='LOCAL_OBSERVED_NOT_INDEPENDENTLY_AUTHENTICATED')
    write_json(out/'registry-entry.json', entry)
    write_json(out/'complete.json', dict(status='LOCAL_DAILY_D304_BUILT', receipt=receipt, authority=dict(AUTHORITIES)))
    write_json(out/'manifest.json', {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    registered = append_snapshot(ROOT, args.registry, entry, now=pd.Timestamp.now(tz='UTC'))
    print(json.dumps(dict(status='LOCAL_DAILY_D304_REGISTERED', registry_record=str(registered))))


if __name__ == '__main__':
    main()
