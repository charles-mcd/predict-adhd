"""Build the precomputed tables from the raw MCS files.

    python scripts/prepare_data.py                 # everything
    python scripts/prepare_data.py --stage auto    # one stage

Run once. The analysis (scripts/run_pipeline.py) reads the output of this.

Stages and their cost:

    cov       labels, covariates and exclusions        seconds
    hand      handcrafted features from MCS aggregates seconds
    ts        .dat files to valid-day timeseries       ~20 min, ~7GB memory
    tsfresh   feature extraction per day               ~15 hours, ~10GB memory
    auto      concat the tsfresh chunks                minutes

Output goes to data/precomputed: cov.csv, hand.csv, auto.csv, plus the
interim ts.csv, tsF.csv and chunks/ which the later stages reload. The
tsfresh stage skips chunks that already exist, so it can be resumed.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from actigraphy_adhd import extract, features, timeseries

STAGES = ['cov', 'hand', 'ts', 'tsfresh', 'auto']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--stage', nargs='+', choices=STAGES + ['all'], default=['all'])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(args.config).resolve().parent
    raw = root / cfg['paths']['raw']
    out = root / cfg['paths']['precomputed']
    dat_folder = raw / 'dat'
    chunk_folder = out / 'chunks'
    out.mkdir(parents=True, exist_ok=True)

    stages = STAGES if 'all' in args.stage else args.stage

    # class labels, valid activity sample and covariates
    print('\nreading parent interviews and activity aggregates')
    pci = extract.parent_interviews(raw)
    adhd = extract.adhd_ids(pci)
    sw4_day, val = extract.valid_days(raw)
    labels = extract.class_labels(val, adhd)

    if 'cov' in stages:
        cov = extract.covariates(raw, pci, labels)
        cov = extract.with_dat_files(cov, dat_folder)
        cov.to_csv(out / 'cov.csv')
        print('saved cov.csv')
    else:
        cov = pd.read_csv(out / 'cov.csv', index_col='CMID')

    if 'hand' in stages:
        print('\nhandcrafted features')
        features.handcrafted(val, cov).to_csv(out / 'hand.csv')
        print('saved hand.csv')

    if 'ts' in stages:
        print('\npreparing raw timeseries (~20 min, ~7GB)')
        ts, dropped = timeseries.build_timeseries(cov, sw4_day, dat_folder)
        ts.to_csv(out / 'ts.csv', index=False, chunksize=100_000)
        print('saved ts.csv')

    if 'tsfresh' in stages:
        print('\ntsfresh extraction (~15 hours, ~10GB)')
        ts = pd.read_csv(out / 'ts.csv')
        tsF = timeseries.day_ids(ts)
        tsF.to_csv(out / 'tsF.csv')
        timeseries.extract_chunks(tsF, chunk_folder)

    if 'auto' in stages:
        print('\nconcatenating tsfresh chunks')
        timeseries.concat_chunks(chunk_folder).to_csv(out / 'auto.csv')
        print('saved auto.csv')

    print(f'\ndone, precomputed tables in {out}')


if __name__ == '__main__':
    main()
