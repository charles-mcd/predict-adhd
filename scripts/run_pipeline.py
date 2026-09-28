"""Run the analysis. Stages follow the order of the notebook.

    python scripts/run_pipeline.py                 # everything
    python scripts/run_pipeline.py --stage final   # one stage

Consensus feature selection is refit inside every training fold, so the
compare stage cannot be cached. The first pass, the
full-sample selection used for interpretation, and the duration feature sets
are saved to data/interim; pass --rebuild to recompute them.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from actigraphy_adhd import data, evaluate, plots, selection
from actigraphy_adhd.models import clf_t_unweighted, small_selectors


def save(df, results, name):
    """Write a results table and print location."""
    df.to_csv(results / f'{name}.csv', index=False)
    print(f'saved {name}.csv')


def build_or_load(path, build, rebuild=False, index_col='CMID'):
    """Load a saved interim file, or build it and save for next time."""
    if path.exists() and not rebuild:
        print(f'loading {path.name}')
        return pd.read_csv(path, index_col=index_col)

    df = build()
    df.to_csv(path)
    print(f'saved {path.name}')
    return df


# ------------------------------------------------------------ load + select

def prepare(cfg, rebuild=False):
    """Load precomputed, build the four feature conditions and select features."""
    cov, hand, auto, auto_ = data.load_precomputed(cfg['precomputed'])

    # first pass runs on each stratification condition
    auto_non = build_or_load(cfg['interim'] / 'auto_non.csv',
                             lambda: selection.first_pass(data.non_stratified(auto)),
                             rebuild)
    auto_str = build_or_load(cfg['interim'] / 'auto_str.csv',
                             lambda: selection.first_pass(data.stratified(auto)),
                             rebuild)

    sets = {
        'hand_non': hand[data.HAND_NON],
        'hand_str': hand[data.HAND_STR],
        'auto_non': auto_non,
        'auto_str': auto_str
    }

    # the compare stage now selects within folds, so the only feature list
    # needed here is the full-sample one, used for the selection matrix, the
    # coefficients reported for interpretation, tuning and the duration study
    path = cfg['interim'] / 'selected_features.json'

    if path.exists() and not rebuild:
        print(f'loading {path.name}')
        fs = json.loads(path.read_text())
    else:
        print('\nauto_str: second pass on the full sample')
        matrix = selection.second_pass(sets['auto_str'], cov)
        matrix.to_csv(cfg['results'] / 'selection_matrix_auto_str.csv')
        fs = {'auto_str': selection.consensus(matrix)}

        path.write_text(json.dumps(fs, indent=2))
        print(f'saved {path.name}')

    return cov, auto_, sets, fs


# --------------------------------------------------------- RQ1/RQ2 results

def compare(cfg, cov, sets, fs):
    """Feature and model analysis across the four conditions."""
    results, roc_scores = [], {}

    for name, df in sets.items():
        print(f'\n{name}:')
        res, scores = evaluate.evaluate(df, cov, selection.condition_selectors[name])
        res.insert(0, 'condition', name)
        results.append(res)
        roc_scores[name] = evaluate.lr_scores(scores)

    save(pd.concat(results), cfg['results'], 'model_comparison')

    # feature set comparison (RQ2)
    print('\nRQ2, tsfresh stratified vs non-stratified:')
    n = len(cov)
    test = evaluate.rq2_test(roc_scores['auto_str'], roc_scores['auto_non'],
                             n * 4 / 5, n / 5)
    save(pd.DataFrame([test]), cfg['results'], 'rq2_test')

    # covariate analysis. Covariates are always kept; the activity features
    # are selected within folds, so the selector applies to the joined frame.
    activity = sets['auto_str']
    cov_sets = {
        # potential causal predictors
        'covariates': (cov[['ETH', 'BMI', 'SEX']], None),

        # with best activity features
        'covariates + auto_str': (activity.join(cov[['ETH', 'BMI', 'SEX']], how='inner'),
                                  selection.condition_selectors['auto_str']),

        # all predictor variables
        'covariates + auto_str + SDQ': (activity.join(cov[['ETH', 'BMI', 'SEX', 'SDQ']], how='inner'),
                                        selection.condition_selectors['auto_str'])
    }

    results = []
    for name, (df, selector) in cov_sets.items():
        print(f'\n{name}:')
        res, _ = evaluate.evaluate(df, cov, selector)
        res.insert(0, 'feature_set', name)
        results.append(res)

    save(pd.concat(results), cfg['results'], 'covariate_models')

    # explore coefficients with the base LR model, fitted on all data
    save(evaluate.get_coefs(cov[['ETH', 'BMI', 'SEX']], cov), cfg['results'], 'coefs_covariates')
    save(evaluate.get_coefs(sets['auto_str'][fs['auto_str']].join(
        cov[['ETH', 'BMI', 'SEX']], how='inner'), cov),
        cfg['results'], 'coefs_covariates_auto_str')


# ---------------------------------------------------------- tuning + final

def tuning(cfg, cov, sets, fs):
    """Hyperparameter tuning on selected model and featureset."""
    save(evaluate.tune(sets['auto_str'][fs['auto_str']], cov), cfg['results'], 'tuning')


def final(cfg, cov, sets, fs):
    """Finalised pipeline: predictions, curves, thresholds, coefficients."""
    # best condition from the comparison, features selected within folds
    best_fs = fs['auto_str']
    selector = selection.condition_selectors['auto_str']
    X, y, oof_probs = evaluate.out_of_fold(sets['auto_str'], cov, selector)

    plots.roc(y, oof_probs, cfg['results'] / 'roc_curve.png')
    plots.pr(y, oof_probs, cfg['results'] / 'pr_curve.png')

    threshold_df = evaluate.threshold_analysis(y, oof_probs)
    save(threshold_df, cfg['results'], 'threshold_analysis')
    plots.sens_spec(threshold_df, cfg['results'] / 'sens_spec_tradeoff.png')

    print("\nBest Balanced Accuracy:")
    print(threshold_df.sort_values("balanced_accuracy", ascending=False).head(10).to_string(index=False))

    print("\nBest specificity while sensitivity >= 0.80:")
    print(threshold_df[threshold_df["sensitivity"] >= 0.80]
          .sort_values("specificity", ascending=False).head(10).to_string(index=False))

    print("\nBest sensitivity while specificity >= 0.80:")
    print(threshold_df[threshold_df["specificity"] >= 0.80]
          .sort_values("sensitivity", ascending=False).head(10).to_string(index=False))

    plots.predicted_probs(y, oof_probs, cfg['results'] / 'predicted_probs.png',
                          cfg['threshold'])

    # unweighted variant on the same folds, used by the subgroup audit
    print('\nunweighted variant:')
    _, _, oof_unweighted = evaluate.out_of_fold(sets['auto_str'], cov, selector,
                                                clf_t_unweighted)

    pd.DataFrame({
        "y": y,
        "prob": oof_probs,
        "prob_unweighted": oof_unweighted,
        "sex": cov.SEX.loc[X.index].to_numpy()
    }, index=X.index).to_csv(cfg['interim'] / 'final_predictions.csv')
    print('saved final_predictions.csv')

    # save coefficients for best feature set, fitted on all data
    save(evaluate.get_coefs(sets['auto_str'][best_fs], cov), cfg['results'], 'coefs')


# ---------------------------------------------------------------- subgroup

def subgroup(cfg, cov, sets, fs):
    """SEX sensitivity analysis."""
    # preliminary check on statistical stability
    df = sets['auto_str'].join(cov, how='inner')
    print('non-white:\n', df[df.ETH.eq('Non-white')].ADHD.value_counts())
    print('\nmale:\n', df[df.SEX.eq('Male')].ADHD.value_counts())
    print('\nfemale:\n', df[df.SEX.eq('Female')].ADHD.value_counts())

    # v1 refit the model with SEX added as a predictor, which changes both the
    # folds and the model being audited. Here the final model's own out-of-fold
    # predictions are split by sex, using the unweighted variant because class
    # weighting shifts the probabilities that calibration is measured on.
    path = cfg['interim'] / 'final_predictions.csv'

    if not path.exists():
        raise FileNotFoundError('run the final stage first')

    pred_df = pd.read_csv(path, index_col='CMID')
    pred_df = pred_df[['y', 'prob_unweighted', 'sex']].rename(
        columns={'prob_unweighted': 'prob'})

    # drop unknown SEX
    pred_df = pred_df[~pred_df.sex.eq('Unknown')]

    groups = {
        'All': pred_df,
        'Female': pred_df[pred_df.sex.eq('Female')],
        'Male': pred_df[pred_df.sex.eq('Male')]
    }

    res = evaluate.subgroup_metrics(groups, cfg['threshold'])
    print(res.to_string(index=False))
    save(res, cfg['results'], 'subgroup_sex')

    # bootstrap interval for each sex's AUROC and the difference between them
    ci = evaluate.subgroup_auroc_ci(pred_df, cfg['n_boot_sub'])
    print(ci.to_string(index=False))
    save(ci, cfg['results'], 'subgroup_auroc_ci')

    # threshold selection EDA
    save(evaluate.subgroup_thresholds(groups, cfg['thresholds']),
         cfg['results'], 'subgroup_thresholds')

    plots.calibration(groups, cfg['results'] / 'subgroup_calibration.png')


# ---------------------------------------------------------------- duration

def duration(cfg, cov, auto_, fs, rebuild=False):
    """Monitoring period analysis."""
    best_fs = fs['auto_str']
    days, df = data.duration_sample(auto_)

    # build each duration's feature set
    dfs = {}
    for d, build in data.DURATIONS.items():
        print(f'\n{d} day duration:')
        dfs[d] = build_or_load(cfg['interim'] / f'duration_{d}.csv',
                               lambda d=d, build=build: build(df, days), rebuild)

    # get OOF preds for each duration, all aligned to the 6 day baseline
    print('\n6 day baseline:')
    X_base, y, probs_6 = evaluate.out_of_fold(dfs[6].sort_index()[best_fs], cov)
    idx_order = X_base.index

    preds = pd.DataFrame({'CMID': idx_order, 'y': y}).set_index('CMID')
    preds['probs_6'] = probs_6

    for d, X in dfs.items():
        if d == 6:
            continue

        print(f'\n{d} day duration:')

        # align rows to baseline, selected features
        _, _, probs = evaluate.out_of_fold(X.loc[idx_order, best_fs], cov)
        preds[f'probs_{d}'] = probs

    summary = evaluate.duration_bootstrap(preds, cfg['n_boot'])
    print(summary.to_string(index=False))
    save(summary, cfg['results'], 'duration_comparison')
    plots.duration_auroc(summary, cfg['results'] / 'duration_auroc.png')


# -------------------------------------------------------------------- main

STAGES = ['compare', 'tuning', 'final', 'subgroup', 'duration']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--stage', nargs='+', choices=STAGES + ['all'], default=['all'])
    parser.add_argument('--quick', action='store_true', help='small bootstrap, separate results folder')
    parser.add_argument('--rebuild', action='store_true', help='ignore saved interim files')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(args.config).resolve().parent
    cfg['precomputed'] = root / cfg['paths']['precomputed']
    cfg['interim'] = root / cfg['paths']['interim']
    cfg['results'] = root / cfg['paths']['results']

    if args.quick:
        cfg['n_boot'] = 200
        cfg['results'] = cfg['results'].with_name(cfg['results'].name + '_quick')

    cfg['interim'].mkdir(parents=True, exist_ok=True)
    cfg['results'].mkdir(parents=True, exist_ok=True)

    cov, auto_, sets, fs = prepare(cfg, args.rebuild)

    stages = STAGES if 'all' in args.stage else args.stage

    if 'compare' in stages:
        compare(cfg, cov, sets, fs)
    if 'tuning' in stages:
        tuning(cfg, cov, sets, fs)
    if 'final' in stages:
        final(cfg, cov, sets, fs)
    if 'subgroup' in stages:
        subgroup(cfg, cov, sets, fs)
    if 'duration' in stages:
        duration(cfg, cov, auto_, fs, args.rebuild)

    print(f'\ndone, results in {cfg["results"]}')


if __name__ == '__main__':
    main()
