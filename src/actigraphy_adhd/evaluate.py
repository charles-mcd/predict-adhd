"""Evaluation: model comparison, coefficients, thresholds, subgroups, durations."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from scipy.stats import t as stats_t
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, make_scorer, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import base_lr, clf_t, clf_t_unweighted, models, preprocess


def build_pipe(preprocessor, model, selector=None):
    """Preprocess, scale, optionally select, then the model.

    The selector sits inside the pipeline so cross_validate and
    cross_val_predict refit it on each training fold.
    """
    steps = [('preprocessor', preprocessor), ("scale", StandardScaler())]

    if selector is not None:
        steps.append(("select", clone(selector)))

    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(df, cov, selector=None):
    """Exploratory Eval."""
    X, y, preprocessor = preprocess(df, cov)

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    results = []
    roc_scores = []

    for name, model in models.items():
        print(f'{name}...')
        pipe = build_pipe(preprocessor, model, selector)

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision",
                "sensitivity": make_scorer(recall_score, pos_label=1),
                "specificity": make_scorer(recall_score, pos_label=0)
            },
            n_jobs=-1
        )

        roc_scores.append({'model': name,
                           'scores': scores["test_roc_auc"]
        })

        results.append({
            "model": name,
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_sd": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
            "sens_mean": scores["test_sensitivity"].mean(),
            "spec_mean": scores["test_specificity"].mean()
        })

    print('sensitivity and specificity at default threshold 0.5')
    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False), roc_scores


def lr_scores(roc_scores):
    """Pull the LR fold scores out of the evaluate() output.

    The notebook indexed this as evaluate(...)[1]['scores'], which does not
    work on a list of dicts; this looks up the model by name instead.
    """
    return next(r['scores'] for r in roc_scores if r['model'] == 'LR')


def rq2_test(strat, non, n_train, n_test):
    """Feature set comparison (RQ2).

    Run with best non- and best stratified features sets (both from auto).
    RQ1 is reported descriptively and is not significance tested.

    Folds from repeated k-fold share training data, so their scores are not
    independent and Wilcoxon is too liberal. The corrected resampled t-test
    inflates the variance by n_test/n_train to account for the overlap
    (Nadeau & Bengio, 2003). Wilcoxon is reported alongside.
    """
    diff = strat - non
    j = len(diff)
    var = diff.var(ddof=1)

    t_stat = diff.mean() / np.sqrt((1 / j + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats_t.sf(abs(t_stat), df=j - 1)

    stat, wilcox_p = wilcoxon(diff)
    print(f"corrected resampled t-test p = {p:.4f} (Wilcoxon p = {wilcox_p:.4f})")

    return {"mean_diff": diff.mean(), "t": t_stat, "df": j - 1,
            "p": p, "wilcoxon_statistic": stat, "wilcoxon_p": wilcox_p}


def get_coefs(df, cov):
    """Return coeffs."""
    X, y, preprocessor = preprocess(df, cov)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ("scale", StandardScaler()),
        ("model", models['LR'])
    ])

    pipe.fit(X, y)

    coefs = pipe.named_steps["model"].coef_[0]
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()

    coef_final = (
        pd.DataFrame({
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs)
        })
        .sort_values("abs_coef", ascending=False)
    )

    return coef_final


def tune(df, cov):
    """Hyperparameter tuning on selected model and featureset."""
    X, y, preprocessor = preprocess(df, cov)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("scale", StandardScaler()),
        ("clf", base_lr)
    ])

    param_grid = [
        # Ridge-like LR for stability across many correlated TSFresh features
        {
            "clf__l1_ratio": [0.0],
            "clf__C": [0.03, 0.1, 0.3, 1.0],
            "clf__class_weight": ["balanced", None],
        },

        # Elastic-net like for greater sparsity
        {
            "clf__l1_ratio": [0.2, 0.5],
            "clf__C": [0.03, 0.1, 0.3],
            "clf__class_weight": ["balanced"],
        },
    ]

    cv_tune = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        refit="roc_auc",
        cv=cv_tune,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid.fit(X, y)

    print("Best params:", grid.best_params_)
    print("Best CV AUROC:", grid.best_score_)

    results = (
        pd.DataFrame(grid.cv_results_)
        .sort_values("rank_test_roc_auc")
    )

    return results[
        [
            "params",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_pr_auc",
            "std_test_pr_auc",
            "mean_train_roc_auc",
            "mean_train_pr_auc",
        ]
    ]


def out_of_fold(df, cov, selector=None, model=None):
    """Get out-of-fold predicted probabilities from the tuned model."""
    X, y, preprocessor = preprocess(df, cov)

    pipe = build_pipe(preprocessor, model if model is not None else clf_t, selector)

    # predictions require non-repeated cv
    oof_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    oof_probs = cross_val_predict(
        clone(pipe),
        X,
        y,
        cv=oof_cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    return X, y, oof_probs


def threshold_analysis(y, oof_probs):
    """Build the threshold table across the range of predicted probabilities."""
    thresholds = np.linspace(oof_probs.min(), oof_probs.max(), 300)
    rows = []

    for t in thresholds:
        pred = oof_probs >= t
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        rows.append({
            "threshold": t,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    return pd.DataFrame(rows)


def subgroup_metrics(groups, threshold):
    """Calculate metrics for each sex and overall."""
    import statsmodels.api as sm

    rows = []

    for group, df in groups.items():

        y_true = np.asarray(df.y)
        probs = np.asarray(df.prob)

        # calibration slope, and calibration-in-the-large: the intercept with
        # the slope held at 1, fitted as an offset (Van Calster et al., 2019).
        # v1 read the intercept off the same model as the slope.
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        logit_p = np.log(probs / (1 - probs))
        X = sm.add_constant(logit_p)
        model = sm.Logit(y_true, X).fit(disp=False)
        cil = sm.Logit(y_true, np.ones(len(y_true)), offset=logit_p).fit(disp=False)

        # confusion matix
        pred = probs >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

        rows.append({
            "group": group,
            "prevalence": int(y_true.sum()) / len(y_true),
            "roc_auc": roc_auc_score(y_true, probs),
            "pr_auc": average_precision_score(y_true, probs),
            "brier": brier_score_loss(y_true, probs),
            "calibration_intercept": cil.params[0],
            "calibration_slope": model.params[1],
            "threshold": threshold,
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    return pd.DataFrame(rows)


def subgroup_auroc_ci(pred_df, n_boot=2000):
    """Bootstrap interval for each sex's AUROC and for the difference.

    Resampled within each group, so the interval reflects how few positive
    cases there are among females.
    """
    rng = np.random.default_rng(42)
    draws = {}

    for group in ('Female', 'Male'):
        df = pred_df[pred_df.sex.eq(group)]
        y_true, probs = df.y.to_numpy(), df.prob.to_numpy()
        scores = []

        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), len(y_true))

            if y_true[idx].min() == y_true[idx].max():
                scores.append(np.nan)
            else:
                scores.append(roc_auc_score(y_true[idx], probs[idx]))

        draws[group] = np.array(scores)

    diff = draws['Female'] - draws['Male']

    rows = [{"group": g, "auroc_ci_low": np.nanpercentile(v, 2.5),
             "auroc_ci_high": np.nanpercentile(v, 97.5)} for g, v in draws.items()]
    rows.append({"group": "Female - Male",
                 "auroc_ci_low": np.nanpercentile(diff, 2.5),
                 "auroc_ci_high": np.nanpercentile(diff, 97.5)})

    return pd.DataFrame(rows)


def subgroup_thresholds(groups, thresholds):
    """Threshold selection EDA across the subgroups."""
    threshold_rows = []

    for group, df in groups.items():

        y_true = np.asarray(df.y)
        probs = np.asarray(df.prob)

        for t in thresholds:

            # confusion matix
            pred = probs >= t
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan

            threshold_rows.append({
                "group": group,
                "threshold": t,
                "sensitivity": sens,
                "specificity": spec,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                "balanced_accuracy": (sens + spec) / 2,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            })

    return pd.DataFrame(threshold_rows)


def metrics(df, col):
    """Duration metrics against the 6 day reference."""
    y = df["y"].to_numpy()
    p = df[col].to_numpy()
    p_ref = df["probs_6"].to_numpy()

    auc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)

    return {
        "roc_auc": auc,
        "pr_auc": pr,
        "delta_auc_vs_6": auc - roc_auc_score(y, p_ref),
        "delta_pr_auc_vs_6": pr - average_precision_score(y, p_ref),
        "spearman_vs_6": spearmanr(p, p_ref).correlation,
    }


def duration_bootstrap(preds, n_boot=5000):
    """Bootstrapped duration comparison."""

    # paired bootstrap
    boot_rows = []
    rng = np.random.default_rng(42)
    duration_cols = {2: "probs_2", 3: "probs_3", 4: "probs_4", 5: "probs_5", 6: "probs_6"}

    for _ in range(n_boot):
        sample = preds.iloc[rng.integers(0, len(preds), len(preds))]

        if sample["y"].nunique() < 2:
            continue

        for days, col in duration_cols.items():
            boot_rows.append({"days": days, **metrics(sample, col)})

    boot = pd.DataFrame(boot_rows)

    # observed metrics
    summary = pd.DataFrame([
        {"days": days, **metrics(preds, col)}
        for days, col in duration_cols.items()
    ])

    # CIs
    for metric in ["delta_auc_vs_6", "delta_pr_auc_vs_6", "roc_auc", "pr_auc"]:
        ci = (
            boot
            .groupby("days")[metric]
            .quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: f"{metric}_ci_low", 0.975: f"{metric}_ci_high"})
        )

        summary = summary.merge(ci, on="days", how="left")

    # bootstrap p-values
    for metric in ["delta_auc_vs_6", "delta_pr_auc_vs_6"]:
        pvals = (
            boot[boot["days"] != 6]
            .groupby("days")[metric]
            .apply(lambda x: min(1, 2 * min((x <= 0).mean(), (x >= 0).mean())))
        )

        # Holm adjustment
        p_col = metric.replace("delta_", "").replace("_vs_6", "_p_holm")

        summary.loc[summary["days"] != 6, p_col] = multipletests(
            pvals,
            method="holm"
        )[1]

    # set column order
    summary = summary[
        [
            "days",
            "roc_auc",
            "roc_auc_ci_low",
            "roc_auc_ci_high",
            "delta_auc_vs_6",
            "delta_auc_vs_6_ci_low",
            "delta_auc_vs_6_ci_high",
            "auc_p_holm",
            "pr_auc",
            "pr_auc_ci_low",
            "pr_auc_ci_high",
            "delta_pr_auc_vs_6",
            "delta_pr_auc_vs_6_ci_low",
            "delta_pr_auc_vs_6_ci_high",
            "pr_auc_p_holm",
            "spearman_vs_6",
        ]
    ].sort_values("days")

    return summary
