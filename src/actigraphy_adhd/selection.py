"""Feature reduction and selection."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import clf, preprocess, selectors, small_selectors

# increasing n_consensus (the number of selectors selecting for that feature)
# explores stricter, more reduced sets
N_CONSENSUS = 2


class ConsensusSelector(SelectorMixin, BaseEstimator):
    """Second pass consensus as a transformer, fitted on training folds only.

    v1 ran the consensus once on the whole sample and passed the resulting
    feature list into cross-validation. Wrapping it as a scikit-learn
    transformer puts it in the pipeline alongside the small-set selectors,
    which were already refit per fold.
    """

    def __init__(self, selectors, n_consensus=N_CONSENSUS):
        self.selectors = selectors
        self.n_consensus = n_consensus

    def fit(self, X, y):
        votes = np.zeros(X.shape[1], dtype=int)

        for selector in self.selectors.values():
            votes += clone(selector).fit(X, y).get_support().astype(int)

        self.votes_ = votes
        self.support_ = votes >= self.n_consensus
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        return self.support_


# selector used for each feature condition, refit within every training fold
condition_selectors = {
    'hand_non': small_selectors['mi_k5'],
    'hand_str': small_selectors['f_classif_k5'],
    'auto_non': ConsensusSelector(selectors),
    'auto_str': ConsensusSelector(selectors)
}


def first_pass(auto):
    """First pass feature reduction for auto."""
    # drop constant value (inc. NaNs) columns
    const = auto.nunique()[auto.nunique() < 2]
    auto = auto.drop(columns=const.index)

    # apply correlation filter to reduce based on collinearity
    corr = auto.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if (upper[col] > 0.95).any()]
    auto = auto.drop(columns=drop_cols)

    print(auto.shape)
    return auto


def second_pass(auto, cov):
    """Second pass feature reduction for auto. Returns the selection matrix."""
    X, y, preprocessor = preprocess(auto, cov)

    selected_features = {}

    for name, selector in selectors.items():
        print(name, '...')
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ("scale", StandardScaler()),
            ("select", selector),
            ("clf", clf)
        ])

        pipe.fit(X, y)

        pre = pipe.named_steps["preprocessor"]
        selector = pipe.named_steps["select"]

        feature_names = pre.get_feature_names_out()
        support = selector.get_support()

        selected_features[name] = feature_names[support].tolist()

    # feature consensus across selection methods
    selected_sets = {
        name: set(vals)
        for name, vals in selected_features.items()
    }

    all_selected = sorted(set().union(*selected_sets.values()))

    selection_matrix = pd.DataFrame({
        name: [feature in selected_sets[name] for feature in all_selected]
        for name in selected_sets
    }, index=all_selected)

    selection_matrix["n_methods"] = selection_matrix.sum(axis=1)
    selection_matrix = selection_matrix.sort_values("n_methods", ascending=False)

    selection_matrix.index = selection_matrix.index.str[6:]

    return selection_matrix


def consensus(selection_matrix, n_consensus=N_CONSENSUS):
    """Selected feature set based on consensus."""
    fs = selection_matrix[selection_matrix.n_methods >= n_consensus].index.tolist()
    print(len(fs), 'features')
    return fs


def compare_selectors(df, cov):
    """Feature select - small sets.

    More intensive, use on handcrafted and already reduced automatic features.
    """
    X, y, preprocessor = preprocess(df, cov)

    # CV
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    results = []

    for name, selector in small_selectors.items():
        print('selector: ', name)
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ("scale", StandardScaler()),
            ("select", selector),
            ("clf", clf)
        ])

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

        results.append({
            "selector": name,
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_sd": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
            "sens_mean": scores["test_sensitivity"].mean(),
            "spec_mean": scores["test_specificity"].mean()
        })

    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)


def selected_by(df, cov, selector):
    """Features kept by one small-set selector, used for the handcrafted sets."""
    X, y, preprocessor = preprocess(df, cov)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ("scale", StandardScaler()),
        ("select", selector)
    ])

    pipe.fit(X, y)

    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    support = pipe.named_steps["select"].get_support()

    return [name[6:] for name in feature_names[support]]
