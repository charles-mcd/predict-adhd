"""Preprocessing, selector models and classifiers."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       f_classif, mutual_info_classif)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def preprocess(df, cov):
    # intersect activity features with covariates and seperate class
    X = df.join(cov.ADHD, how='inner')
    y = LabelEncoder().fit_transform(X.pop('ADHD'))

    # convert bool to int
    for col in X.select_dtypes('bool').columns:
        X[col] = X[col].astype('int8')

    # setup preprocessor
    nums = X.select_dtypes('number').columns
    cats = X.columns.difference(nums)

    num_trans = SimpleImputer(strategy='median', add_indicator=False)

    cat_trans = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        # ('enc', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))            # EDA found preferable results compared with One-hot
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('nums', num_trans, nums),
        ('cats', cat_trans, cats)
    ])

    print('ADHD distribution:\n', pd.Series(y).value_counts())

    return X, y, preprocessor


# eval classifier
clf = LogisticRegression(
    l1_ratio=0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=5000,
    random_state=42
)

# selector models
l1_selector = LogisticRegression(
    l1_ratio=1,
    solver="liblinear",
    C=0.1,
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

elastic_selector = LogisticRegression(
    solver="saga",
    C=0.1,
    l1_ratio=0.5,
    class_weight="balanced",
    max_iter=10000,
    random_state=42
)

et_selector = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# second pass feature reduction selectors
# mutual information is stochastic and is now refit in every fold, so a seed is
# passed to keep runs repeatable
selectors = {
    "MI_top_50": SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=50),
    "f_classif_top_50": SelectKBest(score_func=f_classif, k=50),
    "L1_SelectFromModel": SelectFromModel(l1_selector),
    "ElasticNet_SelectFromModel": SelectFromModel(elastic_selector),
    "ExtraTrees_SelectFromModel": SelectFromModel(et_selector, threshold="median")
}

# small set selectors (more intensive, use on handcrafted and already reduced automatic features)
small_selectors = {
    "none": "passthrough",
    "mi_k5": SelectKBest(
        lambda X, y: mutual_info_classif(X, y, random_state=42), k=5),
    "f_classif_k5": SelectKBest(f_classif, k=5),
    "f_classif_k10": SelectKBest(f_classif, k=10),
    "rfe_k5": RFE(clf, n_features_to_select=5, step=1),
    "l1_selector": SelectFromModel(
        LogisticRegression(
            l1_ratio=1,
            solver="liblinear",
            class_weight="balanced",
            C=0.1,
            max_iter=5000
        )
    ),
    "elastic_selector": SelectFromModel(
        LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            C=0.1,
            max_iter=5000
        )
    )
}

# Models
models = {
    'dummy':
    DummyClassifier(strategy="prior"),

    'LR':
    LogisticRegression(
    l1_ratio=0.5,               # elasticNet balanced L1/L2 penalty, allow some sparsity for high-dimensional sets
    class_weight="balanced",
    solver="saga",              # required solver for elasticNet
    max_iter=5000
    ),

    'NB':
    GaussianNB(priors=None, var_smoothing=1e-09),

    'ET': ExtraTreesClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    ## explored during EDA ##
    # SVC (linear, C=0.1), RandomForest (500 trees, max_depth 3) and
    # XGBClassifier (max_depth 2, lr 0.03) - mediocre results for excessive
    # training times, see thesis 3.3.4
}

# base model for hyperparameter tuning
base_lr = LogisticRegression(
    solver="saga",
    max_iter=10000,
    random_state=42
)

# tuned model
clf_t = LogisticRegression(
            C=0.1,
            l1_ratio=0,
            class_weight='balanced',
            solver="saga",
            max_iter=10000,
            random_state=42
)

# unweighted variant of the tuned model, used for the subgroup audit: class
# weighting shifts predicted probabilities, which calibration depends on
clf_t_unweighted = LogisticRegression(
            C=0.1,
            l1_ratio=0,
            class_weight=None,
            solver="saga",
            max_iter=10000,
            random_state=42
)
