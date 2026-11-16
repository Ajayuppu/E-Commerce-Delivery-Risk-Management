"""Learned models: logistic regression (simple baseline) and LightGBM (gradient boosting)."""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import lightgbm as lgb
from . import config
from .features import NUM_FEATURES, CAT_FEATURES

# heavy-tailed columns get log1p before scaling in the linear model
LOG_COLS = ["approve_delay_hours", "total_price", "total_freight", "max_item_price", "total_weight_g",
            "total_volume_cm3", "max_weight_g", "payment_value", "distance_km", "seller_prior_orders",
            "seller_orders_7d", "seller_prior_outcomes", "seller_open_orders", "seller_overdue_open",
            "customer_prior_orders", "market_overdue_open", "market_orders_7d", "seller_days_active"]


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_logreg(C=1.0, class_weight=None, num_features=NUM_FEATURES, cat_features=CAT_FEATURES):
    log_cols = [c for c in num_features if c in LOG_COLS]
    lin_cols = [c for c in num_features if c not in LOG_COLS]
    pre = ColumnTransformer([
        ("log", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("log", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())]), log_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), lin_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50))]), cat_features),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=20000, tol=1e-8,
                                                              solver="lbfgs", random_state=config.SEED))])


def to_lgb_frame(df, cat_levels=None, num_features=NUM_FEATURES, cat_features=CAT_FEATURES):
    X = df[num_features + cat_features].copy()
    levels = cat_levels or {c: sorted(X[c].dropna().unique().tolist()) for c in cat_features}
    for c in cat_features:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    return X, levels


def make_lgbm(**params):
    base = dict(objective="binary", learning_rate=0.03, n_estimators=400, num_leaves=31, min_child_samples=100,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=config.SEED,
                n_jobs=-1, verbose=-1, deterministic=True, force_col_wise=True, cat_smooth=20, min_data_per_group=100)
    base.update(params)
    return lgb.LGBMClassifier(**base)


LOGREG_GRID = [dict(C=c, class_weight=w) for c in (0.01, 0.1, 1.0) for w in (None, "balanced")]
LGBM_GRID = [dict(num_leaves=nl, n_estimators=ne, min_child_samples=mcs, learning_rate=0.03)
             for nl in (7, 15, 31) for ne in (200, 500) for mcs in (100, 400)]
# feature sets compared during model selection (validation only)
FEATURE_SETS = {
    "all": NUM_FEATURES,
    "no_market": [c for c in NUM_FEATURES if not c.startswith("market_")],
}


class FittedModel:
    """Small wrapper so every model exposes .score(df) -> risk scores."""
    def __init__(self, kind, params, num_features=NUM_FEATURES, cat_features=CAT_FEATURES):
        self.kind, self.params = kind, params
        self.num_features, self.cat_features = list(num_features), list(cat_features)
        self.model, self.levels = None, None

    def fit(self, df, y):
        if self.kind == "logreg":
            self.model = make_logreg(num_features=self.num_features, cat_features=self.cat_features, **self.params)
            self.model.fit(df[self.num_features + self.cat_features], y)
        else:
            X, self.levels = to_lgb_frame(df, num_features=self.num_features, cat_features=self.cat_features)
            self.model = make_lgbm(**self.params)
            self.model.fit(X, y)
        return self

    def score(self, df):
        if self.kind == "logreg":
            return self.model.predict_proba(df[self.num_features + self.cat_features])[:, 1]
        X, _ = to_lgb_frame(df, self.levels, self.num_features, self.cat_features)
        return self.model.predict_proba(X)[:, 1]
