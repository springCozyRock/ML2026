"""Fold-fitted Lasso plus a fixed pseudo-Huber residual booster.

The evaluator supplies only training labels and blends the returned log prices.
No label-dependent iteration selection or cached prediction access is used here.
"""
import time

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from legacy_features import Features


def fit_predict(train_features, y_log, valid_features, test_features, foldseed):
    started = time.time()
    if any("SalePrice" in frame.columns for frame in
           (train_features, valid_features, test_features)):
        raise ValueError("Candidate feature frames must not contain labels")
    target = np.asarray(y_log, dtype=float)
    if target.shape != (len(train_features),) or not np.isfinite(target).all():
        raise ValueError("Training target must be a finite vector")

    prep = Features(level="engineered", transform="boxcox", rare=1)
    prep.fit(train_features)
    train_x = prep.matrix(train_features)
    valid_x = prep.matrix(valid_features)
    test_x = prep.matrix(test_features)

    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train_x)
    linear = Lasso(alpha=0.0006, max_iter=15000, random_state=foldseed)
    linear.fit(scaled_train, target)
    residual = target - linear.predict(scaled_train)
    valid_base = linear.predict(scaler.transform(valid_x))
    test_base = linear.predict(scaler.transform(test_x))

    booster = XGBRegressor(
        objective="reg:pseudohubererror", huber_slope=0.1,
        base_score=0.0, n_estimators=1000, learning_rate=0.025,
        max_depth=3, min_child_weight=2,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=2,
        tree_method="hist", device="cpu", n_jobs=8,
        random_state=foldseed, verbosity=0,
    )
    booster.fit(train_x, residual)
    valid_log = valid_base + booster.predict(valid_x)
    test_log = test_base + booster.predict(test_x)
    if not np.isfinite(valid_log).all() or not np.isfinite(test_log).all():
        raise ValueError("Candidate produced nonfinite predictions")
    print("Robust residual seed={} rows={} rounds=1000 seconds={:.2f}".format(
        foldseed, len(train_features), time.time() - started), flush=True)
    return valid_log, test_log
