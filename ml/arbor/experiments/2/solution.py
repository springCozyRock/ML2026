"""Editable candidate. Fit preprocessing only on train_features; Id is identification.
Return (validation_log_predictions, test_log_predictions), or None for baseline.
Do not read protected files, evaluator labels, network, or cached predictions.
"""
import numpy as np
from sklearn.preprocessing import TargetEncoder
from lightgbm import LGBMRegressor
from legacy_features import Features


def fit_predict(train_features, y_log, valid_features, test_features, foldseed):
    """Cross-fit training target encodings; infer using outer-training maps only."""
    if any("SalePrice" in frame.columns for frame in
           (train_features, valid_features, test_features)):
        raise ValueError("Candidate features must not contain target labels")
    y_log = np.asarray(y_log, dtype=float)
    prep = Features(level="extended", transform="none").fit(train_features)
    frames = [prep.base(frame) for frame in
              (train_features, valid_features, test_features)]
    cats = [prep.clean_cat(frame) for frame in frames]
    encoder = TargetEncoder(target_type="continuous", smooth=20.0,
                            cv=5, shuffle=True, random_state=foldseed)
    # fit_transform is deliberately cross-fitted; fit().transform() would leak.
    categorical = [encoder.fit_transform(cats[0], y_log),
                   encoder.transform(cats[1]), encoder.transform(cats[2])]
    frequencies = {column: cats[0][column].value_counts(normalize=True)
                   for column in prep.cat}
    matrices = []
    for frame, cat, encoded in zip(frames, cats, categorical):
        frequency = np.column_stack([
            cat[column].map(frequencies[column]).fillna(0).to_numpy(dtype=float)
            for column in prep.cat])
        matrix = np.column_stack([prep.numeric(frame), encoded, frequency])
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessing produced nonfinite features")
        matrices.append(matrix)
    model = LGBMRegressor(
        objective="regression", n_estimators=1600, learning_rate=0.025,
        num_leaves=15, max_depth=-1, min_child_samples=20,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=8.0, random_state=foldseed,
        n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True)
    model.fit(matrices[0], y_log)
    return model.predict(matrices[1]), model.predict(matrices[2])
