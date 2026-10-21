"""Fixed fold-fitted spline Ridge candidate; no cached predictions or file reads."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "8"
import numpy as np
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge
from legacy_features import Features


def _smooth_input(features, frame):
    z = features.base(frame)
    columns = ["OverallQual", "OverallCond", "GrLivArea", "TotalSF", "FinishedSF",
               "EffectiveSF", "LotArea", "GarageArea", "TotalBath", "HouseAge",
               "RemodAge", "TotalPorchSF", "TotalQuality", "BsmtQualitySF"]
    smooth = z[columns].astype(float).copy()
    for c in ["GrLivArea", "TotalSF", "FinishedSF", "EffectiveSF", "LotArea",
              "GarageArea", "HouseAge", "RemodAge", "TotalPorchSF", "BsmtQualitySF"]:
        smooth[c] = np.log1p(smooth[c].clip(lower=0))
    return smooth


def fit_predict(train_features, y_log, valid_features, test_features, foldseed):
    features = Features(level="extended", transform="log", rare=1).fit(train_features)
    matrices = [features.matrix(frame) for frame in (train_features, valid_features, test_features)]
    smooth = [_smooth_input(features, frame) for frame in (train_features, valid_features, test_features)]
    median = smooth[0].median().fillna(0)
    smooth = [frame.fillna(median).to_numpy(dtype=float) for frame in smooth]
    spline = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear",
                               include_bias=False)
    basis = [spline.fit_transform(smooth[0]), spline.transform(smooth[1]), spline.transform(smooth[2])]
    design = [np.column_stack([matrix, nonlinear]) for matrix, nonlinear in zip(matrices, basis)]
    scaler = StandardScaler().fit(design[0])
    design = [scaler.transform(matrix) for matrix in design]
    model = Ridge(alpha=20.0, solver="cholesky").fit(design[0], np.asarray(y_log))
    return model.predict(design[1]), model.predict(design[2])
