"""Train a House Prices model and write a Kaggle submission file.

Local score is RMSE on log(SalePrice), the same metric as the competition.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 42
N_SPLITS = 5

NONE_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ORDINAL_MAPS = {
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {
        "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6,
    },
    "BsmtFinType2": {
        "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6,
    },
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {
        "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8,
    },
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}

ZERO_WHEN_MISSING = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]


def rmsle_log(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def engineer(df):
    out = df.copy()
    for col in NONE_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("None")

    for col in ZERO_WHEN_MISSING:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    if "GarageYrBlt" in out.columns:
        out["GarageYrBlt"] = out["GarageYrBlt"].fillna(out["YearBuilt"])

    for col, mapping in ORDINAL_MAPS.items():
        if col in out.columns:
            out[col] = out[col].map(mapping)

    out["MSSubClass"] = out["MSSubClass"].astype(str)
    out["MoSold"] = out["MoSold"].astype(str)
    out["YrSold"] = out["YrSold"].astype(str)

    out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    out["TotalBath"] = (
        out["FullBath"]
        + 0.5 * out["HalfBath"]
        + out["BsmtFullBath"]
        + 0.5 * out["BsmtHalfBath"]
    )
    out["TotalPorchSF"] = (
        out["OpenPorchSF"]
        + out["EnclosedPorch"]
        + out["3SsnPorch"]
        + out["ScreenPorch"]
        + out["WoodDeckSF"]
    )
    out["HouseAge"] = out["YrSold"].astype(int) - out["YearBuilt"]
    out["RemodAge"] = out["YrSold"].astype(int) - out["YearRemodAdd"]
    out["IsNew"] = (out["YearBuilt"] == out["YrSold"].astype(int)).astype(int)
    out["QualArea"] = out["OverallQual"] * out["GrLivArea"]
    out["HasGarage"] = (out["GarageArea"] > 0).astype(int)
    out["HasBsmt"] = (out["TotalBsmtSF"] > 0).astype(int)
    out["HasPool"] = (out["PoolArea"] > 0).astype(int)
    out["HasFireplace"] = (out["Fireplaces"] > 0).astype(int)
    return out


def fit_lot_frontage(train_df):
    med = train_df.groupby("Neighborhood")["LotFrontage"].median()
    global_med = train_df["LotFrontage"].median()
    return med, global_med


def apply_lot_frontage(df, med, global_med):
    out = df.copy()
    filled = out["Neighborhood"].map(med)
    out["LotFrontage"] = out["LotFrontage"].fillna(filled).fillna(global_med)
    return out


def prepare_matrices(train_fe, test_fe):
    y = np.log1p(train_fe["SalePrice"].to_numpy())
    train_x = train_fe.drop(columns=["SalePrice", "Id"])
    test_ids = test_fe["Id"].copy()
    test_x = test_fe.drop(columns=["Id"])

    numeric_cols = train_x.select_dtypes(include=[np.number]).columns.tolist()
    skew = train_x[numeric_cols].skew().abs()
    skewed = skew[skew > 0.75].index.tolist()
    for col in skewed:
        train_x[col] = np.log1p(train_x[col].clip(lower=0))
        test_x[col] = np.log1p(test_x[col].clip(lower=0))

    medians = train_x[numeric_cols].median()
    train_x[numeric_cols] = train_x[numeric_cols].fillna(medians)
    test_x[numeric_cols] = test_x[numeric_cols].fillna(medians)

    cat_cols = train_x.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        mode = train_x[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "None"
        train_x[col] = train_x[col].fillna(fill)
        test_x[col] = test_x[col].fillna(fill)

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    combined = combined.astype(np.float64)
    nunique = combined.nunique()
    combined = combined.loc[:, nunique > 1]
    x_train = combined.iloc[: len(train_x)].reset_index(drop=True)
    x_test = combined.iloc[len(train_x) :].reset_index(drop=True)
    return x_train, y, x_test, test_ids


def make_models():
    return {
        "ridge": lambda: Ridge(alpha=10.0, random_state=SEED),
        "lasso": lambda: Lasso(alpha=0.0005, max_iter=8000, random_state=SEED),
        "elasticnet": lambda: ElasticNet(
            alpha=0.0005, l1_ratio=0.7, max_iter=8000, random_state=SEED
        ),
        "lgb": lambda: lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=12,
            min_child_samples=8,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            n_jobs=4,
            verbosity=-1,
        ),
        "xgb": lambda: xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=4,
            early_stopping_rounds=150,
        ),
    }


def fit_predict_fold(name, model, x_tr, y_tr, x_va, y_va, x_te):
    if name in {"ridge", "lasso", "elasticnet"}:
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_va_s = scaler.transform(x_va)
        x_te_s = scaler.transform(x_te)
        model.fit(x_tr_s, y_tr)
        return model.predict(x_va_s), model.predict(x_te_s)

    if name == "lgb":
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        return model.predict(x_va), model.predict(x_te)

    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    return model.predict(x_va), model.predict(x_te)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    # Partial sales of very large homes distort the fit.
    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    train = train.loc[~outlier_mask].reset_index(drop=True)

    frontage_med, frontage_global = fit_lot_frontage(train)
    train = apply_lot_frontage(train, frontage_med, frontage_global)
    test = apply_lot_frontage(test, frontage_med, frontage_global)

    train_fe = engineer(train)
    test_fe = engineer(test)
    x_train, y, x_test, test_ids = prepare_matrices(train_fe, test_fe)
    print(f"train={x_train.shape} test={x_test.shape}", flush=True)

    models = make_models()
    oof = {name: np.zeros(len(x_train)) for name in models}
    test_pred = {name: np.zeros(len(x_test)) for name in models}
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(folds.split(x_train), start=1):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        for name, factory in models.items():
            pred_va, pred_te = fit_predict_fold(
                name, factory(), x_tr, y_tr, x_va, y_va, x_test
            )
            oof[name][va_idx] = pred_va
            test_pred[name] += pred_te / N_SPLITS
        print(f"fold {fold} done", flush=True)

    scores = {name: rmsle_log(y, pred) for name, pred in oof.items()}
    candidates = {
        "xgb": ["xgb"],
        "trees": ["lgb", "xgb"],
        "xgb_lasso": ["xgb", "lasso"],
        "xgb_lasso_lgb": ["xgb", "lasso", "lgb"],
        "all": list(models),
    }
    blend_name, keep, blend_test = None, None, None
    best_score = np.inf
    for name, members in candidates.items():
        oof_pred = np.mean([oof[m] for m in members], axis=0)
        score = rmsle_log(y, oof_pred)
        scores[name] = score
        if score < best_score:
            best_score = score
            blend_name = name
            keep = members
            blend_test = np.mean([test_pred[m] for m in members], axis=0)

    print("5-fold RMSE on log price:")
    for name, score in scores.items():
        mark = " *" if name == blend_name else ""
        print(f"  {name:14s} {score:.5f}{mark}")
    print("selected:", blend_name, "=", ", ".join(keep))

    sale_price = np.expm1(blend_test)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": sale_price})
    submission = sample[["Id"]].merge(submission, on="Id", how="left")
    if submission["SalePrice"].isna().any():
        raise RuntimeError("submission is missing predictions for some Ids")

    out_path = ROOT / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(submission)}")
    print(submission["SalePrice"].describe().to_string())


if __name__ == "__main__":
    main()
