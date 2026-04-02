import numpy as np
import pandas as pd


def prepare_features(train_df, test_df, feature_cols):
    """
    Prepare features for Part A.

    Rule:
    - Categorical column (Neighborhood) is one-hot encoded using TRAIN set categories.
    - Test set is aligned to the TRAIN set encoded columns.
    - Numeric columns are kept numeric.
    """
    x_train_df = train_df.loc[:, feature_cols].copy()
    x_test_df = test_df.loc[:, feature_cols].copy()

    categorical_col = feature_cols[0]

    # One-hot encode based only on train data
    x_train_encoded = pd.get_dummies(
        x_train_df,
        columns=[categorical_col],
        drop_first=False
    )

    x_test_encoded = pd.get_dummies(
        x_test_df,
        columns=[categorical_col],
        drop_first=False
    )

    # Align test columns to train columns
    x_test_encoded = x_test_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)

    # Convert all columns to numeric
    for col in x_train_encoded.columns:
        x_train_encoded[col] = pd.to_numeric(x_train_encoded[col], errors="coerce")
        x_test_encoded[col] = pd.to_numeric(x_test_encoded[col], errors="coerce")

    return (
        x_train_encoded.values.astype(float),
        x_test_encoded.values.astype(float),
        list(x_train_encoded.columns),
    )


def minmax_fit(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return data_min, data_max


def minmax_transform(data, data_min, data_max):
    denominator = data_max - data_min
    denominator = np.where(denominator == 0, 1, denominator)
    return (data - data_min) / denominator


def minmax_inverse_transform(data_norm, data_min, data_max):
    return data_norm * (data_max - data_min) + data_min


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)