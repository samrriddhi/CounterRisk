import os
import pandas as pd
import numpy as np
import joblib


# ============================================================
# COUNTERRISK — FALSE NEGATIVE INVESTIGATOR
# ============================================================

DATA_DIR = "../data/ellipticpp"

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "txs_features.csv"
)

CLASS_FILE = os.path.join(
    DATA_DIR,
    "txs_classes.csv"
)

MODEL_FILE = "counterrisk_defender.pkl"


print("=" * 60)
print("     COUNTERRISK FALSE NEGATIVE INVESTIGATOR")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]

feature_columns = bundle["features"]

train_medians = bundle["train_medians"]

time_column = bundle["time_column"]


# ============================================================
# LOAD DATA
# ============================================================

features_df = pd.read_csv(
    FEATURE_FILE
)

classes_df = pd.read_csv(
    CLASS_FILE
)


# ============================================================
# MERGE
# ============================================================

df = features_df.merge(
    classes_df,
    on="txId",
    how="inner"
)


# ============================================================
# ONLY LABELED TRANSACTIONS
# ============================================================

df = df[
    df["class"].isin([1, 2])
].copy()


df["is_fraud"] = (
    df["class"] == 1
).astype(int)


# ============================================================
# SAME TEMPORAL TEST SET
# ============================================================

test_df = df[
    df[time_column] >= 35
].copy()


X_test = test_df[
    feature_columns
].copy()


# ============================================================
# CLEAN FEATURES
# ============================================================

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.fillna(
    train_medians
)


# ============================================================
# PREDICT
# ============================================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1
]


test_df["prediction"] = predictions

test_df["fraud_probability"] = probabilities


# ============================================================
# FALSE NEGATIVES
# ============================================================

false_negatives = test_df[
    (test_df["is_fraud"] == 1)
    &
    (test_df["prediction"] == 0)
].copy()


print()
print(
    f"False negatives: "
    f"{len(false_negatives):,}"
)


# ============================================================
# HIGH-CONFIDENCE FALSE NEGATIVES
# ============================================================

false_negatives = false_negatives.sort_values(
    "fraud_probability",
    ascending=False
)


print()
print("=" * 60)
print("TOP MISSED ILLICIT TRANSACTIONS")
print("=" * 60)


display_columns = [
    "txId",
    "Time step",
    "fraud_probability"
]


print(
    false_negatives[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    "../data/counterrisk_false_negatives.csv"
)


false_negatives.to_csv(
    output_path,
    index=False
)


print()
print("=" * 60)

print(
    f"Saved:"
)

print(
    output_path
)

print("=" * 60)