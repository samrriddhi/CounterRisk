import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# COUNTERRISK v3
#
# REAL TRANSACTION FEATURES
# +
# LEAKAGE-SAFE HISTORICAL FRAUD NETWORK
# ============================================================

DATA_DIR = "../data/ellipticpp"

TX_FEATURE_FILE = os.path.join(
    DATA_DIR,
    "txs_features.csv"
)

CLASS_FILE = os.path.join(
    DATA_DIR,
    "txs_classes.csv"
)

NETWORK_FILE = (
    "../data/"
    "counterrisk_historical_fraud_network.csv"
)

MODEL_FILE = "counterrisk_v3.pkl"


print("=" * 70)
print("                 COUNTERRISK v3")
print("      TRANSACTION + HISTORICAL FRAUD NETWORK")
print("=" * 70)


# ============================================================
# 1. LOAD TRANSACTION FEATURES
# ============================================================

print("\nLoading transaction features...")

tx_df = pd.read_csv(
    TX_FEATURE_FILE
)

tx_df["txId"] = (
    tx_df["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


# ============================================================
# 2. LOAD LABELS
# ============================================================

print("Loading transaction labels...")

classes = pd.read_csv(
    CLASS_FILE
)

classes["txId"] = (
    classes["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)

classes["class"] = pd.to_numeric(
    classes["class"],
    errors="coerce"
)


# Only known labels.

classes = classes[
    classes["class"].isin([1, 2])
].copy()


classes["is_fraud"] = (
    classes["class"] == 1
).astype(int)


# ============================================================
# 3. LOAD HISTORICAL NETWORK FEATURES
# ============================================================

print(
    "\nLoading historical network features..."
)

network = pd.read_csv(
    NETWORK_FILE
)

network["txId"] = (
    network["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


network_features = [
    "prior_connected_transactions",
    "prior_illicit_connections",
    "connected_wallets",
    "prior_illicit_ratio"
]


# Verify columns.

for feature in network_features:

    if feature not in network.columns:

        raise ValueError(
            f"Missing network feature: {feature}"
        )


network = network[
    [
        "txId",
        *network_features
    ]
]


# ============================================================
# 4. MERGE
# ============================================================

print("\nMerging datasets...")

df = (
    tx_df
    .merge(
        classes[
            [
                "txId",
                "is_fraud"
            ]
        ],
        on="txId",
        how="inner"
    )
    .merge(
        network,
        on="txId",
        how="left"
    )
)


print(
    f"Labeled transactions: {len(df):,}"
)


# ============================================================
# 5. CLEAN NETWORK FEATURES
# ============================================================

for feature in network_features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).fillna(0)


# ============================================================
# 6. TIME
# ============================================================

TIME_COLUMN = "Time step"

df[TIME_COLUMN] = pd.to_numeric(
    df[TIME_COLUMN],
    errors="coerce"
)

df = df.dropna(
    subset=[TIME_COLUMN]
)


# ============================================================
# 7. TEMPORAL SPLIT
# ============================================================

TRAIN_MAX_TIME = 34
TEST_MIN_TIME = 35


train_df = df[
    df[TIME_COLUMN] <= TRAIN_MAX_TIME
].copy()


test_df = df[
    df[TIME_COLUMN] >= TEST_MIN_TIME
].copy()


print("\n" + "=" * 70)
print("TEMPORAL SPLIT")
print("=" * 70)

print(
    f"Training time steps : 1-{TRAIN_MAX_TIME}"
)

print(
    f"Testing time steps  : {TEST_MIN_TIME}-49"
)

print(
    f"Training rows       : {len(train_df):,}"
)

print(
    f"Testing rows        : {len(test_df):,}"
)


# ============================================================
# 8. SELECT MODEL FEATURES
# ============================================================

excluded = {
    "txId",
    "class",
    "is_fraud"
}


feature_columns = []

for column in df.columns:

    if column in excluded:

        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        feature_columns.append(
            column
        )


print("\n" + "=" * 70)
print("MODEL FEATURES")
print("=" * 70)

print(
    f"Total features: "
    f"{len(feature_columns)}"
)

print(
    "\nHistorical network features:"
)

for feature in network_features:

    print(
        "  →",
        feature
    )


# ============================================================
# 9. BUILD MATRICES
# ============================================================

X_train = train_df[
    feature_columns
].copy()

y_train = train_df[
    "is_fraud"
].copy()


X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    "is_fraud"
].copy()


# ============================================================
# 10. CLEAN NUMERICAL VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


train_medians = X_train.median(
    numeric_only=True
)


X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COUNTERRISK v3")
print("=" * 70)


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=18,

    min_samples_leaf=3,

    class_weight="balanced_subsample",

    random_state=42,

    n_jobs=-1
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# 12. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 13. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


fpr = (
    fp / (fp + tn)
    if fp + tn > 0
    else 0
)

fnr = (
    fn / (fn + tp)
    if fn + tp > 0
    else 0
)


# ============================================================
# 15. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("             COUNTERRISK v3 RESULTS")
print("=" * 70)

print(
    f"\nAccuracy            : "
    f"{accuracy:.4f}"
)

print(
    f"Precision           : "
    f"{precision:.4f}"
)

print(
    f"Recall              : "
    f"{recall:.4f}"
)

print(
    f"F1 Score            : "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC             : "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC              : "
    f"{pr_auc:.4f}"
)

print(
    f"False Positive Rate : "
    f"{fpr:.4f}"
)

print(
    f"False Negative Rate : "
    f"{fnr:.4f}"
)


print("\nConfusion Matrix:")

print(
    cm
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Illicit"
        ],
        zero_division=0
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        feature_columns,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP 25 FEATURES")
print("=" * 70)

print(
    importance
    .head(25)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. HISTORICAL NETWORK IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("HISTORICAL NETWORK FEATURE IMPORTANCE")
print("=" * 70)


network_importance = importance[
    importance["feature"].isin(
        network_features
    )
]


print(
    network_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

bundle = {

    "model":
        model,

    "features":
        feature_columns,

    "train_medians":
        train_medians,

    "network_features":
        network_features,

    "time_column":
        TIME_COLUMN,

    "train_time_steps":
        (
            1,
            TRAIN_MAX_TIME
        ),

    "test_time_steps":
        (
            TEST_MIN_TIME,
            49
        ),

    "dataset":
        "Elliptic++",

    "version":
        "CounterRisk-v3"
}


joblib.dump(
    bundle,
    MODEL_FILE
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    os.path.abspath(
        MODEL_FILE
    )
)

print("=" * 70)