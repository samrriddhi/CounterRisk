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
# COUNTERRISK DEFENDER v2
# REAL ELLIPTIC++ + TEMPORAL NETWORK FEATURES
# ============================================================

DATA_DIR = "../data/ellipticpp"

TRANSACTION_FEATURE_FILE = os.path.join(
    DATA_DIR,
    "txs_features.csv"
)

CLASS_FILE = os.path.join(
    DATA_DIR,
    "txs_classes.csv"
)

NETWORK_FEATURE_FILE = (
    "../data/"
    "counterrisk_temporal_network_features.csv"
)

MODEL_FILE = "counterrisk_v2.pkl"


print("=" * 70)
print("             COUNTERRISK DEFENDER v2")
print("        TRANSACTION + NETWORK INTELLIGENCE")
print("=" * 70)


# ============================================================
# 1. LOAD TRANSACTION FEATURES
# ============================================================

print("\nLoading transaction features...")

tx_df = pd.read_csv(
    TRANSACTION_FEATURE_FILE
)

tx_df["txId"] = (
    tx_df["txId"]
    .apply(lambda x: str(int(float(x))))
)

print(
    f"Transactions: {len(tx_df):,}"
)

print(
    f"Transaction features: "
    f"{len(tx_df.columns):,}"
)


# ============================================================
# 2. LOAD LABELS
# ============================================================

print("\nLoading labels...")

classes_df = pd.read_csv(
    CLASS_FILE
)

classes_df["txId"] = (
    classes_df["txId"]
    .apply(lambda x: str(int(float(x))))
)

classes_df["class"] = pd.to_numeric(
    classes_df["class"],
    errors="coerce"
)

print(
    f"Labels: {len(classes_df):,}"
)


# ============================================================
# 3. KEEP ONLY LABELED TRANSACTIONS
# ============================================================

print("\nFiltering labeled transactions...")

classes_df = classes_df[
    classes_df["class"].isin([1, 2])
].copy()

classes_df["is_fraud"] = (
    classes_df["class"] == 1
).astype(int)


# ============================================================
# 4. LOAD NETWORK FEATURES
# ============================================================

print("\nLoading temporal network features...")

network_df = pd.read_csv(
    NETWORK_FEATURE_FILE
)

network_df["txId"] = (
    network_df["txId"]
    .apply(lambda x: str(int(float(x))))
)

print(
    f"Network feature rows: "
    f"{len(network_df):,}"
)


# ============================================================
# 5. REMOVE DUPLICATE TIME COLUMN FROM NETWORK DATA
# ============================================================

# The transaction dataset already contains the authoritative
# Time step. We only need the network-derived columns here.

network_features = [
    "prior_wallet_connections",
    "active_wallets",
    "total_wallet_count",
    "network_first_seen",
    "network_age",
    "historical_network_activity"
]

missing_network_features = [
    feature
    for feature in network_features
    if feature not in network_df.columns
]

if missing_network_features:

    raise ValueError(
        "Missing network features: "
        + str(missing_network_features)
    )


network_df = network_df[
    ["txId"] + network_features
].copy()


# ============================================================
# 6. MERGE TRANSACTIONS + LABELS + NETWORK FEATURES
# ============================================================

print("\nMerging datasets...")

df = (
    tx_df
    .merge(
        classes_df[
            [
                "txId",
                "is_fraud"
            ]
        ],
        on="txId",
        how="inner"
    )
    .merge(
        network_df,
        on="txId",
        how="left"
    )
)


print(
    f"Merged labeled transactions: "
    f"{len(df):,}"
)


# ============================================================
# 7. CHECK TIME COLUMN
# ============================================================

time_column = "Time step"

if time_column not in df.columns:

    raise ValueError(
        "Time step column was not found after merge."
    )


df[time_column] = pd.to_numeric(
    df[time_column],
    errors="coerce"
)

df = df.dropna(
    subset=[time_column]
)


# ============================================================
# 8. CLEAN NETWORK FEATURES
# ============================================================

for feature in network_features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    df[feature] = (
        df[feature]
        .fillna(0)
    )


# ============================================================
# 9. TEMPORAL SPLIT
# ============================================================

TRAIN_MAX_TIME = 34
TEST_MIN_TIME = 35


train_df = df[
    df[time_column] <= TRAIN_MAX_TIME
].copy()

test_df = df[
    df[time_column] >= TEST_MIN_TIME
].copy()


print("\n" + "=" * 70)
print("TEMPORAL SPLIT")
print("=" * 70)

print(
    f"Training: time steps 1-{TRAIN_MAX_TIME}"
)

print(
    f"Testing : time steps {TEST_MIN_TIME}-49"
)

print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Testing rows : {len(test_df):,}"
)


# ============================================================
# 10. SELECT MODEL FEATURES
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
    f"Transaction + network features: "
    f"{len(feature_columns)}"
)

print(
    "\nNetwork features:"
)

for feature in network_features:

    print(
        "  →",
        feature
    )


# ============================================================
# 11. TRAIN / TEST MATRICES
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
# 12. CLEAN NUMERICAL FEATURES
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
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COUNTERRISK v2")
print("=" * 70)

print(
    "\nTraining Random Forest..."
)


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=18,

    min_samples_leaf=3,

    class_weight="balanced_subsample",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)

print(
    "Training complete!"
)


# ============================================================
# 14. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. METRICS
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
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("          COUNTERRISK v2 RESULTS")
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
    f"{false_positive_rate:.4f}"
)

print(
    f"False Negative Rate : "
    f"{false_negative_rate:.4f}"
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
# 18. FEATURE IMPORTANCE
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
# 19. NETWORK FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("NETWORK FEATURE IMPORTANCE")
print("=" * 70)


network_importance = (
    importance[
        importance["feature"]
        .isin(network_features)
    ]
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    network_importance.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

model_bundle = {

    "model":
        model,

    "features":
        feature_columns,

    "train_medians":
        train_medians,

    "network_features":
        network_features,

    "time_column":
        time_column,

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
        "CounterRisk-v2"
}


joblib.dump(
    model_bundle,
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