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
# COUNTERRISK — REAL ELLIPTIC++ DEFENDER
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
print("          COUNTERRISK DEFENDER")
print("          ELLIPTIC++ REAL DATA")
print("=" * 60)


# ============================================================
# 1. LOAD FEATURES
# ============================================================

print("\nLoading transaction features...")

features_df = pd.read_csv(FEATURE_FILE)

print(f"Raw dataset size: {len(features_df):,}")
print(f"Raw columns: {len(features_df.columns)}")


# ============================================================
# 2. LOAD LABELS
# ============================================================

print("\nLoading transaction labels...")

classes_df = pd.read_csv(CLASS_FILE)

print(f"Label rows: {len(classes_df):,}")


# ============================================================
# 3. IDENTIFY COLUMNS
# ============================================================

print("\nFeature columns:")
print(features_df.columns[:10].tolist())

print("\nClass columns:")
print(classes_df.columns.tolist())


# ============================================================
# 4. NORMALIZE COLUMN NAMES
# ============================================================

# Elliptic++ uses txId as the transaction identifier.
# Some CSV versions may contain slightly different casing.

feature_id_col = None
class_id_col = None
class_label_col = None


for col in features_df.columns:
    if str(col).lower() == "txid":
        feature_id_col = col
        break


for col in classes_df.columns:

    col_lower = str(col).lower()

    if col_lower == "txid":
        class_id_col = col

    if col_lower in ["class", "label"]:
        class_label_col = col


if feature_id_col is None:
    raise ValueError(
        "Could not find txId column in txs_features.csv"
    )


if class_id_col is None:
    raise ValueError(
        "Could not find txId column in txs_classes.csv"
    )


if class_label_col is None:
    raise ValueError(
        "Could not find class/label column in txs_classes.csv"
    )


print("\nDetected:")
print(f"Transaction ID column : {feature_id_col}")
print(f"Class ID column       : {class_id_col}")
print(f"Class label column    : {class_label_col}")


# ============================================================
# 5. MERGE FEATURES + LABELS
# ============================================================

print("\nMerging transaction features with labels...")

df = features_df.merge(
    classes_df[
        [
            class_id_col,
            class_label_col
        ]
    ],
    left_on=feature_id_col,
    right_on=class_id_col,
    how="inner"
)


print(f"Merged rows: {len(df):,}")


# ============================================================
# 6. LABEL DISTRIBUTION
# ============================================================

print("\nOriginal class distribution:")

print(
    df[class_label_col]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. REMOVE UNKNOWN TRANSACTIONS
# ============================================================

# Elliptic++:
#
# 1 = illicit
# 2 = licit
# 3 = unknown
#
# Unknown transactions cannot be used as supervised labels.

print("\nRemoving unknown transactions...")

df = df[
    df[class_label_col].isin([1, 2])
].copy()


print(
    f"Labeled transactions remaining: "
    f"{len(df):,}"
)


# ============================================================
# 8. CONVERT LABEL
# ============================================================

# CounterRisk target:
#
# 0 = legitimate
# 1 = illicit

df["is_fraud"] = (
    df[class_label_col] == 1
).astype(int)


print("\nCounterRisk target distribution:")

print(
    df["is_fraud"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. FIND TIME COLUMN
# ============================================================

time_column = None

for col in df.columns:

    col_lower = str(col).lower()

    if col_lower in [
        "time step",
        "time_step",
        "timestep"
    ]:

        time_column = col
        break


if time_column is None:

    raise ValueError(
        "Could not find Time step column."
    )


print(
    f"\nTime column detected: {time_column}"
)


print("\nTime-step distribution:")

print(
    df[time_column]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. TEMPORAL SPLIT
# ============================================================

# IMPORTANT:
#
# We do NOT randomly split transactions.
#
# CounterRisk should simulate:
#
# Past transactions → train
# Future transactions → test
#
# Elliptic++ contains 49 time steps.
#
# We use:
#
# 1–34 → training
# 35–49 → testing
#
# This prevents future transactions from leaking
# into the training data.

TRAIN_MAX_TIME = 34

TEST_MIN_TIME = 35


train_df = df[
    df[time_column] <= TRAIN_MAX_TIME
].copy()


test_df = df[
    df[time_column] >= TEST_MIN_TIME
].copy()


print("\n" + "=" * 60)
print("TEMPORAL SPLIT")
print("=" * 60)

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
# 11. CHECK THAT BOTH CLASSES EXIST
# ============================================================

if train_df["is_fraud"].nunique() < 2:

    raise ValueError(
        "Training data contains only one class."
    )


if test_df["is_fraud"].nunique() < 2:

    raise ValueError(
        "Testing data contains only one class."
    )


# ============================================================
# 12. SELECT FEATURES
# ============================================================

# We use the 183 transaction features.
#
# We explicitly remove:
#
# txId       → identifier, not predictive information
# class      → label
# is_fraud   → our target
#
# Time step is kept because temporal behavior can
# legitimately matter for the model.

excluded_columns = {
    feature_id_col,
    class_id_col,
    class_label_col,
    "is_fraud"
}


feature_columns = []

for col in train_df.columns:

    if col not in excluded_columns:

        # Only use numeric columns.
        if pd.api.types.is_numeric_dtype(
            train_df[col]
        ):

            feature_columns.append(col)


print("\n" + "=" * 60)
print("MODEL FEATURES")
print("=" * 60)

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

print(
    "\nFirst 15 features:"
)

print(
    feature_columns[:15]
)


# ============================================================
# 13. BUILD X / Y
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
# 14. HANDLE MISSING / INFINITE VALUES
# ============================================================

print("\nCleaning numerical features...")

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# Use training medians only.
# This prevents test information leaking into training.

train_medians = X_train.median()


X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


# ============================================================
# 15. TRAIN DEFENDER
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COUNTERRISK DEFENDER")
print("=" * 60)

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


print("Training complete!")


# ============================================================
# 16. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)


probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 17. METRICS
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
# 18. CONFUSION MATRIX
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
# 19. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("       COUNTERRISK DEFENDER RESULTS")
print("=" * 60)

print(
    f"\nAccuracy            : {accuracy:.4f}"
)

print(
    f"Precision           : {precision:.4f}"
)

print(
    f"Recall              : {recall:.4f}"
)

print(
    f"F1 Score            : {f1:.4f}"
)

print(
    f"ROC-AUC             : {roc_auc:.4f}"
)

print(
    f"PR-AUC              : {pr_auc:.4f}"
)

print(
    f"False Positive Rate : {false_positive_rate:.4f}"
)

print(
    f"False Negative Rate : {false_negative_rate:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


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
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP DEFENDER FEATURES")
print("=" * 60)


importance = pd.DataFrame({

    "feature": feature_columns,

    "importance": model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 21. SAVE MODEL BUNDLE
# ============================================================

model_bundle = {

    "model": model,

    "features": feature_columns,

    "train_medians": train_medians,

    "time_column": time_column,

    "label_mapping": {

        0: "legitimate",

        1: "illicit"

    },

    "dataset": "Elliptic++",

    "train_time_steps": (
        1,
        TRAIN_MAX_TIME
    ),

    "test_time_steps": (
        TEST_MIN_TIME,
        49
    )

}


joblib.dump(
    model_bundle,
    MODEL_FILE
)


print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"\nSaved to:"
)

print(
    os.path.abspath(MODEL_FILE)
)

print("\n" + "=" * 60)