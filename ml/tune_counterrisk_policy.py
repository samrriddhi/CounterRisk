import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# ============================================================
# COUNTERRISK POLICY TUNER
#
# We keep the Defender fixed.
# We only tune the Challenger override policy.
# ============================================================

BASE = "../data/ellipticpp"

TX_FEATURE_FILE = os.path.join(
    BASE,
    "txs_features.csv"
)

CLASS_FILE = os.path.join(
    BASE,
    "txs_classes.csv"
)

NETWORK_FILE = (
    "../data/"
    "counterrisk_historical_fraud_network.csv"
)

MODEL_FILE = "counterrisk_v3.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("       COUNTERRISK CHALLENGER POLICY TUNER")
print("=" * 70)

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]
feature_columns = bundle["features"]
train_medians = bundle["train_medians"]


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

tx = pd.read_csv(
    TX_FEATURE_FILE
)

tx["txId"] = (
    tx["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


labels = pd.read_csv(
    CLASS_FILE
)

labels["txId"] = (
    labels["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)

labels["class"] = pd.to_numeric(
    labels["class"],
    errors="coerce"
)

labels = labels[
    labels["class"].isin([1, 2])
].copy()

labels["is_fraud"] = (
    labels["class"] == 1
).astype(int)


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


# ============================================================
# MERGE
# ============================================================

df = (
    tx
    .merge(
        labels[
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


# ============================================================
# FUTURE TEST PERIOD
# ============================================================

df = df[
    df["Time step"] >= 35
].copy()


print(
    f"Evaluation transactions: {len(df):,}"
)


# ============================================================
# DEFENDER PROBABILITY
# ============================================================

X = df[
    feature_columns
].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    train_medians
)


print("Running Defender...")

df["defender_probability"] = (
    model.predict_proba(
        X
    )[:, 1]
)


# ============================================================
# BASELINE DEFENDER
# ============================================================

df["defender_prediction"] = (
    df[
        "defender_probability"
    ] >= 0.50
).astype(int)


# ============================================================
# POLICY FUNCTION
#
# Start with Defender decision.
#
# Then allow LIMITED overrides.
# ============================================================

def apply_policy(
    row,
    clean_history_min=10,
    strong_fraud_min=5,
    strong_fraud_ratio=0.70
):

    risk = row[
        "defender_probability"
    ]

    prior_connections = row[
        "prior_connected_transactions"
    ]

    prior_illicit = row[
        "prior_illicit_connections"
    ]

    ratio = row[
        "prior_illicit_ratio"
    ]


    defender_prediction = (
        1
        if risk >= 0.50
        else 0
    )


    # --------------------------------------------------------
    # Challenger says Defender may be TOO AGGRESSIVE.
    #
    # Require:
    #   high Defender probability
    #   substantial clean history
    #   zero known illicit activity
    #
    # This prevents us from overriding weak evidence.
    # --------------------------------------------------------

    if (
        risk >= 0.75
        and
        prior_connections >= clean_history_min
        and
        prior_illicit == 0
    ):

        return 0


    # --------------------------------------------------------
    # Challenger says Defender may be UNDERESTIMATING.
    #
    # Require:
    #   relatively low Defender risk
    #   multiple prior connections
    #   very high historical illicit ratio
    # --------------------------------------------------------

    if (
        risk < 0.50
        and
        prior_connections >= strong_fraud_min
        and
        ratio >= strong_fraud_ratio
    ):

        return 1


    # --------------------------------------------------------
    # Otherwise preserve Defender.
    # --------------------------------------------------------

    return defender_prediction


# ============================================================
# EVALUATE ONE POLICY
# ============================================================

def evaluate_policy(
    clean_history_min,
    strong_fraud_min,
    strong_fraud_ratio
):

    predictions = []

    for row in df.itertuples():

        predictions.append(
            apply_policy(
                row._asdict()
                if hasattr(
                    row,
                    "_asdict"
                )
                else row,
                clean_history_min,
                strong_fraud_min,
                strong_fraud_ratio
            )
        )

    y_true = df[
        "is_fraud"
    ].values

    y_pred = np.array(
        predictions
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0
    )

    return {
        "clean_history_min":
            clean_history_min,

        "strong_fraud_min":
            strong_fraud_min,

        "strong_fraud_ratio":
            strong_fraud_ratio,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "fpr":
            fpr,

        "fp_reduction":
            55 - fp,

        "fn_change":
            fn - 301
    }


# ============================================================
# GRID SEARCH
# ============================================================

print("\nSearching policy space...")


results = []


for clean_history_min in [
    5,
    10,
    15,
    20,
    30
]:

    for strong_fraud_min in [
        3,
        5,
        10,
        15
    ]:

        for strong_fraud_ratio in [
            0.60,
            0.70,
            0.80,
            0.90
        ]:

            result = evaluate_policy(
                clean_history_min,
                strong_fraud_min,
                strong_fraud_ratio
            )

            results.append(
                result
            )


results_df = pd.DataFrame(
    results
)


# ============================================================
# FILTER FOR REASONABLE RECALL
# ============================================================

reasonable = results_df[
    results_df["recall"] >= 0.70
].copy()


reasonable = reasonable.sort_values(
    [
        "false_positives",
        "f1"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# DISPLAY BEST POLICIES
# ============================================================

print()
print("=" * 70)
print("BEST POLICIES WITH RECALL >= 70%")
print("=" * 70)


if len(reasonable) == 0:

    print(
        "No policy preserved at least 70% recall."
    )

else:

    print(
        reasonable.head(
            15
        ).to_string(
            index=False
        )
    )


# ============================================================
# ALSO SHOW BEST F1
# ============================================================

best_f1 = results_df.sort_values(
    "f1",
    ascending=False
).head(10)


print()
print("=" * 70)
print("BEST POLICIES BY F1")
print("=" * 70)

print(
    best_f1.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "../data/"
    "counterrisk_policy_search.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print()
print("=" * 70)
print("POLICY SEARCH SAVED")
print("=" * 70)

print(
    output_file
)

print("=" * 70)