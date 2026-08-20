import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from challenger import CounterRiskChallenger


# ============================================================
# COUNTERRISK — FULL TEMPORAL EVALUATION
#
# Defender vs Defender + Challenger
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
print("        COUNTERRISK FULL EVALUATION")
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

print("\nLoading transaction features...")

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


print("Loading labels...")

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


print("Loading historical network evidence...")

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
# FUTURE TEST SET
# ============================================================

df = df[
    df["Time step"] >= 35
].copy()


print(
    f"\nEvaluation transactions: "
    f"{len(df):,}"
)


# ============================================================
# MODEL INPUT
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


# ============================================================
# DEFENDER
# ============================================================

print("\nRunning Defender...")

df["defender_probability"] = (
    model.predict_proba(
        X
    )[:, 1]
)

df["defender_prediction"] = (
    df[
        "defender_probability"
    ] >= 0.50
).astype(int)


# ============================================================
# CHALLENGER
# ============================================================

print(
    "Running Challenger..."
)

challenger = (
    CounterRiskChallenger()
)


challenger_outcomes = []

challenger_directions = []

challenger_disputes = []


for row in df.itertuples():

    result = challenger.investigate(

        defender_probability=
            row.defender_probability,

        prior_connected_transactions=
            getattr(
                row,
                "prior_connected_transactions",
                0
            ),

        prior_illicit_connections=
            getattr(
                row,
                "prior_illicit_connections",
                0
            ),

        prior_illicit_ratio=
            getattr(
                row,
                "prior_illicit_ratio",
                0
            ),

        connected_wallets=
            getattr(
                row,
                "connected_wallets",
                0
            )
    )


    challenger_outcomes.append(
        result["outcome"]
    )

    challenger_directions.append(
        result["direction"]
    )

    challenger_disputes.append(
        result["disputes_defender"]
    )


df["challenger_outcome"] = (
    challenger_outcomes
)

df["challenger_direction"] = (
    challenger_directions
)

df["challenger_disputes"] = (
    challenger_disputes
)


# ============================================================
# FINAL DECISION
# ============================================================

def final_decision(row):

    risk = row[
        "defender_probability"
    ]

    outcome = row[
        "challenger_outcome"
    ]

    direction = row[
        "challenger_direction"
    ]

    ratio = row[
        "prior_illicit_ratio"
    ]


    # --------------------------------------------------------
    # DEFENDER HIGH
    # --------------------------------------------------------

    if risk >= 0.75:

        if (
            outcome == "DISPUTE"
            and
            direction ==
            "DEFENDER_TOO_AGGRESSIVE"
        ):

            return 0

        return 1


    # --------------------------------------------------------
    # DEFENDER LOW
    # --------------------------------------------------------

    if risk < 0.45:

        if (
            outcome == "DISPUTE"
            and
            direction ==
            "DEFENDER_UNDERESTIMATES_RISK"
        ):

            if ratio >= 0.50:

                return 1

            return 0

        return 0


    # --------------------------------------------------------
    # BORDERLINE
    # --------------------------------------------------------

    if outcome == "DISPUTE":

        if direction == (
            "DEFENDER_UNDERESTIMATES_RISK"
        ):

            if ratio >= 0.50:

                return 1

        return 0


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return 0


df["counterrisk_prediction"] = (
    df.apply(
        final_decision,
        axis=1
    )
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "fpr":
            fp / (fp + tn)
            if fp + tn > 0
            else 0
    }


# ============================================================
# RESULTS
# ============================================================

defender_metrics = calculate_metrics(

    df["is_fraud"],

    df[
        "defender_prediction"
    ]
)


counterrisk_metrics = calculate_metrics(

    df["is_fraud"],

    df[
        "counterrisk_prediction"
    ]
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("DEFENDER VS COUNTERRISK")
print("=" * 70)


print(
    "\nDEFENDER"
)

for key, value in defender_metrics.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key:20}: {value:.4f}"
        )

    else:

        print(
            f"{key:20}: {value}"
        )


print(
    "\nCOUNTERRISK"
)

for key, value in counterrisk_metrics.items():

    if isinstance(
        value,
        float
    ):

        print(
            f"{key:20}: {value:.4f}"
        )

    else:

        print(
            f"{key:20}: {value}"
        )


# ============================================================
# CHALLENGER STATISTICS
# ============================================================

print()
print("=" * 70)
print("CHALLENGER ACTIVITY")
print("=" * 70)

print(
    "Transactions challenged :",
    df[
        "challenger_disputes"
    ].sum()
)

print(
    "Defender too aggressive  :",
    (
        df[
            "challenger_direction"
        ]
        ==
        "DEFENDER_TOO_AGGRESSIVE"
    ).sum()
)

print(
    "Defender underestimated  :",
    (
        df[
            "challenger_direction"
        ]
        ==
        "DEFENDER_UNDERESTIMATES_RISK"
    ).sum()
)


# ============================================================
# IMPROVEMENT
# ============================================================

fp_improvement = (

    defender_metrics[
        "false_positives"
    ]
    -
    counterrisk_metrics[
        "false_positives"
    ]
)

fn_change = (

    counterrisk_metrics[
        "false_negatives"
    ]
    -
    defender_metrics[
        "false_negatives"
    ]
)


print()
print("=" * 70)
print("COUNTERRISK IMPACT")
print("=" * 70)

print(
    "False-positive change :",
    fp_improvement
)

print(
    "False-negative change :",
    fn_change
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "../data/"
    "counterrisk_evaluation.csv"
)


df.to_csv(
    output_file,
    index=False
)


print()
print(
    "Saved:"
)

print(
    output_file
)

print("=" * 70)