import os
import pandas as pd
import numpy as np
import joblib


# ============================================================
# COUNTERRISK — DECISION ENGINE
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

NETWORK_FILE = (
    "../data/"
    "counterrisk_temporal_network_features.csv"
)

MODEL_FILE = "counterrisk_v2.pkl"


print("=" * 70)
print("              COUNTERRISK DECISION ENGINE")
print("=" * 70)


# ============================================================
# 1. LOAD MODEL
# ============================================================

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]

feature_columns = bundle["features"]

train_medians = bundle["train_medians"]

time_column = bundle["time_column"]

network_features = bundle[
    "network_features"
]


# ============================================================
# 2. LOAD TRANSACTION DATA
# ============================================================

tx_df = pd.read_csv(
    FEATURE_FILE
)

tx_df["txId"] = (
    tx_df["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


# ============================================================
# 3. LOAD LABELS
# ============================================================

classes_df = pd.read_csv(
    CLASS_FILE
)

classes_df["txId"] = (
    classes_df["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)

classes_df = classes_df[
    classes_df["class"].isin([1, 2])
].copy()

classes_df["is_fraud"] = (
    classes_df["class"] == 1
).astype(int)


# ============================================================
# 4. LOAD NETWORK FEATURES
# ============================================================

network_df = pd.read_csv(
    NETWORK_FILE
)

network_df["txId"] = (
    network_df["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


network_df = network_df[
    [
        "txId",
        *network_features
    ]
].copy()


# ============================================================
# 5. MERGE
# ============================================================

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


# ============================================================
# 6. TEMPORAL TEST SET
# ============================================================

df = df[
    df[time_column] >= 35
].copy()


# ============================================================
# 7. CLEAN MODEL FEATURES
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
# 8. DEFENDER PREDICTION
# ============================================================

df["defender_probability"] = (
    model.predict_proba(X)[:, 1]
)


# ============================================================
# 9. NORMALIZED NETWORK SCORE
#
# This is NOT a fraud label.
# It represents historical network activity.
# ============================================================

# Historical activity
activity = (
    df["historical_network_activity"]
    .clip(
        lower=0
    )
)


# Wallet participation
wallets = (
    df["total_wallet_count"]
    .clip(
        lower=0
    )
)


# Network age
age = (
    df["network_age"]
    .clip(
        lower=0
    )
)


# Log transforms prevent huge wallets from
# dominating the score.

activity_score = (
    np.log1p(activity)
    / np.log1p(
        max(
            activity.max(),
            1
        )
    )
)


wallet_score = (
    np.log1p(wallets)
    / np.log1p(
        max(
            wallets.max(),
            1
        )
    )
)


age_score = (
    np.log1p(age)
    / np.log1p(
        max(
            age.max(),
            1
        )
    )
)


# ============================================================
# 10. COUNTER EVIDENCE SCORE
# ============================================================

df["network_evidence_score"] = (
    (
        0.50 * activity_score
        +
        0.35 * wallet_score
        +
        0.15 * age_score
    )
    * 100
)


# ============================================================
# 11. RISK + NETWORK COMBINATION
# ============================================================

df["counter_risk_score"] = (
    (
        0.70
        *
        df["defender_probability"]
        *
        100
    )
    +
    (
        0.30
        *
        df["network_evidence_score"]
    )
)


# ============================================================
# 12. DECISION POLICY
# ============================================================

def decide(
    defender_probability,
    network_score
):

    # Very high ML confidence.
    if defender_probability >= 0.85:

        if network_score >= 40:

            return (
                "BLOCK",
                "High model risk with "
                "supporting historical network evidence."
            )

        return (
            "STEP-UP",
            "High model risk but limited "
            "network confirmation."
        )


    # Medium / uncertain risk.
    if defender_probability >= 0.45:

        if network_score >= 50:

            return (
                "BLOCK",
                "Moderate model risk combined "
                "with strong historical network evidence."
            )

        if network_score >= 25:

            return (
                "STEP-UP",
                "Model uncertainty with meaningful "
                "network evidence."
            )

        return (
            "STEP-UP",
            "Transaction requires additional verification."
        )


    # Low model risk.
    if network_score >= 60:

        return (
            "STEP-UP",
            "Low model risk but unusually strong "
            "historical network activity."
        )

    return (
        "ALLOW",
        "Low model risk and limited historical "
        "network evidence."
    )


# ============================================================
# 13. APPLY POLICY
# ============================================================

decisions = []

reasons = []


for row in df.itertuples():

    decision, reason = decide(
        row.defender_probability,
        row.network_evidence_score
    )

    decisions.append(
        decision
    )

    reasons.append(
        reason
    )


df["decision"] = decisions

df["decision_reason"] = reasons


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COUNTERRISK DECISION DISTRIBUTION")
print("=" * 70)

print(
    df["decision"]
    .value_counts()
)


# ============================================================
# 15. SHOW INTERESTING CASES
# ============================================================

print("\n" + "=" * 70)
print("HIGH NETWORK EVIDENCE CASES")
print("=" * 70)


interesting = (
    df[
        df["network_evidence_score"] >= 60
    ]
    .sort_values(
        "network_evidence_score",
        ascending=False
    )
)


columns = [
    "txId",
    "Time step",
    "is_fraud",
    "defender_probability",
    "network_evidence_score",
    "counter_risk_score",
    "decision"
]


print(
    interesting[
        columns
    ]
    .head(25)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE
# ============================================================

OUTPUT_FILE = (
    "../data/"
    "counterrisk_decisions.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("DECISIONS SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print("=" * 70)