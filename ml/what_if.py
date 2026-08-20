import os
import sys
import pandas as pd
import numpy as np
import joblib


# ============================================================
# COUNTERRISK WHAT-IF SIMULATOR
# REAL v3 MODEL
# ============================================================

DATA_DIR = "../data/ellipticpp"

TX_FEATURE_FILE = os.path.join(
    DATA_DIR,
    "txs_features.csv"
)

NETWORK_FILE = (
    "../data/"
    "counterrisk_historical_fraud_network.csv"
)

MODEL_FILE = "counterrisk_v3.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CounterRisk v3...")

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]
feature_columns = bundle["features"]
train_medians = bundle["train_medians"]


# ============================================================
# LOAD TRANSACTION FEATURES
# ============================================================

print("Loading transaction data...")

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
# LOAD NETWORK FEATURES
# ============================================================

print("Loading network evidence...")

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


# ============================================================
# MERGE
# ============================================================

df = tx_df.merge(
    network_df,
    on="txId",
    how="left"
)


# ============================================================
# CLEAN MODEL FEATURES
# ============================================================

for column in feature_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# DECISION SIMULATOR
# ============================================================

def simulate(
    defender_probability,
    prior_illicit_ratio,
    high_threshold=0.85,
    medium_threshold=0.45,
    network_threshold=0.20
):

    # Very high model risk
    if defender_probability >= high_threshold:

        if prior_illicit_ratio >= network_threshold:
            return "BLOCK"

        return "STEP-UP"


    # Borderline model risk
    if defender_probability >= medium_threshold:

        if prior_illicit_ratio >= network_threshold:
            return "BLOCK"

        return "STEP-UP"


    # Low model risk but strong historical
    # fraud network evidence
    if prior_illicit_ratio >= 0.50:

        return "STEP-UP"


    return "ALLOW"


# ============================================================
# WHAT-IF
# ============================================================

def what_if(
    tx_id,
    policies
):

    tx_id = str(tx_id).strip()

    matches = df[
        df["txId"] == tx_id
    ]

    if len(matches) == 0:

        print(
            f"Transaction {tx_id} not found."
        )

        return

    row = matches.iloc[0]


    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    X = pd.DataFrame(
        [
            row[feature_columns]
        ]
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        train_medians
    )


    # ========================================================
    # REAL DEFENDER PROBABILITY
    # ========================================================

    defender_probability = model.predict_proba(
        X
    )[0][1]


    # ========================================================
    # NETWORK EVIDENCE
    # ========================================================

    prior_illicit_ratio = float(
        row.get(
            "prior_illicit_ratio",
            0
        )
    )

    prior_illicit_connections = int(
        row.get(
            "prior_illicit_connections",
            0
        )
    )

    prior_connected_transactions = int(
        row.get(
            "prior_connected_transactions",
            0
        )
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 70)
    print("              COUNTERRISK WHAT-IF")
    print("=" * 70)

    print()
    print(
        f"Transaction ID              : {tx_id}"
    )

    print(
        f"Defender probability        : "
        f"{defender_probability:.2%}"
    )

    print(
        f"Prior connected transactions: "
        f"{prior_connected_transactions}"
    )

    print(
        f"Prior illicit connections   : "
        f"{prior_illicit_connections}"
    )

    print(
        f"Historical illicit ratio    : "
        f"{prior_illicit_ratio:.2%}"
    )


    # ========================================================
    # SIMULATE POLICIES
    # ========================================================

    print()
    print("POLICY SIMULATIONS")
    print("-" * 70)


    for policy in policies:

        action = simulate(

            defender_probability,

            prior_illicit_ratio,

            high_threshold=policy[
                "high"
            ],

            medium_threshold=policy[
                "medium"
            ],

            network_threshold=policy[
                "network"
            ]
        )


        print()

        print(
            f"Policy: {policy['name']}"
        )

        print(
            f"  High threshold    : "
            f"{policy['high']:.0%}"
        )

        print(
            f"  Medium threshold  : "
            f"{policy['medium']:.0%}"
        )

        print(
            f"  Network threshold : "
            f"{policy['network']:.0%}"
        )

        print(
            f"  → ACTION: {action}"
        )


    print()
    print("=" * 70)


# ============================================================
# POLICIES
# ============================================================

policies = [

    {
        "name": "Conservative",

        "high": 0.90,

        "medium": 0.55,

        "network": 0.30
    },

    {
        "name": "Balanced",

        "high": 0.85,

        "medium": 0.45,

        "network": 0.20
    },

    {
        "name": "Sensitive",

        "high": 0.75,

        "medium": 0.35,

        "network": 0.10
    }
]


# ============================================================
# COMMAND LINE
# ============================================================

if len(sys.argv) > 1:

    transaction_id = sys.argv[1]

else:

    transaction_id = "10000476"


what_if(
    transaction_id,
    policies
)