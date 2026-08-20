import os
import pandas as pd


# ============================================================
# COUNTERRISK — LEAKAGE-SAFE TEMPORAL NETWORK FEATURES
# ============================================================

DATA_DIR = "../data/ellipticpp"

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "txs_features.csv"
)

ADDR_TX_FILE = os.path.join(
    DATA_DIR,
    "AddrTx_edgelist.csv"
)

TX_ADDR_FILE = os.path.join(
    DATA_DIR,
    "TxAddr_edgelist.csv"
)

OUTPUT_FILE = (
    "../data/"
    "counterrisk_temporal_network_features.csv"
)


print("=" * 70)
print("   COUNTERRISK TEMPORAL NETWORK ENGINE")
print("   LEAKAGE-SAFE VERSION")
print("=" * 70)


# ============================================================
# 1. TRANSACTION TIMES
# ============================================================

print("\nLoading transaction times...")

features = pd.read_csv(
    FEATURE_FILE,
    usecols=["txId", "Time step"]
)

features["txId"] = (
    features["txId"]
    .apply(lambda x: str(int(float(x))))
)

features["Time step"] = pd.to_numeric(
    features["Time step"],
    errors="coerce"
)

features = features.dropna(
    subset=["Time step"]
)

features = features.drop_duplicates(
    subset=["txId"]
)

print(
    "Transactions:",
    len(features)
)


# ============================================================
# 2. TRANSACTION → INPUT WALLET
# ============================================================

print("\nLoading AddrTx edges...")

addr_tx = pd.read_csv(
    ADDR_TX_FILE
)

addr_tx["txId"] = (
    addr_tx["txId"]
    .apply(lambda x: str(int(float(x))))
)

addr_tx["input_address"] = (
    addr_tx["input_address"]
    .astype(str)
    .str.strip()
)

addr_tx = addr_tx[
    [
        "txId",
        "input_address"
    ]
].drop_duplicates()

addr_tx = addr_tx.rename(
    columns={
        "input_address": "address"
    }
)

addr_tx["direction"] = "input"


# ============================================================
# 3. TRANSACTION → OUTPUT WALLET
# ============================================================

print("Loading TxAddr edges...")

tx_addr = pd.read_csv(
    TX_ADDR_FILE
)

tx_addr["txId"] = (
    tx_addr["txId"]
    .apply(lambda x: str(int(float(x))))
)

tx_addr["output_address"] = (
    tx_addr["output_address"]
    .astype(str)
    .str.strip()
)

tx_addr = tx_addr[
    [
        "txId",
        "output_address"
    ]
].drop_duplicates()

tx_addr = tx_addr.rename(
    columns={
        "output_address": "address"
    }
)

tx_addr["direction"] = "output"


# ============================================================
# 4. COMBINE
# ============================================================

print("Combining transaction-wallet relationships...")

relations = pd.concat(
    [
        addr_tx,
        tx_addr
    ],
    ignore_index=True
)

relations = relations.drop_duplicates(
    subset=[
        "txId",
        "address"
    ]
)


# ============================================================
# 5. ADD TIME
# ============================================================

relations = relations.merge(
    features,
    on="txId",
    how="inner"
)


print(
    "Transaction-wallet relationships:",
    len(relations)
)


# ============================================================
# 6. IMPORTANT:
#    ONLY PREVIOUS TIME STEPS COUNT
# ============================================================

print("\nBuilding leakage-safe historical features...")


# For each wallet + time step:
# how many transactions used that wallet?

wallet_time = (
    relations
    .groupby(
        [
            "address",
            "Time step"
        ]
    )
    .size()
    .reset_index(
        name="transactions_at_time"
    )
)


# ============================================================
# 7. CUMULATIVE ACTIVITY FROM STRICTLY EARLIER TIMES
# ============================================================

wallet_time = wallet_time.sort_values(
    [
        "address",
        "Time step"
    ]
)


# Shift BEFORE cumulative sum.
#
# This means:
#
# time 1 → 0 previous
# time 2 → activity from time 1
# time 3 → activity from time 1-2
#
# Current time is NEVER included.

wallet_time[
    "prior_transactions"
] = (
    wallet_time
    .groupby("address")[
        "transactions_at_time"
    ]
    .transform(
        lambda x:
        x.cumsum().shift(
            fill_value=0
        )
    )
)


# ============================================================
# 8. FIRST TIME WALLET WAS SEEN
# ============================================================

wallet_first_seen = (
    wallet_time
    .groupby("address")[
        "Time step"
    ]
    .min()
    .reset_index()
)

wallet_first_seen = (
    wallet_first_seen
    .rename(
        columns={
            "Time step":
                "wallet_first_seen"
        }
    )
)


# ============================================================
# 9. MAP WALLET HISTORY BACK TO RELATIONSHIPS
# ============================================================

relations = relations.merge(
    wallet_time[
        [
            "address",
            "Time step",
            "prior_transactions"
        ]
    ],
    on=[
        "address",
        "Time step"
    ],
    how="left"
)


relations = relations.merge(
    wallet_first_seen,
    on="address",
    how="left"
)


# ============================================================
# 10. TRANSACTION-LEVEL NETWORK FEATURES
# ============================================================

print("Aggregating transaction-level features...")


transaction_network = (
    relations
    .groupby("txId")
    .agg(
        prior_wallet_connections=(
            "prior_transactions",
            "sum"
        ),

        active_wallets=(
            "address",
            "nunique"
        ),

        total_wallet_count=(
            "address",
            "nunique"
        ),

        network_first_seen=(
            "wallet_first_seen",
            "min"
        )
    )
    .reset_index()
)


# ============================================================
# 11. CURRENT TRANSACTION TIME
# ============================================================

transaction_network = transaction_network.merge(
    features,
    on="txId",
    how="left"
)


# ============================================================
# 12. NETWORK AGE
# ============================================================

transaction_network[
    "network_age"
] = (
    transaction_network[
        "Time step"
    ]
    -
    transaction_network[
        "network_first_seen"
    ]
)


# A transaction occurring at the first observed
# time step has network age 0.

transaction_network[
    "network_age"
] = transaction_network[
    "network_age"
].clip(
    lower=0
)


# ============================================================
# 13. HISTORICAL NETWORK ACTIVITY
# ============================================================

transaction_network[
    "historical_network_activity"
] = (
    transaction_network[
        "prior_wallet_connections"
    ]
)


# ============================================================
# 14. CLEAN
# ============================================================

network_columns = [
    "prior_wallet_connections",
    "active_wallets",
    "total_wallet_count",
    "network_age",
    "historical_network_activity"
]


for column in network_columns:

    transaction_network[
        column
    ] = pd.to_numeric(
        transaction_network[
            column
        ],
        errors="coerce"
    ).fillna(0)


# ============================================================
# 15. KEEP ONLY WHAT WE NEED
# ============================================================

output = transaction_network[
    [
        "txId",
        "Time step",
        "prior_wallet_connections",
        "active_wallets",
        "total_wallet_count",
        "network_first_seen",
        "network_age",
        "historical_network_activity"
    ]
].copy()


# ============================================================
# 16. SAVE
# ============================================================

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL NETWORK FEATURES")
print("=" * 70)

print(
    output.head(20).to_string(
        index=False
    )
)


print("\nValidation:")

time_one = output[
    output["Time step"] == 1
]

print(
    "Time-step-1 transactions:",
    len(time_one)
)

print(
    "Max prior activity at time 1:",
    time_one[
        "prior_wallet_connections"
    ].max()
)


print("\nFeatures created:")

for column in output.columns:
    print(
        "  →",
        column
    )


print("\n" + "=" * 70)
print("Saved:")
print(OUTPUT_FILE)
print("=" * 70)