import os
import pandas as pd


# ============================================================
# COUNTERRISK — LEAKAGE-SAFE HISTORICAL FRAUD NETWORK
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
    "counterrisk_historical_fraud_network.csv"
)


print("=" * 70)
print("   COUNTERRISK HISTORICAL FRAUD NETWORK")
print("   LEAKAGE-SAFE VERSION")
print("=" * 70)


# ============================================================
# 1. LOAD TRANSACTION TIMES
# ============================================================

print("\nLoading transaction times...")

features = pd.read_csv(
    FEATURE_FILE,
    usecols=[
        "txId",
        "Time step"
    ]
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


# ============================================================
# 2. LOAD HISTORICAL LABELS
# ============================================================

print("Loading historical labels...")

classes = pd.read_csv(
    CLASS_FILE
)

classes["txId"] = (
    classes["txId"]
    .apply(lambda x: str(int(float(x))))
)

classes["class"] = pd.to_numeric(
    classes["class"],
    errors="coerce"
)

classes = classes[
    classes["class"].isin([1, 2])
].copy()

classes["is_illicit"] = (
    classes["class"] == 1
).astype(int)

print(
    "Transactions with confirmed labels:",
    len(classes)
)


# ============================================================
# 3. MERGE TIME + HISTORICAL LABEL
# ============================================================

tx_info = features.merge(
    classes[
        [
            "txId",
            "is_illicit"
        ]
    ],
    on="txId",
    how="left"
)


# ============================================================
# 4. LOAD AddrTx
# ============================================================

print("\nLoading AddrTx...")

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

addr_tx = addr_tx.rename(
    columns={
        "input_address": "address"
    }
)


# ============================================================
# 5. LOAD TxAddr
# ============================================================

print("Loading TxAddr...")

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

tx_addr = tx_addr.rename(
    columns={
        "output_address": "address"
    }
)


# ============================================================
# 6. COMBINE WALLET RELATIONSHIPS
# ============================================================

relations = pd.concat(
    [
        addr_tx[
            [
                "txId",
                "address"
            ]
        ],
        tx_addr[
            [
                "txId",
                "address"
            ]
        ]
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
# 7. ATTACH TIME + LABEL
# ============================================================

relations = relations.merge(
    tx_info,
    on="txId",
    how="left"
)


relations = relations.sort_values(
    [
        "address",
        "Time step",
        "txId"
    ]
)


# ============================================================
# 8. BUILD WALLET-TIME HISTORY
#
# THIS IS THE IMPORTANT PART:
#
# For a transaction at time t, only information from
# times < t is allowed.
# ============================================================

print(
    "\nCalculating historical wallet activity..."
)


wallet_time = (
    relations
    .groupby(
        [
            "address",
            "Time step"
        ]
    )
    .agg(

        transactions_at_time=(
            "txId",
            "nunique"
        ),

        illicit_at_time=(
            "is_illicit",
            "sum"
        )
    )
    .reset_index()
)


wallet_time = wallet_time.sort_values(
    [
        "address",
        "Time step"
    ]
)


# ============================================================
# 9. STRICTLY PRIOR TRANSACTIONS
# ============================================================

wallet_time[
    "prior_transactions"
] = (
    wallet_time
    .groupby("address")[
        "transactions_at_time"
    ]
    .cumsum()
    .groupby(
        wallet_time["address"]
    )
    .shift(
        fill_value=0
    )
)


# ============================================================
# 10. STRICTLY PRIOR ILLICIT TRANSACTIONS
# ============================================================

wallet_time[
    "prior_illicit"
] = (
    wallet_time
    .groupby("address")[
        "illicit_at_time"
    ]
    .cumsum()
    .groupby(
        wallet_time["address"]
    )
    .shift(
        fill_value=0
    )
)


# ============================================================
# 11. ATTACH HISTORY BACK TO RELATIONSHIPS
# ============================================================

relations = relations.merge(
    wallet_time[
        [
            "address",
            "Time step",
            "prior_transactions",
            "prior_illicit"
        ]
    ],
    on=[
        "address",
        "Time step"
    ],
    how="left"
)


# ============================================================
# 12. TRANSACTION-LEVEL AGGREGATION
# ============================================================

print(
    "Aggregating historical fraud evidence..."
)


network = (
    relations
    .groupby("txId")
    .agg(

        prior_connected_transactions=(
            "prior_transactions",
            "sum"
        ),

        prior_illicit_connections=(
            "prior_illicit",
            "sum"
        ),

        connected_wallets=(
            "address",
            "nunique"
        ),

        time_step=(
            "Time step",
            "first"
        )
    )
    .reset_index()
)


# ============================================================
# 13. HISTORICAL ILLICIT RATIO
# ============================================================

network[
    "prior_illicit_ratio"
] = (
    network[
        "prior_illicit_connections"
    ]
    /
    network[
        "prior_connected_transactions"
    ]
    .replace(
        0,
        1
    )
)

network[
    "prior_illicit_ratio"
] = network[
    "prior_illicit_ratio"
].clip(
    0,
    1
)


# ============================================================
# 14. CLEAN NUMBERS
# ============================================================

numeric_columns = [
    "prior_connected_transactions",
    "prior_illicit_connections",
    "connected_wallets",
    "prior_illicit_ratio"
]

for column in numeric_columns:

    network[column] = pd.to_numeric(
        network[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# 15. SAVE
# ============================================================

network.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 16. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("HISTORICAL FRAUD NETWORK FEATURES")
print("=" * 70)

print(
    network.head(20).to_string(
        index=False
    )
)

time_one = network[
    network["time_step"] == 1
]


print("\nValidation:")
print(
    "Time-step-1 transactions:",
    len(time_one)
)

print(
    "Max prior transactions at time 1:",
    time_one[
        "prior_connected_transactions"
    ].max()
)

print(
    "Max prior illicit at time 1:",
    time_one[
        "prior_illicit_connections"
    ].max()
)


print("\nFeatures created:")

for column in network.columns:

    print(
        "  →",
        column
    )


print("\n" + "=" * 70)
print("Saved:")
print(OUTPUT_FILE)
print("=" * 70)