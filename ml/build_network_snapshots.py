import os
import json
import pandas as pd


# ============================================================
# COUNTERRISK — NETWORK SNAPSHOT BUILDER
# ============================================================

BASE = "../data/ellipticpp"

TX_FEATURES = os.path.join(
    BASE,
    "txs_features.csv"
)

ADDR_TX_FILE = os.path.join(
    BASE,
    "AddrTx_edgelist.csv"
)

TX_ADDR_FILE = os.path.join(
    BASE,
    "TxAddr_edgelist.csv"
)

OUTPUT_FILE = (
    "../data/"
    "counterrisk_network_snapshots.json"
)


# ============================================================
# NORMALIZE IDS
# ============================================================

def normalize_tx_id(value):
    return str(
        int(
            float(value)
        )
    )


print("=" * 70)
print("        COUNTERRISK NETWORK SNAPSHOT BUILDER")
print("=" * 70)


# ============================================================
# 1. LOAD TRANSACTION TIMES
# ============================================================

print("\nLoading transaction times...")

features = pd.read_csv(
    TX_FEATURES,
    usecols=[
        "txId",
        "Time step"
    ]
)

features["txId"] = (
    features["txId"]
    .apply(normalize_tx_id)
)

features["Time step"] = pd.to_numeric(
    features["Time step"],
    errors="coerce"
)

tx_time = dict(
    zip(
        features["txId"],
        features["Time step"]
    )
)


# ============================================================
# 2. LOAD INPUT WALLETS
# ============================================================

print("Loading input wallets...")

addr_tx = pd.read_csv(
    ADDR_TX_FILE
)

addr_tx["txId"] = (
    addr_tx["txId"]
    .apply(normalize_tx_id)
)

addr_tx["input_address"] = (
    addr_tx["input_address"]
    .astype(str)
    .str.strip()
)


tx_inputs = (
    addr_tx
    .groupby("txId")["input_address"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 3. LOAD OUTPUT WALLETS
# ============================================================

print("Loading output wallets...")

tx_addr = pd.read_csv(
    TX_ADDR_FILE
)

tx_addr["txId"] = (
    tx_addr["txId"]
    .apply(normalize_tx_id)
)

tx_addr["output_address"] = (
    tx_addr["output_address"]
    .astype(str)
    .str.strip()
)


tx_outputs = (
    tx_addr
    .groupby("txId")["output_address"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 4. BUILD WALLET → TRANSACTIONS
# ============================================================

print("Building wallet index...")

wallet_transactions = {}


for tx_id, wallets in tx_inputs.items():

    for wallet in wallets:

        wallet_transactions.setdefault(
            wallet,
            []
        ).append(
            tx_id
        )


for tx_id, wallets in tx_outputs.items():

    for wallet in wallets:

        wallet_transactions.setdefault(
            wallet,
            []
        ).append(
            tx_id
        )


# Remove duplicates.

for wallet in wallet_transactions:

    wallet_transactions[wallet] = list(
        set(
            wallet_transactions[wallet]
        )
    )


# ============================================================
# 5. BUILD SNAPSHOTS
#
# Every transaction receives a compact historical view:
#
# current transaction
#     ↓
# wallets
#     ↓
# PREVIOUS transactions only
# ============================================================

print("Building historical network snapshots...")

snapshots = {}


for tx_id, current_time in tx_time.items():

    if pd.isna(current_time):
        continue

    current_time = int(
        current_time
    )


    wallets = set()

    wallets.update(
        tx_inputs.get(
            tx_id,
            []
        )
    )

    wallets.update(
        tx_outputs.get(
            tx_id,
            []
        )
    )


    previous_transactions = set()


    for wallet in wallets:

        related = wallet_transactions.get(
            wallet,
            []
        )

        for other_tx in related:

            if other_tx == tx_id:
                continue

            other_time = tx_time.get(
                other_tx
            )

            if other_time is None:
                continue

            if pd.isna(other_time):
                continue

            # STRICTLY EARLIER
            if int(other_time) < current_time:

                previous_transactions.add(
                    other_tx
                )


    # --------------------------------------------------------
    # Keep graph compact.
    # --------------------------------------------------------

    previous_transactions = list(
        previous_transactions
    )

    previous_transactions = (
        previous_transactions[:20]
    )


    snapshots[tx_id] = {

        "time_step":
            current_time,

        "wallets":
            list(wallets)[:10],

        "previous_transactions":
            previous_transactions
    }


# ============================================================
# 6. SAVE JSON
# ============================================================

print("\nSaving snapshots...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        snapshots,
        f
    )


print("\n" + "=" * 70)
print("NETWORK SNAPSHOTS COMPLETE")
print("=" * 70)

print(
    f"Transactions indexed: "
    f"{len(snapshots):,}"
)

print(
    f"Wallets indexed: "
    f"{len(wallet_transactions):,}"
)

print(
    f"Saved: {OUTPUT_FILE}"
)

print("=" * 70)