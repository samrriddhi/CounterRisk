import os
import pandas as pd


# ============================================================
# COUNTERRISK NETWORK INVESTIGATOR
# ============================================================

DATA_DIR = "../data/ellipticpp"

ADDR_TX_FILE = os.path.join(
    DATA_DIR,
    "AddrTx_edgelist.csv"
)

TX_ADDR_FILE = os.path.join(
    DATA_DIR,
    "TxAddr_edgelist.csv"
)

ADDR_ADDR_FILE = os.path.join(
    DATA_DIR,
    "AddrAddr_edgelist.csv"
)

CLASS_FILE = os.path.join(
    DATA_DIR,
    "txs_classes.csv"
)

FN_FILE = (
    "../data/"
    "counterrisk_false_negatives.csv"
)


print("=" * 70)
print("          COUNTERRISK NETWORK INVESTIGATOR")
print("=" * 70)


# ============================================================
# HELPER
# ============================================================

def normalize_tx_id(value):
    """
    Elliptic++ transaction IDs are numeric.

    Pandas can sometimes read them as floats:
        72637933.0

    We normalize both forms to:
        72637933
    """

    try:
        return str(int(float(value)))
    except:
        return str(value).strip()


# ============================================================
# 1. LOAD FALSE NEGATIVES
# ============================================================

false_negatives = pd.read_csv(
    FN_FILE
)

false_negatives["txId"] = (
    false_negatives["txId"]
    .apply(normalize_tx_id)
)

false_negative_ids = set(
    false_negatives["txId"]
)


print()
print(
    f"False negatives: "
    f"{len(false_negative_ids):,}"
)


# ============================================================
# 2. LOAD LABELS
# ============================================================

print()
print("Loading transaction labels...")


classes = pd.read_csv(
    CLASS_FILE
)

classes["txId"] = (
    classes["txId"]
    .apply(normalize_tx_id)
)

classes["class"] = (
    classes["class"]
    .astype(str)
    .str.strip()
)


# Elliptic++:
# 1 = illicit
# 2 = legitimate

illicit_transactions = set(
    classes.loc[
        classes["class"] == "1",
        "txId"
    ]
)

legitimate_transactions = set(
    classes.loc[
        classes["class"] == "2",
        "txId"
    ]
)


print(
    f"Known illicit transactions: "
    f"{len(illicit_transactions):,}"
)

print(
    f"Known legitimate transactions: "
    f"{len(legitimate_transactions):,}"
)


# ============================================================
# 3. INPUT ADDRESS → TRANSACTION
# ============================================================

print()
print("=" * 70)
print("BUILDING TRANSACTION → WALLET INDEX")
print("=" * 70)

print()
print("Reading AddrTx edges...")


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


input_addresses = {}


for row in addr_tx.itertuples(
    index=False
):

    tx_id = row.txId

    address = row.input_address

    if tx_id not in input_addresses:

        input_addresses[
            tx_id
        ] = set()

    input_addresses[
        tx_id
    ].add(address)


print(
    "Transactions with input addresses:",
    len(input_addresses)
)


# ============================================================
# 4. TRANSACTION → OUTPUT ADDRESS
# ============================================================

print()
print("Reading TxAddr edges...")


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


output_addresses = {}


for row in tx_addr.itertuples(
    index=False
):

    tx_id = row.txId

    address = row.output_address

    if tx_id not in output_addresses:

        output_addresses[
            tx_id
        ] = set()

    output_addresses[
        tx_id
    ].add(address)


print(
    "Transactions with output addresses:",
    len(output_addresses)
)


# ============================================================
# 5. CHECK OUR FALSE NEGATIVES
# ============================================================

print()
print("=" * 70)
print("CHECKING TRANSACTION ID MATCHING")
print("=" * 70)


matched_input = (
    false_negative_ids
    &
    set(input_addresses.keys())
)

matched_output = (
    false_negative_ids
    &
    set(output_addresses.keys())
)


print(
    "False negatives with input wallets:",
    len(matched_input)
)

print(
    "False negatives with output wallets:",
    len(matched_output)
)


# ============================================================
# 6. BUILD ADDRESS → TRANSACTION INDEX
# ============================================================

print()
print("=" * 70)
print("BUILDING WALLET → TRANSACTION INDEX")
print("=" * 70)


address_to_transactions = {}


for tx_id, addresses in input_addresses.items():

    for address in addresses:

        if address not in address_to_transactions:

            address_to_transactions[
                address
            ] = set()

        address_to_transactions[
            address
        ].add(tx_id)


for tx_id, addresses in output_addresses.items():

    for address in addresses:

        if address not in address_to_transactions:

            address_to_transactions[
                address
            ] = set()

        address_to_transactions[
            address
        ].add(tx_id)


print(
    "Wallets indexed:",
    len(address_to_transactions)
)


# ============================================================
# 7. BUILD WALLET → WALLET NETWORK
# ============================================================

print()
print("=" * 70)
print("BUILDING WALLET NETWORK")
print("=" * 70)

print()
print("Reading AddrAddr edges...")


addr_addr = pd.read_csv(
    ADDR_ADDR_FILE
)


addr_addr[
    "input_address"
] = (
    addr_addr[
        "input_address"
    ]
    .astype(str)
    .str.strip()
)

addr_addr[
    "output_address"
] = (
    addr_addr[
        "output_address"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. TARGET WALLETS
# ============================================================

target_addresses = set()


for tx_id in false_negative_ids:

    target_addresses.update(
        input_addresses.get(
            tx_id,
            set()
        )
    )

    target_addresses.update(
        output_addresses.get(
            tx_id,
            set()
        )
    )


print()
print(
    "Wallets belonging to missed transactions:",
    len(target_addresses)
)


# ============================================================
# 9. FIND WALLET NEIGHBORS
# ============================================================

address_neighbors = {}


for row in addr_addr.itertuples(
    index=False
):

    source = row.input_address

    target = row.output_address


    if source in target_addresses:

        if source not in address_neighbors:

            address_neighbors[
                source
            ] = set()

        address_neighbors[
            source
        ].add(target)


    if target in target_addresses:

        if target not in address_neighbors:

            address_neighbors[
                target
            ] = set()

        address_neighbors[
            target
        ].add(source)


print(
    "Wallets with actual network connections:",
    len(address_neighbors)
)


# ============================================================
# 10. INVESTIGATE TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("MISSED TRANSACTION NETWORK ANALYSIS")
print("=" * 70)


results = []


for _, row in false_negatives.iterrows():

    tx_id = str(
        row["txId"]
    )


    # --------------------------------------------------------
    # Wallets belonging to transaction
    # --------------------------------------------------------

    inputs = input_addresses.get(
        tx_id,
        set()
    )

    outputs = output_addresses.get(
        tx_id,
        set()
    )

    all_addresses = (
        inputs |
        outputs
    )


    # --------------------------------------------------------
    # Transactions sharing wallets
    # --------------------------------------------------------

    connected_transactions = set()


    for address in all_addresses:

        connected_transactions.update(
            address_to_transactions.get(
                address,
                set()
            )
        )


    connected_transactions.discard(
        tx_id
    )


    # --------------------------------------------------------
    # Illicit / legitimate neighbors
    # --------------------------------------------------------

    illicit_connected = (
        connected_transactions
        &
        illicit_transactions
    )

    legitimate_connected = (
        connected_transactions
        &
        legitimate_transactions
    )


    # --------------------------------------------------------
    # Wallet network
    # --------------------------------------------------------

    wallet_neighbors = set()


    for address in all_addresses:

        wallet_neighbors.update(
            address_neighbors.get(
                address,
                set()
            )
        )


    wallet_neighbors -= all_addresses


    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    connected_count = len(
        connected_transactions
    )

    illicit_count = len(
        illicit_connected
    )

    legitimate_count = len(
        legitimate_connected
    )

    wallet_neighbor_count = len(
        wallet_neighbors
    )


    # --------------------------------------------------------
    # Network risk
    # --------------------------------------------------------

    if illicit_count >= 5:

        network_risk = "CRITICAL"

    elif illicit_count >= 2:

        network_risk = "HIGH"

    elif illicit_count == 1:

        network_risk = "MEDIUM"

    else:

        network_risk = "LOW"


    # --------------------------------------------------------
    # Network score
    # --------------------------------------------------------

    network_score = min(
        100,
        illicit_count * 20
    )


    results.append({

        "txId":
            tx_id,

        "defender_probability":
            row[
                "fraud_probability"
            ],

        "input_wallets":
            len(inputs),

        "output_wallets":
            len(outputs),

        "wallet_neighbors":
            wallet_neighbor_count,

        "connected_transactions":
            connected_count,

        "illicit_connected":
            illicit_count,

        "legitimate_connected":
            legitimate_count,

        "network_score":
            network_score,

        "network_risk":
            network_risk
    })


# ============================================================
# 11. RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    [
        "illicit_connected",
        "wallet_neighbors",
        "defender_probability"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


print()

print(
    results_df[
        [
            "txId",
            "defender_probability",
            "connected_transactions",
            "illicit_connected",
            "wallet_neighbors",
            "network_score",
            "network_risk"
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. SUMMARY
# ============================================================

print()
print("=" * 70)
print("NETWORK EVIDENCE SUMMARY")
print("=" * 70)


critical = results_df[
    results_df["network_risk"]
    == "CRITICAL"
]

high = results_df[
    results_df["network_risk"]
    == "HIGH"
]

medium = results_df[
    results_df["network_risk"]
    == "MEDIUM"
]

low = results_df[
    results_df["network_risk"]
    == "LOW"
]


print(
    f"CRITICAL : {len(critical):,}"
)

print(
    f"HIGH     : {len(high):,}"
)

print(
    f"MEDIUM   : {len(medium):,}"
)

print(
    f"LOW      : {len(low):,}"
)


# ============================================================
# 13. SAVE
# ============================================================

output_path = (
    "../data/"
    "counterrisk_network_evidence.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print()
print("=" * 70)
print("Network evidence saved:")
print(output_path)
print("=" * 70)