import pandas as pd
import networkx as nx


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "../data/transactions.csv"
)


# ============================================================
# CREATE GRAPH
# ============================================================

G = nx.Graph()


# ============================================================
# ADD TRANSACTION / ENTITY NODES
# ============================================================

for _, row in df.iterrows():

    transaction = f"TX:{row['transaction_id']}"
    user = f"USER:{row['user_id']}"
    device = f"DEVICE:{row['device_id']}"
    ip = f"IP:{row['ip_address']}"
    receiver = f"RECEIVER:{row['receiver_id']}"


    # --------------------------------------------------------
    # Transaction
    # --------------------------------------------------------

    G.add_node(
        transaction,
        type="transaction",
        is_fraud=int(row["is_fraud"])
    )


    # --------------------------------------------------------
    # User
    # --------------------------------------------------------

    G.add_node(
        user,
        type="user"
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    G.add_node(
        device,
        type="device"
    )


    # --------------------------------------------------------
    # IP
    # --------------------------------------------------------

    G.add_node(
        ip,
        type="ip"
    )


    # --------------------------------------------------------
    # Receiver
    # --------------------------------------------------------

    G.add_node(
        receiver,
        type="receiver"
    )


    # --------------------------------------------------------
    # Relationships
    # --------------------------------------------------------

    G.add_edge(
        transaction,
        user,
        relation="BELONGS_TO"
    )

    G.add_edge(
        transaction,
        device,
        relation="USES_DEVICE"
    )

    G.add_edge(
        transaction,
        ip,
        relation="USES_IP"
    )

    G.add_edge(
        transaction,
        receiver,
        relation="PAYS"
    )


# ============================================================
# INVESTIGATE TRANSACTION
# ============================================================

def investigate_transaction(
    transaction_id
):

    transaction = f"TX:{transaction_id}"


    if transaction not in G:

        print(
            "Transaction not found."
        )

        return


    print()
    print("========================================")
    print(
        f"INVESTIGATION: {transaction_id}"
    )
    print("========================================")


    # --------------------------------------------------------
    # Find direct entities
    # --------------------------------------------------------

    neighbors = list(
        G.neighbors(transaction)
    )


    print()
    print("Direct connections:")
    print()


    for neighbor in neighbors:

        relation = G[
            transaction
        ][
            neighbor
        ]["relation"]


        print(
            f"{relation:15} → {neighbor}"
        )


    # --------------------------------------------------------
    # Find connected users through DEVICE / IP
    # --------------------------------------------------------

    connected_users = set()


    for entity in neighbors:

        entity_type = G.nodes[
            entity
        ]["type"]


        # Only investigate infrastructure.
        if entity_type not in [
            "device",
            "ip"
        ]:

            continue


        # ----------------------------------------------------
        # Find transactions using this infrastructure
        # ----------------------------------------------------

        related_transactions = [

            node

            for node in G.neighbors(
                entity
            )

            if G.nodes[node]["type"]
            == "transaction"
        ]


        for related_tx in related_transactions:

            # ------------------------------------------------
            # Don't count the transaction itself.
            # ------------------------------------------------

            if related_tx == transaction:

                continue


            # ------------------------------------------------
            # Find the user associated with that transaction.
            # ------------------------------------------------

            for user_node in G.neighbors(
                related_tx
            ):

                if G.nodes[
                    user_node
                ]["type"] == "user":

                    connected_users.add(
                        user_node
                    )


    # ========================================================
    # NETWORK ANALYSIS
    # ========================================================

    print()

    print(
        "Users sharing infrastructure:"
    )

    if not connected_users:

        print(
            "  None"
        )

    else:

        for user in sorted(
            connected_users
        ):

            print(
                " ",
                user
            )


    # --------------------------------------------------------
    # Count suspicious connected transactions
    # --------------------------------------------------------

    suspicious_transactions = set()


    for entity in neighbors:

        entity_type = G.nodes[
            entity
        ]["type"]


        if entity_type not in [
            "device",
            "ip"
        ]:

            continue


        for related_tx in G.neighbors(
            entity
        ):

            if related_tx == transaction:

                continue


            if (
                G.nodes[
                    related_tx
                ]["type"]
                == "transaction"
            ):

                tx_data = df[
                    df["transaction_id"]
                    ==
                    related_tx.replace(
                        "TX:",
                        ""
                    )
                ]


                if len(tx_data) == 0:

                    continue


                tx_row = tx_data.iloc[0]


                # Only count actual fraud.
                if tx_row["is_fraud"] == 1:

                    suspicious_transactions.add(
                        related_tx
                    )


    # ========================================================
    # FINAL NETWORK RISK
    # ========================================================

    network_risk = len(
        suspicious_transactions
    )


    print()

    print(
        "Suspicious connected transactions:",
        network_risk
    )


    if network_risk >= 5:

        print(
            "Network assessment: HIGH RISK"
        )

    elif network_risk >= 2:

        print(
            "Network assessment: MODERATE RISK"
        )

    else:

        print(
            "Network assessment: LOW RISK"
        )


    print()
    print("========================================")


# ============================================================
# GRAPH SUMMARY
# ============================================================

print()
print("========================================")
print("       COUNTERRISK FRAUD GRAPH")
print("========================================")

print()

print(
    "Total nodes:",
    G.number_of_nodes()
)

print(
    "Total relationships:",
    G.number_of_edges()
)


# ============================================================
# FIND A REAL FRAUD TRANSACTION
# ============================================================

fraud_transactions = df[
    df["is_fraud"] == 1
]


sample_transaction = (
    fraud_transactions
    .iloc[0]
    ["transaction_id"]
)


investigate_transaction(
    sample_transaction
)