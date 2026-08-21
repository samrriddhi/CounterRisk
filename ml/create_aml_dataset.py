import random
import pandas as pd

random.seed(42)

OUTPUT = "../data/aml_transactions.csv"

rows = []
tx_id = 1


def add_transaction(sender, receiver, amount, scenario, step):
    global tx_id

    rows.append({
        "transaction_id": f"AML{tx_id}",
        "sender": sender,
        "receiver": receiver,
        "amount": round(amount, 2),
        "time_step": step,
        "scenario": scenario,
        "is_aml": 0 if scenario == "normal" else 1
    })

    tx_id += 1


# ============================================================
# 1. NORMAL TRANSACTIONS
# ============================================================

for _ in range(7000):

    sender = f"U{random.randint(1000, 2999)}"
    receiver = f"U{random.randint(1000, 2999)}"

    while receiver == sender:
        receiver = f"U{random.randint(1000, 2999)}"

    amount = random.uniform(100, 15000)

    add_transaction(
        sender,
        receiver,
        amount,
        "normal",
        random.randint(1, 1000)
    )


# ============================================================
# 2. MONEY MULE
#
# Multiple unrelated users send money to a mule account.
# The mule then distributes the money to other accounts.
# ============================================================

for mule_id in range(100):

    mule = f"MULE_{mule_id}"

    sources = [
        f"SOURCE_{mule_id}_{i}"
        for i in range(8)
    ]

    destinations = [
        f"DEST_{mule_id}_{i}"
        for i in range(5)
    ]

    step = 1000 + mule_id * 10

    for source in sources:

        add_transaction(
            source,
            mule,
            random.uniform(2000, 12000),
            "money_mule",
            step
        )

        step += 1

    for destination in destinations:

        add_transaction(
            mule,
            destination,
            random.uniform(1500, 10000),
            "money_mule",
            step
        )

        step += 1


# ============================================================
# 3. LAYERING
#
# Funds move through multiple intermediary accounts.
#
# A → B → C → D → E
# ============================================================

for chain_id in range(100):

    chain = [
        f"L{chain_id}_A",
        f"L{chain_id}_B",
        f"L{chain_id}_C",
        f"L{chain_id}_D",
        f"L{chain_id}_E"
    ]

    amount = random.uniform(10000, 50000)

    start_step = 2000 + chain_id * 10

    for i in range(len(chain) - 1):

        amount *= random.uniform(
            0.85,
            0.98
        )

        add_transaction(
            chain[i],
            chain[i + 1],
            amount,
            "layering",
            start_step + i
        )


# ============================================================
# 4. STRUCTURING / SMURFING
#
# Instead of one large transaction, money is divided into
# many smaller transactions.
# ============================================================

for case_id in range(100):

    sender = f"STRUCT_SOURCE_{case_id}"
    receiver = f"STRUCT_TARGET_{case_id}"

    base_step = 3000 + case_id * 10

    for i in range(8):

        amount = random.uniform(
            800,
            1900
        )

        add_transaction(
            sender,
            receiver,
            amount,
            "structuring",
            base_step + i
        )


# ============================================================
# 5. ROUND-TRIPPING
#
# Money eventually returns to the original entity.
#
# A → B → C → D → A
# ============================================================

for case_id in range(100):

    entities = [
        f"ROUND_{case_id}_A",
        f"ROUND_{case_id}_B",
        f"ROUND_{case_id}_C",
        f"ROUND_{case_id}_D"
    ]

    amount = random.uniform(
        15000,
        60000
    )

    step = 4000 + case_id * 10

    for i in range(len(entities)):

        sender = entities[i]

        receiver = entities[
            (i + 1) % len(entities)
        ]

        amount *= random.uniform(
            0.90,
            0.99
        )

        add_transaction(
            sender,
            receiver,
            amount,
            "round_tripping",
            step + i
        )


# ============================================================
# 6. FRAUD → LAUNDERING
#
# Fraud proceeds enter a network and are subsequently moved
# through intermediary accounts.
# ============================================================

for case_id in range(100):

    fraud_source = (
        f"FRAUD_SOURCE_{case_id}"
    )

    mule = (
        f"FRAUD_MULE_{case_id}"
    )

    layer_1 = (
        f"FRAUD_LAYER1_{case_id}"
    )

    layer_2 = (
        f"FRAUD_LAYER2_{case_id}"
    )

    final_account = (
        f"FRAUD_FINAL_{case_id}"
    )

    amount = random.uniform(
        20000,
        80000
    )

    start_step = (
        5000 + case_id * 10
    )

    add_transaction(
        fraud_source,
        mule,
        amount,
        "fraud_to_laundering",
        start_step
    )

    amount *= random.uniform(
        0.85,
        0.97
    )

    add_transaction(
        mule,
        layer_1,
        amount,
        "fraud_to_laundering",
        start_step + 1
    )

    amount *= random.uniform(
        0.85,
        0.97
    )

    add_transaction(
        layer_1,
        layer_2,
        amount,
        "fraud_to_laundering",
        start_step + 2
    )

    amount *= random.uniform(
        0.85,
        0.97
    )

    add_transaction(
        layer_2,
        final_account,
        amount,
        "fraud_to_laundering",
        start_step + 3
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

df = df.sort_values(
    "time_step"
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("       COUNTERRISK AML DATASET GENERATOR")
print("=" * 70)

print()
print(f"Total transactions : {len(df):,}")
print(f"AML transactions   : {df['is_aml'].sum():,}")
print(
    f"Normal transactions: "
    f"{(df['is_aml'] == 0).sum():,}"
)

print()
print("Scenario distribution:")
print(
    df["scenario"]
    .value_counts()
    .to_string()
)

print()
print("Unique senders   :", df["sender"].nunique())
print("Unique receivers :", df["receiver"].nunique())

print()
print(f"Saved to: {OUTPUT}")

print("=" * 70)