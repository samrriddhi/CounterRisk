import pandas as pd
import numpy as np

INPUT_PATH = "../data/aml_behavior_features.csv"
OUTPUT_PATH = "../data/aml_mule_results_v2.csv"

print("=" * 70)
print("       COUNTERRISK — MONEY MULE DETECTOR V2")
print("=" * 70)

print("\nLoading behavioral features...")

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ============================================================
# ACCOUNT-LEVEL FEATURES
# ============================================================

account = df.groupby("sender_id").agg(
    transactions=("transaction_id", "count"),
    total_sent=("amount", "sum"),
    unique_receivers=("receiver_id", "nunique"),
    avg_transaction=("amount", "mean"),
    max_transaction=("amount", "max"),
    aml_transactions=("is_aml", "sum")
).reset_index()

# Incoming activity
incoming = df.groupby("receiver_id").agg(
    incoming_transactions=("transaction_id", "count"),
    total_received=("amount", "sum"),
    unique_senders=("sender_id", "nunique")
).reset_index()

incoming = incoming.rename(
    columns={"receiver_id": "sender_id"}
)

account = account.merge(
    incoming,
    on="sender_id",
    how="outer"
).fillna(0)


# ============================================================
# CORE MONEY-MULE SIGNALS
# ============================================================

# Many unrelated people sending money into the account
account["sender_diversity_score"] = np.clip(
    account["unique_senders"] / 10,
    0,
    1
)

# Account distributes money to many different destinations
account["receiver_diversity_score"] = np.clip(
    account["unique_receivers"] / 10,
    0,
    1
)

# Incoming and outgoing money are relatively balanced.
# A mule often receives money and rapidly distributes it.
account["flow_balance"] = (
    np.minimum(
        account["total_received"],
        account["total_sent"]
    )
    /
    np.maximum(
        account["total_received"],
        account["total_sent"],
    ).replace(0, np.nan)
).fillna(0)


# ============================================================
# TRANSIT / RAPID REDISTRIBUTION SIGNAL
# ============================================================

account["transit_ratio"] = np.minimum(
    account["total_sent"],
    account["total_received"]
) / np.maximum(
    account["total_received"],
    1
)

account["transit_ratio"] = np.clip(
    account["transit_ratio"],
    0,
    1
)


# ============================================================
# COUNTERPARTY DIVERSITY
# ============================================================

account["counterparty_diversity"] = (
    account["unique_senders"]
    +
    account["unique_receivers"]
)

account["counterparty_score"] = np.clip(
    account["counterparty_diversity"] / 15,
    0,
    1
)


# ============================================================
# VOLUME SIGNAL
# ============================================================

account["volume_score"] = np.clip(
    account["transactions"] / 20,
    0,
    1
)


# ============================================================
# AML BEHAVIOR SIGNAL
# ============================================================

account["aml_history_score"] = np.clip(
    account["aml_transactions"] /
    np.maximum(account["transactions"], 1),
    0,
    1
)


# ============================================================
# MONEY MULE SCORE
#
# We deliberately DON'T rely on one rule.
# ============================================================

account["mule_score"] = (

    0.22 * account["sender_diversity_score"]

    +

    0.18 * account["receiver_diversity_score"]

    +

    0.20 * account["flow_balance"]

    +

    0.18 * account["transit_ratio"]

    +

    0.12 * account["counterparty_score"]

    +

    0.10 * account["volume_score"]
)


account["mule_score"] = np.clip(
    account["mule_score"],
    0,
    1
)


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(score):

    if score >= 0.75:
        return "HIGH"

    if score >= 0.50:
        return "MEDIUM"

    return "LOW"


account["risk"] = account[
    "mule_score"
].apply(risk_level)


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def generate_evidence(row):

    evidence = []

    if row["unique_senders"] >= 5:

        evidence.append(
            f"Received funds from "
            f"{int(row['unique_senders'])} "
            f"distinct senders."
        )

    if row["unique_receivers"] >= 5:

        evidence.append(
            f"Distributed funds across "
            f"{int(row['unique_receivers'])} "
            f"distinct receivers."
        )

    if row["flow_balance"] >= 0.70:

        evidence.append(
            "Incoming and outgoing value are "
            "closely balanced, suggesting transit behavior."
        )

    if row["transit_ratio"] >= 0.70:

        evidence.append(
            "Large proportion of received value "
            "is subsequently redistributed."
        )

    if row["counterparty_diversity"] >= 10:

        evidence.append(
            "High counterparty diversity."
        )

    if row["transactions"] >= 15:

        evidence.append(
            "Elevated transaction activity."
        )

    if not evidence:

        evidence.append(
            "No strong mule-specific behavioral evidence."
        )

    return " ".join(evidence)


account["evidence"] = account.apply(
    generate_evidence,
    axis=1
)


# ============================================================
# RECOMMENDATION
# ============================================================

def recommendation(risk):

    if risk == "HIGH":
        return "ENHANCED_REVIEW"

    if risk == "MEDIUM":
        return "MONITOR"

    return "NORMAL"


account["recommendation"] = account[
    "risk"
].apply(recommendation)


# ============================================================
# SORT
# ============================================================

account = account.sort_values(
    "mule_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

account.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print(
    f"\nAccounts analyzed: {len(account):,}"
)

print(
    "\nRisk distribution:"
)

print(
    account["risk"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 70)
print("TOP 10 MONEY MULE RISK ACCOUNTS")
print("=" * 70)

for i, row in account.head(10).iterrows():

    print()

    print(
        f"{i + 1}. User: {row['sender_id']}"
    )

    print(
        f"   Risk: {row['risk']}"
    )

    print(
        f"   Mule Score: "
        f"{row['mule_score']:.2%}"
    )

    print(
        f"   Transactions: "
        f"{int(row['transactions'])}"
    )

    print(
        f"   Incoming senders: "
        f"{int(row['unique_senders'])}"
    )

    print(
        f"   Outgoing receivers: "
        f"{int(row['unique_receivers'])}"
    )

    print(
        f"   Total received: "
        f"₹{row['total_received']:,.2f}"
    )

    print(
        f"   Total sent: "
        f"₹{row['total_sent']:,.2f}"
    )

    print(
        f"   Evidence: "
        f"{row['evidence']}"
    )

    print(
        f"   Recommendation: "
        f"{row['recommendation']}"
    )


print()

print(
    f"Saved results to: {OUTPUT_PATH}"
)

print("=" * 70)