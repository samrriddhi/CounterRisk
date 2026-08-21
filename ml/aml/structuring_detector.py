import pandas as pd
import numpy as np

INPUT_PATH = "../data/aml_behavior_features.csv"
OUTPUT_PATH = "../data/aml_structuring_results_v2.csv"

THRESHOLD = 50000

print("=" * 70)
print("       COUNTERRISK — STRUCTURING / SMURFING DETECTOR V2")
print("=" * 70)

print("\nLoading transactions...")

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["sender_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# 1. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "transaction_id",
    "sender_id",
    "receiver_id",
    "amount",
    "timestamp",
    "payment_method"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 2. NEAR-THRESHOLD TRANSACTION
# ============================================================
#
# This is NOT:
#
#     amount < 50000
#
# We specifically look for transactions close to the
# monitoring threshold.
# ============================================================

df["near_threshold"] = (
    (df["amount"] >= THRESHOLD * 0.75)
    &
    (df["amount"] < THRESHOLD)
).astype(int)


# ============================================================
# 3. CREATE ORIGINAL ROW ID
# ============================================================
#
# This lets us calculate rolling features independently and
# merge them back to the exact transaction.
# ============================================================

df["_row_id"] = np.arange(len(df))


# ============================================================
# 4. 24-HOUR ROLLING FEATURES
# ============================================================

rolling_frames = []

print("\nCalculating 24-hour behavioral windows...")

for sender_id, group in df.groupby(
    "sender_id",
    sort=False
):

    group = group.sort_values(
        "timestamp"
    ).copy()

    group = group.set_index(
        "timestamp"
    )

    # --------------------------------------------------------
    # Number of transactions
    # --------------------------------------------------------

    transactions_24h = (
        group["transaction_id"]
        .rolling(
            "24h",
            min_periods=1
        )
        .count()
    )

    # --------------------------------------------------------
    # Near-threshold transactions
    # --------------------------------------------------------

    near_threshold_24h = (
        group["near_threshold"]
        .rolling(
            "24h",
            min_periods=1
        )
        .sum()
    )

    # --------------------------------------------------------
    # Total transaction value
    # --------------------------------------------------------

    amount_24h = (
        group["amount"]
        .rolling(
            "24h",
            min_periods=1
        )
        .sum()
    )

    rolling_result = pd.DataFrame({

        "_row_id":
            group["_row_id"].values,

        "transactions_24h":
            transactions_24h.values,

        "near_threshold_24h":
            near_threshold_24h.values,

        "amount_24h":
            amount_24h.values
    })

    rolling_frames.append(
        rolling_result
    )


rolling_features = pd.concat(
    rolling_frames,
    ignore_index=True
)


# ============================================================
# 5. MERGE ROLLING FEATURES BACK
# ============================================================

df = df.merge(
    rolling_features,
    on="_row_id",
    how="left"
)


# ============================================================
# 6. CLEANUP
# ============================================================

df = df.drop(
    columns=["_row_id"]
)


# ============================================================
# 7. AMOUNT SIMILARITY
# ============================================================

df["amount_deviation"] = (
    abs(
        df["amount"] - THRESHOLD
    )
    /
    THRESHOLD
)

df["amount_similarity"] = np.clip(
    1 - df["amount_deviation"],
    0,
    1
)


# ============================================================
# 8. RECEIVER DIVERSITY
# ============================================================

df["receiver_count"] = (
    df.groupby(
        "sender_id"
    )["receiver_id"]
    .transform("nunique")
)

df["receiver_diversity_score"] = np.clip(
    df["receiver_count"] / 8,
    0,
    1
)


# ============================================================
# 9. PAYMENT CHANNEL DIVERSITY
# ============================================================

df["payment_method_count"] = (
    df.groupby(
        "sender_id"
    )["payment_method"]
    .transform("nunique")
)

df["payment_channel_score"] = np.clip(
    df["payment_method_count"] / 3,
    0,
    1
)


# ============================================================
# 10. TRANSACTION FREQUENCY SCORE
# ============================================================

df["frequency_score"] = np.clip(
    df["transactions_24h"] / 10,
    0,
    1
)


# ============================================================
# 11. NEAR-THRESHOLD CONCENTRATION SCORE
# ============================================================

df["threshold_score"] = np.clip(
    df["near_threshold_24h"] / 5,
    0,
    1
)


# ============================================================
# 12. CUMULATIVE VALUE SCORE
# ============================================================

df["cumulative_score"] = np.clip(
    df["amount_24h"] / 200000,
    0,
    1
)


# ============================================================
# 13. STRUCTURING SCORE
# ============================================================
#
# Multiple behavioral signals are combined.
#
# Near-threshold concentration     25%
# Transaction frequency             20%
# Cumulative value                  20%
# Amount similarity                 15%
# Receiver diversity                10%
# Payment-channel diversity         10%
#
# ============================================================

df["structuring_score"] = (

    0.25 *
    df["threshold_score"]

    +

    0.20 *
    df["frequency_score"]

    +

    0.20 *
    df["cumulative_score"]

    +

    0.15 *
    df["amount_similarity"]

    +

    0.10 *
    df["receiver_diversity_score"]

    +

    0.10 *
    df["payment_channel_score"]
)

df["structuring_score"] = np.clip(
    df["structuring_score"],
    0,
    1
)


# ============================================================
# 14. RISK LEVEL
# ============================================================

def risk_level(score):

    if score >= 0.75:

        return "HIGH"

    elif score >= 0.50:

        return "MEDIUM"

    else:

        return "LOW"


df["risk"] = (
    df["structuring_score"]
    .apply(risk_level)
)


# ============================================================
# 15. EXPLAINABLE EVIDENCE
# ============================================================

def generate_evidence(row):

    signals = []

    # Repeated near-threshold activity

    if row["near_threshold_24h"] >= 2:

        signals.append(
            f"{int(row['near_threshold_24h'])} "
            "near-threshold transactions within 24 hours"
        )

    # High transaction frequency

    if row["transactions_24h"] >= 5:

        signals.append(
            f"{int(row['transactions_24h'])} "
            "transactions within 24 hours"
        )

    # Large cumulative amount

    if row["amount_24h"] >= 100000:

        signals.append(
            f"₹{row['amount_24h']:,.2f} "
            "cumulative transaction value within 24 hours"
        )

    # Transaction close to threshold

    if row["amount_similarity"] >= 0.85:

        signals.append(
            "transaction amount is unusually close "
            "to the monitoring threshold"
        )

    # Multiple receivers

    if row["receiver_count"] >= 5:

        signals.append(
            f"funds distributed across "
            f"{int(row['receiver_count'])} receivers"
        )

    # Multiple channels

    if row["payment_method_count"] >= 2:

        signals.append(
            "activity spans multiple payment channels"
        )

    if not signals:

        signals.append(
            "No strong structuring-specific "
            "behavioral evidence"
        )

    return "; ".join(signals)


df["evidence"] = df.apply(
    generate_evidence,
    axis=1
)


# ============================================================
# 16. RECOMMENDATION
# ============================================================

def recommendation(risk):

    if risk == "HIGH":

        return "ENHANCED_REVIEW"

    elif risk == "MEDIUM":

        return "MONITOR"

    else:

        return "NORMAL"


df["recommendation"] = (
    df["risk"]
    .apply(recommendation)
)


# ============================================================
# 17. SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 18. SUMMARY
# ============================================================

print(
    f"\nTransactions analyzed: {len(df):,}"
)

print(
    "\nRisk distribution:"
)

print(
    df["risk"]
    .value_counts()
    .to_string()
)


# ============================================================
# 19. TOP 10
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 10 STRUCTURING RISK TRANSACTIONS"
)

print(
    "=" * 70
)

top = (
    df.sort_values(
        "structuring_score",
        ascending=False
    )
    .head(10)
)


for i, (_, row) in enumerate(
    top.iterrows(),
    1
):

    print()

    print(
        f"{i}. Transaction: "
        f"{row['transaction_id']}"
    )

    print(
        f"   Sender: "
        f"{row['sender_id']}"
    )

    print(
        f"   Amount: "
        f"₹{row['amount']:,.2f}"
    )

    print(
        f"   Risk: "
        f"{row['risk']}"
    )

    print(
        f"   Structuring Score: "
        f"{row['structuring_score']:.2%}"
    )

    print(
        f"   24h Transactions: "
        f"{int(row['transactions_24h'])}"
    )

    print(
        f"   24h Value: "
        f"₹{row['amount_24h']:,.2f}"
    )

    print(
        f"   Near-threshold 24h: "
        f"{int(row['near_threshold_24h'])}"
    )

    print(
        f"   Receivers: "
        f"{int(row['receiver_count'])}"
    )

    print(
        f"   Payment Channels: "
        f"{int(row['payment_method_count'])}"
    )

    print(
        f"   Evidence: "
        f"{row['evidence']}"
    )

    print(
        f"   Recommendation: "
        f"{row['recommendation']}"
    )


# ============================================================
# 20. COMPLETE
# ============================================================

print()

print(
    f"Saved to: {OUTPUT_PATH}"
)

print(
    "=" * 70
)