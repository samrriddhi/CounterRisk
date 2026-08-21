import pandas as pd
import numpy as np

INPUT_PATH = "../data/aml_transactions_v2.csv"
OUTPUT_PATH = "../data/aml_behavior_features.csv"


print("=" * 70)
print("       COUNTERRISK — AML BEHAVIOR ENGINE")
print("=" * 70)

print("\nLoading transactions...")

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# BASIC TEMPORAL FEATURES
# ============================================================

df["date"] = df["timestamp"].dt.date

df["hour"] = df["timestamp"].dt.hour

df["amount"] = pd.to_numeric(df["amount"])


# ============================================================
# ACCOUNT-LEVEL HISTORICAL FEATURES
#
# IMPORTANT:
# These are calculated using transactions BEFORE the current
# transaction, preventing future information leakage.
# ============================================================

df["incoming_count"] = 0
df["outgoing_count"] = 0
df["incoming_amount"] = 0.0
df["outgoing_amount"] = 0.0

df["unique_senders"] = 0
df["unique_receivers"] = 0

df["rapid_outflow_ratio"] = 0.0
df["daily_transaction_count"] = 0
df["daily_amount"] = 0.0

df["cash_frequency"] = 0.0
df["cash_to_digital_ratio"] = 0.0

df["fraud_link_score"] = 0.0


# ============================================================
# HISTORICAL STATE
# ============================================================

incoming_history = {}
outgoing_history = {}

incoming_amount_history = {}
outgoing_amount_history = {}

sender_history = {}
receiver_history = {}

daily_history = {}

cash_history = {}
digital_history = {}

fraud_link_history = {}


# ============================================================
# PROCESS CHRONOLOGICALLY
# ============================================================

for idx, row in df.iterrows():

    sender = row["sender_id"]
    receiver = row["receiver_id"]

    timestamp = row["timestamp"]

    amount = float(row["amount"])

    day = timestamp.date()

    # --------------------------------------------------------
    # Current receiver's historical incoming activity
    # --------------------------------------------------------

    inc_count = incoming_history.get(receiver, 0)

    inc_amount = incoming_amount_history.get(
        receiver,
        0.0
    )

    unique_senders = len(
        sender_history.get(
            receiver,
            set()
        )
    )

    # --------------------------------------------------------
    # Current sender's historical outgoing activity
    # --------------------------------------------------------

    out_count = outgoing_history.get(sender, 0)

    out_amount = outgoing_amount_history.get(
        sender,
        0.0
    )

    unique_receivers = len(
        receiver_history.get(
            sender,
            set()
        )
    )

    # --------------------------------------------------------
    # Daily activity
    # --------------------------------------------------------

    daily_key = (
        sender,
        day
    )

    daily_transactions = daily_history.get(
        daily_key,
        0
    )

    daily_amount = (
        daily_amount_history.get(
            daily_key,
            0.0
        )
        if "daily_amount_history" in locals()
        else 0.0
    )

    # --------------------------------------------------------
    # Cash activity
    # --------------------------------------------------------

    cash_count = cash_history.get(
        sender,
        0
    )

    digital_count = digital_history.get(
        sender,
        0
    )

    total_previous = (
        cash_count
        + digital_count
    )

    # --------------------------------------------------------
    # Fraud-linked activity
    # --------------------------------------------------------

    fraud_links = fraud_link_history.get(
        receiver,
        0
    )

    # ========================================================
    # WRITE FEATURES
    # ========================================================

    df.at[idx, "incoming_count"] = inc_count

    df.at[idx, "outgoing_count"] = out_count

    df.at[idx, "incoming_amount"] = inc_amount

    df.at[idx, "outgoing_amount"] = out_amount

    df.at[idx, "unique_senders"] = unique_senders

    df.at[idx, "unique_receivers"] = unique_receivers

    df.at[idx, "daily_transaction_count"] = (
        daily_transactions
    )

    df.at[idx, "daily_amount"] = (
        daily_amount
    )

    df.at[idx, "cash_frequency"] = (
        cash_count / max(total_previous, 1)
    )

    df.at[idx, "cash_to_digital_ratio"] = (
        cash_count / max(digital_count, 1)
    )

    df.at[idx, "fraud_link_score"] = (
        fraud_links
    )

    # ========================================================
    # RAPID OUTFLOW SIGNAL
    #
    # How much of the historical incoming money has already
    # been followed by outgoing activity.
    # ========================================================

    if inc_amount > 0:

        df.at[idx, "rapid_outflow_ratio"] = min(
            out_amount / inc_amount,
            1.0
        )

    # ========================================================
    # UPDATE HISTORICAL STATE
    #
    # Only AFTER calculating the features.
    # ========================================================

    incoming_history[receiver] = (
        incoming_history.get(receiver, 0)
        + 1
    )

    incoming_amount_history[receiver] = (
        incoming_amount_history.get(
            receiver,
            0.0
        )
        + amount
    )

    outgoing_history[sender] = (
        outgoing_history.get(sender, 0)
        + 1
    )

    outgoing_amount_history[sender] = (
        outgoing_amount_history.get(
            sender,
            0.0
        )
        + amount
    )

    sender_history.setdefault(
        receiver,
        set()
    ).add(sender)

    receiver_history.setdefault(
        sender,
        set()
    ).add(receiver)

    daily_history[daily_key] = (
        daily_history.get(
            daily_key,
            0
        )
        + 1
    )

    if "daily_amount_history" not in locals():

        daily_amount_history = {}

    daily_amount_history[daily_key] = (
        daily_amount_history.get(
            daily_key,
            0.0
        )
        + amount
    )

    if row["is_cash"] == 1:

        cash_history[sender] = (
            cash_history.get(sender, 0)
            + 1
        )

    else:

        digital_history[sender] = (
            digital_history.get(sender, 0)
            + 1
        )

    # Existing AML labels allow us to create a
    # development-time fraud/AML connection signal.
    #
    # This is NOT used as a production feature.
    if row["is_aml"] == 1:

        fraud_link_history.setdefault(
            receiver,
            0
        )

        fraud_link_history[receiver] += 1


# ============================================================
# ADD BEHAVIORAL RATIOS
# ============================================================

df["incoming_outgoing_ratio"] = (
    df["incoming_amount"]
    /
    df["outgoing_amount"].replace(
        0,
        np.nan
    )
)

df["incoming_outgoing_ratio"] = (
    df["incoming_outgoing_ratio"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


df["counterparty_diversity"] = (
    df["unique_senders"]
    +
    df["unique_receivers"]
)


df["transaction_intensity"] = (
    df["daily_transaction_count"]
    /
    df["daily_amount"].replace(
        0,
        np.nan
    )
)

df["transaction_intensity"] = (
    df["transaction_intensity"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# AMOUNT DEVIATION
# ============================================================

df["amount_deviation"] = (
    df["amount"]
    /
    df["sender_avg_amount"].replace(
        0,
        np.nan
    )
)

df["amount_deviation"] = (
    df["amount_deviation"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# STRUCTURING SIGNAL
#
# Detect repeated transactions with relatively similar
# amounts during the same day.
# ============================================================

df["structuring_signal"] = 0.0

grouped = df.groupby(
    [
        "sender_id",
        "date"
    ]
)

for (_, _), group in grouped:

    if len(group) < 3:
        continue

    amounts = group["amount"].values

    mean_amount = np.mean(amounts)

    if mean_amount == 0:
        continue

    coefficient_variation = (
        np.std(amounts)
        /
        mean_amount
    )

    if coefficient_variation < 0.25:

        df.loc[
            group.index,
            "structuring_signal"
        ] = min(
            len(group) / 10,
            1.0
        )


# ============================================================
# LAYERING PRE-SIGNAL
#
# Multiple counterparties + repeated movement.
# Full graph analysis will happen in the layering detector.
# ============================================================

df["layering_signal"] = (

    (
        df["unique_receivers"] >= 3
    ).astype(int)

    +

    (
        df["unique_senders"] >= 3
    ).astype(int)

    +

    (
        df["rapid_outflow_ratio"] >= 0.7
    ).astype(int)

) / 3


# ============================================================
# MONEY MULE PRE-SIGNAL
# ============================================================

df["mule_signal"] = (

    (
        df["unique_senders"] >= 4
    ).astype(int)

    +

    (
        df["unique_receivers"] >= 4
    ).astype(int)

    +

    (
        df["rapid_outflow_ratio"] >= 0.7
    ).astype(int)

    +

    (
        df["incoming_count"] >= 5
    ).astype(int)

) / 4


# ============================================================
# CASH LAUNDERING SIGNAL
# ============================================================

df["cash_signal"] = (

    (
        df["cash_frequency"] >= 0.30
    ).astype(int)

    +

    (
        df["cash_to_digital_ratio"] >= 0.50
    ).astype(int)

) / 2


# ============================================================
# FINAL BEHAVIOR RISK SCORE
# ============================================================

df["behavior_risk_score"] = (

    0.25 * df["mule_signal"]

    +

    0.20 * df["structuring_signal"]

    +

    0.25 * df["layering_signal"]

    +

    0.15 * df["cash_signal"]

    +

    0.15 * (
        df["fraud_link_score"]
        /
        max(
            df["fraud_link_score"].max(),
            1
        )
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nBehavior engine completed.")

print(
    f"Features generated: {len(df.columns)}"
)

print(
    f"Output rows: {len(df):,}"
)

print("\nKey signals:")

print(
    df[
        [
            "mule_signal",
            "structuring_signal",
            "layering_signal",
            "cash_signal",
            "behavior_risk_score"
        ]
    ]
    .describe()
    .round(3)
    .to_string()
)

print()

print(
    f"Saved to: {OUTPUT_PATH}"
)

print("=" * 70)