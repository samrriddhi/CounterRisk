import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# COUNTERRISK — AML DATASET V2
# ============================================================

random.seed(42)
np.random.seed(42)

NUM_TRANSACTIONS = 20000

OUTPUT_PATH = "../data/aml_transactions_v2.csv"

START_DATE = datetime(2026, 1, 1)


# ============================================================
# ENTITY POOLS
# ============================================================

USERS = [f"U{i}" for i in range(1000, 4000)]

RECEIVERS = [f"R{i}" for i in range(1000, 5000)]

DEVICES = [f"D{i}" for i in range(1000, 5000)]

IPS = [f"IP{i}" for i in range(1000, 5000)]

BRANCHES = [f"BR{i}" for i in range(1, 101)]

LOCATIONS = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
]

PAYMENT_METHODS = [
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET",
    "BANK_TRANSFER",
]


# ============================================================
# FRAUD / AML NETWORK POOLS
# ============================================================

AML_USERS = random.sample(USERS, 300)

MULE_USERS = AML_USERS[:60]

STRUCTURING_USERS = AML_USERS[60:120]

LAYERING_USERS = AML_USERS[120:180]

CASH_USERS = AML_USERS[180:240]

MIXED_RISK_USERS = AML_USERS[240:300]


# ============================================================
# USER PROFILES
# ============================================================

profiles = {}

for user in USERS:

    profiles[user] = {

        "account_age_days":
            random.randint(30, 2500),

        "avg_amount":
            round(
                np.random.lognormal(
                    mean=7,
                    sigma=0.65
                ),
                2
            ),

        "device":
            random.choice(DEVICES),

        "ip":
            random.choice(IPS),

        "location":
            random.choice(LOCATIONS),

        "branch":
            random.choice(BRANCHES),

        "receiver":
            random.choice(RECEIVERS)
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_timestamp():

    day_offset = random.randint(0, 179)

    seconds = random.randint(
        0,
        23 * 3600 + 59 * 60 + 59
    )

    return (
        START_DATE
        + timedelta(days=day_offset, seconds=seconds)
    )


def normal_amount(user):

    avg = profiles[user]["avg_amount"]

    return round(
        avg * random.uniform(0.5, 2.0),
        2
    )


def suspicious_amount(user):

    avg = profiles[user]["avg_amount"]

    return round(
        avg * random.uniform(2.5, 8.0),
        2
    )


# ============================================================
# TRANSACTION STORAGE
# ============================================================

transactions = []


# ============================================================
# NORMAL TRANSACTIONS
# ============================================================

for i in range(NUM_TRANSACTIONS):

    user = random.choice(USERS)

    scenario = "normal"

    timestamp = random_timestamp()

    amount = normal_amount(user)

    receiver = profiles[user]["receiver"]

    transaction_type = random.choice([
        "UPI",
        "CARD",
        "BANK_TRANSFER",
        "WALLET"
    ])

    transactions.append({

        "transaction_id":
            f"AT{i + 1}",

        "timestamp":
            timestamp,

        "sender_id":
            user,

        "receiver_id":
            receiver,

        "amount":
            amount,

        "payment_method":
            transaction_type,

        "transaction_type":
            "DIGITAL",

        "sender_account_age":
            profiles[user]["account_age_days"],

        "sender_avg_amount":
            profiles[user]["avg_amount"],

        "sender_expected_daily_volume":
            round(
                profiles[user]["avg_amount"] * 5,
                2
            ),

        "sender_location":
            profiles[user]["location"],

        "receiver_location":
            random.choice(LOCATIONS),

        "device_id":
            profiles[user]["device"],

        "ip_address":
            profiles[user]["ip"],

        "branch_id":
            profiles[user]["branch"],

        "is_cash":
            0,

        "scenario":
            scenario,

        "is_aml":
            0
    })


# ============================================================
# MONEY MULE PATTERNS
# ============================================================

for user in MULE_USERS:

    base_time = random_timestamp()

    # Several unrelated senders
    senders = random.sample(
        USERS,
        random.randint(6, 12)
    )

    mule_amounts = []

    for j, sender in enumerate(senders):

        amount = round(
            random.uniform(
                8000,
                45000
            ),
            2
        )

        mule_amounts.append(amount)

        transactions.append({

            "transaction_id":
                f"MULE_IN_{user}_{j}",

            "timestamp":
                base_time
                + timedelta(
                    minutes=j * random.randint(3, 15)
                ),

            "sender_id":
                sender,

            "receiver_id":
                user,

            "amount":
                amount,

            "payment_method":
                random.choice(PAYMENT_METHODS),

            "transaction_type":
                "DIGITAL",

            "sender_account_age":
                profiles[sender]["account_age_days"],

            "sender_avg_amount":
                profiles[sender]["avg_amount"],

            "sender_expected_daily_volume":
                profiles[sender]["avg_amount"] * 5,

            "sender_location":
                profiles[sender]["location"],

            "receiver_location":
                profiles[user]["location"],

            "device_id":
                profiles[user]["device"],

            "ip_address":
                profiles[user]["ip"],

            "branch_id":
                profiles[user]["branch"],

            "is_cash":
                0,

            "scenario":
                "money_mule",

            "is_aml":
                1
        })


    # Quickly forward most of the money
    outbound_total = sum(mule_amounts) * random.uniform(
        0.85,
        0.98
    )

    receivers = random.sample(
        RECEIVERS,
        random.randint(5, 10)
    )

    remaining = outbound_total

    for j, receiver in enumerate(receivers):

        if j == len(receivers) - 1:

            amount = max(
                remaining,
                0
            )

        else:

            amount = round(
                outbound_total
                * random.uniform(
                    0.05,
                    0.25
                ),
                2
            )

            amount = min(
                amount,
                remaining
            )

        remaining -= amount

        transactions.append({

            "transaction_id":
                f"MULE_OUT_{user}_{j}",

            "timestamp":
                base_time
                + timedelta(
                    minutes=random.randint(
                        10,
                        90
                    )
                ),

            "sender_id":
                user,

            "receiver_id":
                receiver,

            "amount":
                round(amount, 2),

            "payment_method":
                random.choice(PAYMENT_METHODS),

            "transaction_type":
                "DIGITAL",

            "sender_account_age":
                profiles[user]["account_age_days"],

            "sender_avg_amount":
                profiles[user]["avg_amount"],

            "sender_expected_daily_volume":
                profiles[user]["avg_amount"] * 5,

            "sender_location":
                profiles[user]["location"],

            "receiver_location":
                random.choice(LOCATIONS),

            "device_id":
                profiles[user]["device"],

            "ip_address":
                profiles[user]["ip"],

            "branch_id":
                profiles[user]["branch"],

            "is_cash":
                0,

            "scenario":
                "money_mule",

            "is_aml":
                1
        })


# ============================================================
# STRUCTURING
# ============================================================

for user in STRUCTURING_USERS:

    base_time = random_timestamp()

    target_amount = round(
        random.uniform(
            40000,
            150000
        ),
        2
    )

    pieces = random.randint(
        6,
        14
    )

    receiver = profiles[user]["receiver"]

    remaining = target_amount

    for j in range(pieces):

        if j == pieces - 1:

            amount = remaining

        else:

            amount = round(
                target_amount / pieces
                * random.uniform(
                    0.80,
                    1.20
                ),
                2
            )

            amount = min(
                amount,
                remaining
            )

        remaining -= amount

        transactions.append({

            "transaction_id":
                f"STRUCT_{user}_{j}",

            "timestamp":
                base_time
                + timedelta(
                    minutes=j * random.randint(
                        5,
                        30
                    )
                ),

            "sender_id":
                user,

            "receiver_id":
                receiver,

            "amount":
                round(amount, 2),

            "payment_method":
                random.choice([
                    "UPI",
                    "WALLET",
                    "BANK_TRANSFER"
                ]),

            "transaction_type":
                "DIGITAL",

            "sender_account_age":
                profiles[user]["account_age_days"],

            "sender_avg_amount":
                profiles[user]["avg_amount"],

            "sender_expected_daily_volume":
                profiles[user]["avg_amount"] * 5,

            "sender_location":
                profiles[user]["location"],

            "receiver_location":
                random.choice(LOCATIONS),

            "device_id":
                profiles[user]["device"],

            "ip_address":
                profiles[user]["ip"],

            "branch_id":
                profiles[user]["branch"],

            "is_cash":
                0,

            "scenario":
                "structuring",

            "is_aml":
                1
        })


# ============================================================
# LAYERING
# ============================================================

for chain_id in range(20):

    chain_users = random.sample(
        LAYERING_USERS,
        7
    )

    base_time = random_timestamp()

    amount = random.uniform(
        50000,
        200000
    )

    for hop in range(
        len(chain_users) - 1
    ):

        sender = chain_users[hop]

        receiver = chain_users[hop + 1]

        amount *= random.uniform(
            0.88,
            0.98
        )

        transactions.append({

            "transaction_id":
                f"LAYER_{chain_id}_{hop}",

            "timestamp":
                base_time
                + timedelta(
                    minutes=random.randint(
                        5,
                        45
                    )
                    * hop
                ),

            "sender_id":
                sender,

            "receiver_id":
                receiver,

            "amount":
                round(amount, 2),

            "payment_method":
                "BANK_TRANSFER",

            "transaction_type":
                "DIGITAL",

            "sender_account_age":
                profiles[sender]["account_age_days"],

            "sender_avg_amount":
                profiles[sender]["avg_amount"],

            "sender_expected_daily_volume":
                profiles[sender]["avg_amount"] * 5,

            "sender_location":
                profiles[sender]["location"],

            "receiver_location":
                profiles[receiver]["location"],

            "device_id":
                profiles[sender]["device"],

            "ip_address":
                profiles[sender]["ip"],

            "branch_id":
                profiles[sender]["branch"],

            "is_cash":
                0,

            "scenario":
                "layering",

            "is_aml":
                1
        })


# ============================================================
# CASH LAUNDERING
# ============================================================

for user in CASH_USERS:

    base_time = random_timestamp()

    for j in range(
        random.randint(
            5,
            10
        )
    ):

        amount = round(
            random.uniform(
                40000,
                200000
            ),
            2
        )

        transactions.append({

            "transaction_id":
                f"CASH_{user}_{j}",

            "timestamp":
                base_time
                + timedelta(
                    hours=j * random.randint(
                        2,
                        12
                    )
                ),

            "sender_id":
                user,

            "receiver_id":
                user,

            "amount":
                amount,

            "payment_method":
                "CASH",

            "transaction_type":
                "CASH_DEPOSIT",

            "sender_account_age":
                profiles[user]["account_age_days"],

            "sender_avg_amount":
                profiles[user]["avg_amount"],

            "sender_expected_daily_volume":
                profiles[user]["avg_amount"] * 5,

            "sender_location":
                profiles[user]["location"],

            "receiver_location":
                profiles[user]["location"],

            "device_id":
                profiles[user]["device"],

            "ip_address":
                profiles[user]["ip"],

            "branch_id":
                profiles[user]["branch"],

            "is_cash":
                1,

            "scenario":
                "cash_laundering",

            "is_aml":
                1
        })


# ============================================================
# MIXED / NORMAL-LOOKING AML ACTIVITY
# ============================================================

for user in MIXED_RISK_USERS:

    base_time = random_timestamp()

    receiver = random.choice(
        RECEIVERS
    )

    for j in range(
        random.randint(
            3,
            6
        )
    ):

        transactions.append({

            "transaction_id":
                f"MIXED_{user}_{j}",

            "timestamp":
                base_time
                + timedelta(
                    hours=j
                ),

            "sender_id":
                user,

            "receiver_id":
                receiver,

            "amount":
                round(
                    random.uniform(
                        5000,
                        30000
                    ),
                    2
                ),

            "payment_method":
                random.choice(
                    PAYMENT_METHODS
                ),

            "transaction_type":
                "DIGITAL",

            "sender_account_age":
                profiles[user]["account_age_days"],

            "sender_avg_amount":
                profiles[user]["avg_amount"],

            "sender_expected_daily_volume":
                profiles[user]["avg_amount"] * 5,

            "sender_location":
                profiles[user]["location"],

            "receiver_location":
                random.choice(LOCATIONS),

            "device_id":
                profiles[user]["device"],

            "ip_address":
                profiles[user]["ip"],

            "branch_id":
                profiles[user]["branch"],

            "is_cash":
                0,

            "scenario":
                "mixed_risk",

            "is_aml":
                1
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    transactions
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# DERIVED FEATURES
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["date"] = df["timestamp"].dt.date

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
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

print("=" * 70)

print(
    "       COUNTERRISK — AML DATASET V2"
)

print("=" * 70)

print()

print(
    f"Total transactions : {len(df):,}"
)

print(
    f"AML transactions   : {df['is_aml'].sum():,}"
)

print(
    f"Normal transactions: {(df['is_aml'] == 0).sum():,}"
)

print()

print("Scenario distribution:")

print(
    df["scenario"]
    .value_counts()
    .to_string()
)

print()

print(
    f"Unique senders   : {df['sender_id'].nunique():,}"
)

print(
    f"Unique receivers : {df['receiver_id'].nunique():,}"
)

print()

print(
    "Time range:"
)

print(
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)

print()

print(
    f"Saved to: {OUTPUT_PATH}"
)

print("=" * 70)