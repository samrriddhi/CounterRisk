import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

NUM_TRANSACTIONS = 20000

locations = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata"
]

payment_methods = [
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET"
]

transactions = []

# ============================================================
# ENTITY POOLS
# ============================================================

USERS = [f"U{i}" for i in range(1000, 3000)]

# Lots of infrastructure for normal users.
NORMAL_DEVICES = [f"D_NORMAL_{i}" for i in range(1000, 2900)]
NORMAL_IPS = [f"IP_NORMAL_{i}" for i in range(1000, 2900)]

RECEIVERS = [f"R{i}" for i in range(1000, 5000)]


# ============================================================
# FRAUD NETWORK POOLS
#
# These are intentionally SMALL and shared.
# This creates actual fraud clusters.
# ============================================================

FRAUD_DEVICES = [
    f"D_FRAUD_{i}"
    for i in range(1, 31)
]

FRAUD_IPS = [
    f"IP_FRAUD_{i}"
    for i in range(1, 31)
]

FRAUD_RECEIVERS = [
    f"R_FRAUD_{i}"
    for i in range(1, 51)
]


# ============================================================
# USER PROFILES
#
# Every normal user gets a stable device + IP.
# This prevents the whole graph from becoming connected.
# ============================================================

user_profiles = {}

for user in USERS:

    user_profiles[user] = {
        "device_id": random.choice(NORMAL_DEVICES),
        "ip_address": random.choice(NORMAL_IPS),
        "receiver_id": random.choice(RECEIVERS)
    }


# ============================================================
# FRAUD CLUSTERS
#
# Each cluster has a small group of users, devices,
# IPs and receivers.
# ============================================================

fraud_clusters = []

for cluster_id in range(10):

    cluster_users = random.sample(
        USERS,
        20
    )

    cluster_devices = random.sample(
        FRAUD_DEVICES,
        3
    )

    cluster_ips = random.sample(
        FRAUD_IPS,
        3
    )

    cluster_receivers = random.sample(
        FRAUD_RECEIVERS,
        5
    )

    fraud_clusters.append({

        "users": cluster_users,

        "devices": cluster_devices,

        "ips": cluster_ips,

        "receivers": cluster_receivers
    })


# ============================================================
# GENERATE TRANSACTIONS
# ============================================================

for i in range(NUM_TRANSACTIONS):

    user_id = random.choice(USERS)

    account_age_days = random.randint(
        30,
        2000
    )

    user_avg_amount = round(
        np.random.lognormal(
            mean=7,
            sigma=0.7
        ),
        2
    )

    # ========================================================
    # SCENARIO
    # ========================================================

    scenario = random.choices(

        [
            "normal",
            "suspicious_legitimate",
            "account_takeover",
            "fraud_ring",
            "velocity_attack"
        ],

        weights=[
            58,
            22,
            8,
            7,
            5
        ]
    )[0]


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    profile = user_profiles[user_id]

    device_id = profile["device_id"]

    ip_address = profile["ip_address"]

    receiver_id = profile["receiver_id"]

    amount = round(
        np.random.lognormal(
            mean=7,
            sigma=0.75
        ),
        2
    )

    is_new_device = 0

    location_changed = 0

    transaction_velocity = random.randint(
        1,
        7
    )

    receiver_risk = round(
        random.uniform(
            0.0,
            0.35
        ),
        2
    )

    previous_fraud_count = 0

    shared_device_accounts = 1

    shared_ip_accounts = 1

    receiver_fraud_rate = round(
        random.uniform(
            0.0,
            0.10
        ),
        2
    )

    known_receiver = 1

    impossible_travel = 0

    is_fraud = 0


    # ========================================================
    # NORMAL
    # ========================================================

    if scenario == "normal":

        amount = round(
            user_avg_amount *
            random.uniform(
                0.4,
                2.5
            ),
            2
        )

        is_new_device = random.choices(
            [0, 1],
            weights=[95, 5]
        )[0]

        location_changed = random.choices(
            [0, 1],
            weights=[95, 5]
        )[0]

        transaction_velocity = random.randint(
            1,
            7
        )

        receiver_risk = round(
            random.uniform(
                0.0,
                0.30
            ),
            2
        )

        previous_fraud_count = 0

        receiver_fraud_rate = round(
            random.uniform(
                0.0,
                0.08
            ),
            2
        )

        known_receiver = 1

        impossible_travel = random.choices(
            [0, 1],
            weights=[99, 1]
        )[0]

        is_fraud = 0


    # ========================================================
    # SUSPICIOUS LEGITIMATE
    #
    # Suspicious behaviour, but legitimate infrastructure.
    # ========================================================

    elif scenario == "suspicious_legitimate":

        amount = round(
            user_avg_amount *
            random.uniform(
                3.0,
                9.0
            ),
            2
        )

        is_new_device = random.choices(
            [0, 1],
            weights=[20, 80]
        )[0]

        location_changed = random.choices(
            [0, 1],
            weights=[20, 80]
        )[0]

        transaction_velocity = random.randint(
            6,
            16
        )

        receiver_risk = round(
            random.uniform(
                0.25,
                0.75
            ),
            2
        )

        previous_fraud_count = 0

        receiver_fraud_rate = round(
            random.uniform(
                0.02,
                0.30
            ),
            2
        )

        known_receiver = random.choices(
            [0, 1],
            weights=[20, 80]
        )[0]

        impossible_travel = random.choices(
            [0, 1],
            weights=[95, 5]
        )[0]

        # IMPORTANT:
        # suspicious legitimate users remain mostly isolated.

        device_id = profile["device_id"]

        ip_address = profile["ip_address"]

        receiver_id = profile["receiver_id"]

        is_fraud = 0


    # ========================================================
    # ACCOUNT TAKEOVER
    # ========================================================

    elif scenario == "account_takeover":

        cluster = random.choice(
            fraud_clusters
        )

        device_id = random.choice(
            cluster["devices"]
        )

        ip_address = random.choice(
            cluster["ips"]
        )

        receiver_id = random.choice(
            cluster["receivers"]
        )

        amount = round(
            user_avg_amount *
            random.uniform(
                2.5,
                10.0
            ),
            2
        )

        is_new_device = random.choices(
            [0, 1],
            weights=[10, 90]
        )[0]

        location_changed = random.choices(
            [0, 1],
            weights=[20, 80]
        )[0]

        transaction_velocity = random.randint(
            5,
            18
        )

        receiver_risk = round(
            random.uniform(
                0.45,
                0.95
            ),
            2
        )

        previous_fraud_count = random.choices(
            [0, 1, 2],
            weights=[40, 40, 20]
        )[0]

        shared_device_accounts = random.randint(
            4,
            12
        )

        shared_ip_accounts = random.randint(
            4,
            15
        )

        receiver_fraud_rate = round(
            random.uniform(
                0.15,
                0.75
            ),
            2
        )

        known_receiver = random.choices(
            [0, 1],
            weights=[70, 30]
        )[0]

        impossible_travel = random.choices(
            [0, 1],
            weights=[60, 40]
        )[0]

        is_fraud = 1


    # ========================================================
    # FRAUD RING
    # ========================================================

    elif scenario == "fraud_ring":

        cluster = random.choice(
            fraud_clusters
        )

        device_id = random.choice(
            cluster["devices"]
        )

        ip_address = random.choice(
            cluster["ips"]
        )

        receiver_id = random.choice(
            cluster["receivers"]
        )

        amount = round(
            random.uniform(
                3000,
                70000
            ),
            2
        )

        is_new_device = random.choices(
            [0, 1],
            weights=[20, 80]
        )[0]

        location_changed = random.choices(
            [0, 1],
            weights=[25, 75]
        )[0]

        transaction_velocity = random.randint(
            5,
            18
        )

        receiver_risk = round(
            random.uniform(
                0.55,
                0.98
            ),
            2
        )

        previous_fraud_count = random.choices(
            [0, 1, 2, 3],
            weights=[15, 30, 35, 20]
        )[0]

        shared_device_accounts = random.randint(
            8,
            20
        )

        shared_ip_accounts = random.randint(
            8,
            25
        )

        receiver_fraud_rate = round(
            random.uniform(
                0.30,
                0.90
            ),
            2
        )

        known_receiver = random.choices(
            [0, 1],
            weights=[80, 20]
        )[0]

        impossible_travel = random.choices(
            [0, 1],
            weights=[55, 45]
        )[0]

        is_fraud = 1


    # ========================================================
    # VELOCITY ATTACK
    # ========================================================

    else:

        cluster = random.choice(
            fraud_clusters
        )

        device_id = random.choice(
            cluster["devices"]
        )

        ip_address = random.choice(
            cluster["ips"]
        )

        receiver_id = random.choice(
            cluster["receivers"]
        )

        amount = round(
            random.uniform(
                300,
                15000
            ),
            2
        )

        is_new_device = random.choices(
            [0, 1],
            weights=[25, 75]
        )[0]

        location_changed = random.choices(
            [0, 1],
            weights=[30, 70]
        )[0]

        transaction_velocity = random.randint(
            12,
            25
        )

        receiver_risk = round(
            random.uniform(
                0.35,
                0.90
            ),
            2
        )

        previous_fraud_count = random.choices(
            [0, 1, 2],
            weights=[40, 40, 20]
        )[0]

        shared_device_accounts = random.randint(
            5,
            15
        )

        shared_ip_accounts = random.randint(
            5,
            18
        )

        receiver_fraud_rate = round(
            random.uniform(
                0.15,
                0.75
            ),
            2
        )

        known_receiver = random.choices(
            [0, 1],
            weights=[50, 50]
        )[0]

        impossible_travel = random.choices(
            [0, 1],
            weights=[70, 30]
        )[0]

        is_fraud = 1


    # ========================================================
    # DERIVED FEATURE
    # ========================================================

    amount_ratio = round(
        amount /
        max(
            user_avg_amount,
            1
        ),
        2
    )


    # ========================================================
    # SAVE
    # ========================================================

    transactions.append({

        "transaction_id":
            f"T{i + 1}",

        "user_id":
            user_id,

        "device_id":
            device_id,

        "ip_address":
            ip_address,

        "receiver_id":
            receiver_id,

        "amount":
            amount,

        "user_avg_amount":
            round(
                user_avg_amount,
                2
            ),

        "amount_ratio":
            amount_ratio,

        "account_age_days":
            account_age_days,

        "is_new_device":
            is_new_device,

        "location_changed":
            location_changed,

        "transaction_velocity":
            transaction_velocity,

        "receiver_risk":
            receiver_risk,

        "previous_fraud_count":
            previous_fraud_count,

        "shared_device_accounts":
            shared_device_accounts,

        "shared_ip_accounts":
            shared_ip_accounts,

        "receiver_fraud_rate":
            receiver_fraud_rate,

        "known_receiver":
            known_receiver,

        "impossible_travel":
            impossible_travel,

        "payment_method":
            random.choice(
                payment_methods
            ),

        "location":
            random.choice(
                locations
            ),

        "scenario":
            scenario,

        "is_fraud":
            is_fraud
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    transactions
)


# ============================================================
# SAVE
# ============================================================

output_path = "../data/transactions.csv"

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("       COUNTERRISK DATASET")
print("========================================")

print(
    f"Total transactions : {len(df)}"
)

print(
    f"Fraudulent         : "
    f"{df['is_fraud'].sum()}"
)

print(
    f"Legitimate         : "
    f"{len(df) - df['is_fraud'].sum()}"
)

print()
print("Scenario distribution:")
print()

print(
    df["scenario"].value_counts()
)

print()
print("Fraud distribution by scenario:")
print()

print(
    pd.crosstab(
        df["scenario"],
        df["is_fraud"]
    )
)

print()
print("Unique network entities:")
print()

print(
    "Users     :",
    df["user_id"].nunique()
)

print(
    "Devices   :",
    df["device_id"].nunique()
)

print(
    "IPs       :",
    df["ip_address"].nunique()
)

print(
    "Receivers :",
    df["receiver_id"].nunique()
)

print()
print("Fraud network infrastructure:")
print()

print(
    "Fraud devices :",
    len(FRAUD_DEVICES)
)

print(
    "Fraud IPs     :",
    len(FRAUD_IPS)
)

print(
    "Fraud receivers:",
    len(FRAUD_RECEIVERS)
)

print()
print("Saved to:")
print(output_path)

print("========================================")