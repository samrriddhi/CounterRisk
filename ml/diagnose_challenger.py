import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("../data/transactions.csv")

model = joblib.load(
    "counterrisk_defender.pkl"
)


features = [
    "amount",
    "user_avg_amount",
    "amount_ratio",
    "account_age_days",
    "is_new_device",
    "location_changed",
    "transaction_velocity",
    "receiver_risk",
    "receiver_fraud_rate",
    "known_receiver",
    "previous_fraud_count",
    "shared_device_accounts",
    "shared_ip_accounts",
    "impossible_travel",
    "payment_method",
    "location"
]


X = df[features]

y = df["is_fraud"]


# Same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Defender probabilities
probability = model.predict_proba(
    X_test
)[:, 1]


prediction = (
    probability >= 0.50
).astype(int)


results = df.loc[X_test.index].copy()

results["defender_probability"] = probability

results["defender_prediction"] = prediction


# ==================================================
# ONLY DEFENDER-POSITIVE TRANSACTIONS
# ==================================================

positive = results[
    results["defender_prediction"] == 1
]


false_positives = positive[
    positive["is_fraud"] == 0
]


true_positives = positive[
    positive["is_fraud"] == 1
]


# ==================================================
# PRINT COUNTS
# ==================================================

print()
print("========================================")
print("       CHALLENGER DIAGNOSTICS")
print("========================================")

print()

print(
    f"Defender-positive transactions: "
    f"{len(positive)}"
)

print(
    f"Actual fraud: "
    f"{len(true_positives)}"
)

print(
    f"Actual legitimate: "
    f"{len(false_positives)}"
)


# ==================================================
# FEATURE COMPARISON
# ==================================================

columns = [
    "defender_probability",
    "amount_ratio",
    "account_age_days",
    "is_new_device",
    "location_changed",
    "transaction_velocity",
    "receiver_risk",
    "receiver_fraud_rate",
    "known_receiver",
    "previous_fraud_count",
    "shared_device_accounts",
    "shared_ip_accounts",
    "impossible_travel"
]


print()
print("========================================")
print("       FALSE POSITIVE AVERAGES")
print("========================================")

print(
    false_positives[columns]
    .mean()
    .round(3)
)


print()
print("========================================")
print("       TRUE POSITIVE AVERAGES")
print("========================================")

print(
    true_positives[columns]
    .mean()
    .round(3)
)


# ==================================================
# PROBABILITY DISTRIBUTION
# ==================================================

print()
print("========================================")
print("     DEFENDER RISK DISTRIBUTION")
print("========================================")

print()

print("FALSE POSITIVES:")

print(
    false_positives[
        "defender_probability"
    ]
    .describe()
    .round(3)
)


print()
print("TRUE POSITIVES:")

print(
    true_positives[
        "defender_probability"
    ]
    .describe()
    .round(3)
)


# ==================================================
# SHOW ALL FALSE POSITIVES
# ==================================================

print()
print("========================================")
print("       FALSE POSITIVE CASES")
print("========================================")

display_columns = [
    "transaction_id",
    "defender_probability",
    "amount_ratio",
    "account_age_days",
    "is_new_device",
    "location_changed",
    "transaction_velocity",
    "receiver_risk",
    "receiver_fraud_rate",
    "known_receiver",
    "previous_fraud_count",
    "shared_device_accounts",
    "shared_ip_accounts",
    "impossible_travel"
]


print(
    false_positives[
        display_columns
    ]
    .sort_values(
        "defender_probability"
    )
    .to_string(
        index=False
    )
)


print()
print("========================================")