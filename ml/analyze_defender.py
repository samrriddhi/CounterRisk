import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("../data/transactions.csv")

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


# ==========================================
# RECREATE SAME TEST SET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD DEFENDER
# ==========================================

model = joblib.load("counterrisk_defender.pkl")


# ==========================================
# PREDICT
# ==========================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# RESTORE TEST METADATA
# ==========================================

test_data = df.loc[X_test.index].copy()

test_data["predicted_fraud"] = predictions

test_data["risk_score"] = probabilities * 100


# ==========================================
# FALSE POSITIVES
# ==========================================

false_positives = test_data[
    (test_data["is_fraud"] == 0) &
    (test_data["predicted_fraud"] == 1)
]


print("\n========================================")
print("       DEFENDER ERROR ANALYSIS")
print("========================================")

print(f"\nTotal test transactions: {len(test_data)}")

print(f"False positives: {len(false_positives)}")


# ==========================================
# FALSE POSITIVE SCENARIOS
# ==========================================

print("\nFalse positives by scenario:")

print(
    false_positives["scenario"]
    .value_counts()
)


# ==========================================
# SUSPICIOUS LEGITIMATE ANALYSIS
# ==========================================

suspicious_legit = test_data[
    test_data["scenario"] == "suspicious_legitimate"
]


suspicious_legit_fp = suspicious_legit[
    suspicious_legit["predicted_fraud"] == 1
]


print("\n========================================")
print("   SUSPICIOUS LEGITIMATE ANALYSIS")
print("========================================")

print(
    f"Suspicious legitimate transactions: "
    f"{len(suspicious_legit)}"
)

print(
    f"Flagged by Defender: "
    f"{len(suspicious_legit_fp)}"
)

if len(suspicious_legit) > 0:

    rate = (
        len(suspicious_legit_fp)
        / len(suspicious_legit)
    )

    print(
        f"False-positive rate for this group: "
        f"{rate:.4f}"
    )


# ==========================================
# SHOW EXAMPLES
# ==========================================

print("\nTop suspicious legitimate transactions flagged:")

columns = [
    "transaction_id",
    "amount",
    "amount_ratio",
    "is_new_device",
    "location_changed",
    "transaction_velocity",
    "receiver_risk",
    "known_receiver",
    "shared_device_accounts",
    "shared_ip_accounts",
    "risk_score"
]


print(
    suspicious_legit_fp[
        columns
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\n========================================")