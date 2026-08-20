import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD DATA + DEFENDER
# ============================================================

df = pd.read_csv("../data/transactions.csv")

defender = joblib.load(
    "counterrisk_defender.pkl"
)


# ============================================================
# 2. FEATURES
# ============================================================

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


# ============================================================
# 3. SAME TEST SPLIT AS DEFENDER
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. DEFENDER OUTPUT
# ============================================================

defender_probability = defender.predict_proba(
    X_test
)[:, 1]

defender_prediction = (
    defender_probability >= 0.50
).astype(int)


results = df.loc[X_test.index].copy()

results["defender_probability"] = (
    defender_probability
)

results["defender_prediction"] = (
    defender_prediction
)


# ============================================================
# 5. COUNTER-EVIDENCE
#
# The Challenger is deliberately conservative.
#
# It does NOT ask:
# "Can I find one legitimate signal?"
#
# It asks:
# "Is there a strong collection of legitimate signals
#  AND is the fraud evidence relatively weak?"
# ============================================================

def calculate_counter_evidence(row):

    score = 0
    evidence = []


    # --------------------------------------------------------
    # STRONG LEGITIMATE EVIDENCE
    # --------------------------------------------------------

    if row["known_receiver"] == 1:
        score += 2
        evidence.append("Known receiver")


    if row["account_age_days"] >= 365:
        score += 2
        evidence.append("Established account")


    if row["receiver_fraud_rate"] <= 0.25:
        score += 2
        evidence.append("Low receiver fraud history")


    if row["shared_device_accounts"] <= 5:
        score += 1
        evidence.append("Limited device sharing")


    if row["shared_ip_accounts"] <= 6:
        score += 1
        evidence.append("Limited IP sharing")


    if row["impossible_travel"] == 0:
        score += 2
        evidence.append("No impossible travel")


    # --------------------------------------------------------
    # FRAUD HISTORY
    #
    # Previous fraud is a very strong reason NOT to challenge.
    # --------------------------------------------------------

    if row["previous_fraud_count"] == 0:
        score += 2
        evidence.append("No previous fraud history")

    elif row["previous_fraud_count"] >= 2:
        score -= 4
        evidence.append("Multiple previous fraud events")

    else:
        score -= 2
        evidence.append("Previous fraud history")


    # --------------------------------------------------------
    # NETWORK RISK
    # --------------------------------------------------------

    if row["shared_device_accounts"] >= 8:
        score -= 3
        evidence.append("Heavy device sharing")


    if row["shared_ip_accounts"] >= 10:
        score -= 3
        evidence.append("Heavy IP sharing")


    # --------------------------------------------------------
    # RECEIVER RISK
    # --------------------------------------------------------

    if row["receiver_risk"] >= 0.75:
        score -= 3
        evidence.append("High receiver risk")


    if row["receiver_fraud_rate"] >= 0.40:
        score -= 3
        evidence.append("High receiver fraud rate")


    # --------------------------------------------------------
    # IMPOSSIBLE TRAVEL
    #
    # This should almost completely block a challenge.
    # --------------------------------------------------------

    if row["impossible_travel"] == 1:
        score -= 5
        evidence.append("Impossible travel")


    return score, evidence


# ============================================================
# 6. CALCULATE CHALLENGER SCORES
# ============================================================

counter_scores = []
counter_evidence = []


for _, row in X_test.iterrows():

    score, evidence = calculate_counter_evidence(
        row
    )

    counter_scores.append(score)
    counter_evidence.append(evidence)


results["counter_score"] = counter_scores

results["counter_evidence"] = counter_evidence


# ============================================================
# 7. FINAL DECISION
# ============================================================

results["final_prediction"] = (
    results["defender_prediction"]
)


# ============================================================
# 8. VERY IMPORTANT:
#
# Challenger only reviews borderline Defender decisions.
#
# Extremely high-risk transactions are NOT overturned.
# ============================================================

for index, row in results.iterrows():

    defender_risk = row["defender_probability"]

    counter_score = row["counter_score"]


    # Challenger only gets authority over this
    # borderline/high-risk region.
    #
    # It does NOT touch probabilities >= 0.90.

    borderline = (
        0.50 <= defender_risk < 0.90
    )


    strong_counter_evidence = (
        counter_score >= 9
    )


    # Additional safety conditions.

    safe_receiver = (
        row["receiver_risk"] < 0.75
    )


    no_impossible_travel = (
        row["impossible_travel"] == 0
    )


    low_network_risk = (
        row["shared_device_accounts"] < 8
        and
        row["shared_ip_accounts"] < 10
    )


    limited_fraud_history = (
        row["previous_fraud_count"] == 0
    )


    # --------------------------------------------------------
    # CHALLENGE ONLY IF EVERYTHING AGREES
    # --------------------------------------------------------

    if (
        row["defender_prediction"] == 1
        and borderline
        and strong_counter_evidence
        and safe_receiver
        and no_impossible_travel
        and low_network_risk
        and limited_fraud_history
    ):

        results.at[
            index,
            "final_prediction"
        ] = 0


# ============================================================
# 9. BASELINE METRICS
# ============================================================

baseline_cm = confusion_matrix(
    y_test,
    results["defender_prediction"]
)

baseline_tn, baseline_fp, \
baseline_fn, baseline_tp = baseline_cm.ravel()


baseline_fpr = (
    baseline_fp /
    (baseline_fp + baseline_tn)
)


baseline_recall = (
    baseline_tp /
    (baseline_tp + baseline_fn)
)


# ============================================================
# 10. FINAL METRICS
# ============================================================

final_cm = confusion_matrix(
    y_test,
    results["final_prediction"]
)

final_tn, final_fp, \
final_fn, final_tp = final_cm.ravel()


final_accuracy = accuracy_score(
    y_test,
    results["final_prediction"]
)


final_precision = precision_score(
    y_test,
    results["final_prediction"],
    zero_division=0
)


final_recall = recall_score(
    y_test,
    results["final_prediction"],
    zero_division=0
)


final_f1 = f1_score(
    y_test,
    results["final_prediction"],
    zero_division=0
)


final_fpr = (
    final_fp /
    (final_fp + final_tn)
)


# ============================================================
# 11. CHALLENGED TRANSACTIONS
# ============================================================

challenged = results[
    (
        results["defender_prediction"] == 1
    )
    &
    (
        results["final_prediction"] == 0
    )
]


correctly_overturned = challenged[
    challenged["is_fraud"] == 0
]


incorrectly_overturned = challenged[
    challenged["is_fraud"] == 1
]


# ============================================================
# 12. FALSE-POSITIVE IMPROVEMENT
# ============================================================

if baseline_fpr > 0:

    fpr_improvement = (
        (
            baseline_fpr - final_fpr
        )
        /
        baseline_fpr
    ) * 100

else:

    fpr_improvement = 0


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print()
print("========================================")
print("       COUNTERRISK CHALLENGER")
print("========================================")


print()
print("DEFENDER BASELINE")
print("----------------------------------------")

print(
    f"False positives : {baseline_fp}"
)

print(
    f"False positive rate : "
    f"{baseline_fpr:.4f}"
)

print(
    f"Fraud recall : "
    f"{baseline_recall:.4f}"
)


print()
print("CHALLENGER ACTIVITY")
print("----------------------------------------")

print(
    f"Decisions challenged : "
    f"{len(challenged)}"
)

print(
    f"Correctly overturned : "
    f"{len(correctly_overturned)}"
)

print(
    f"Incorrectly overturned : "
    f"{len(incorrectly_overturned)}"
)


print()
print("FINAL COUNTERRISK RESULTS")
print("----------------------------------------")

print(
    f"Accuracy : "
    f"{final_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{final_precision:.4f}"
)

print(
    f"Recall : "
    f"{final_recall:.4f}"
)

print(
    f"F1 Score : "
    f"{final_f1:.4f}"
)

print(
    f"False positives : "
    f"{final_fp}"
)

print(
    f"False positive rate : "
    f"{final_fpr:.4f}"
)

print(
    f"False-positive improvement : "
    f"{fpr_improvement:.2f}%"
)


# ============================================================
# 14. CONFUSION MATRICES
# ============================================================

print()
print("BASELINE CONFUSION MATRIX")
print("----------------------------------------")

print(
    baseline_cm
)


print()
print("FINAL CONFUSION MATRIX")
print("----------------------------------------")

print(
    final_cm
)


# ============================================================
# 15. SHOW CHALLENGED TRANSACTIONS
# ============================================================

print()
print("TOP CHALLENGED TRANSACTIONS")
print("----------------------------------------")


if len(challenged) > 0:

    columns = [
        "transaction_id",
        "scenario",
        "defender_probability",
        "counter_score",
        "known_receiver",
        "account_age_days",
        "previous_fraud_count",
        "receiver_risk",
        "receiver_fraud_rate",
        "shared_device_accounts",
        "shared_ip_accounts",
        "impossible_travel",
        "is_fraud"
    ]


    print(
        challenged[
            columns
        ]
        .sort_values(
            "defender_probability",
            ascending=False
        )
        .head(15)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No transactions were challenged."
    )


print()
print("========================================")