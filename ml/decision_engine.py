import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from evidence_engine import analyze_transaction


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
# 3. SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. DEFENDER RISK
# ============================================================

risk_scores = defender.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 5. DECISION ENGINE
# ============================================================

def make_decision(
    risk,
    evidence
):

    evidence_verdict = (
        evidence["evidence_verdict"]
    )

    evidence_balance = (
        evidence["evidence_balance"]
    )


    # ========================================================
    # VERY LOW RISK
    # ========================================================

    if risk < 0.30:

        return {
            "decision": "ALLOW",
            "reason":
                "Low fraud probability."
        }


    # ========================================================
    # LOW / MODERATE RISK
    # ========================================================

    if risk < 0.60:

        if evidence_verdict == "LEGITIMACY_SUPPORTED":

            return {
                "decision": "ALLOW",
                "reason":
                    "Moderate risk but strong "
                    "legitimacy evidence."
            }

        else:

            return {
                "decision": "STEP-UP",
                "reason":
                    "Moderate risk with "
                    "insufficient certainty."
            }


    # ========================================================
    # HIGH RISK
    # ========================================================

    if risk < 0.85:

        if evidence_verdict == "FRAUD_SUPPORTED":

            return {
                "decision": "BLOCK",
                "reason":
                    "High risk supported by "
                    "fraud evidence."
            }

        elif evidence_verdict == "LEGITIMACY_SUPPORTED":

            return {
                "decision": "STEP-UP",
                "reason":
                    "High risk conflicts with "
                    "strong legitimacy evidence."
            }

        else:

            return {
                "decision": "STEP-UP",
                "reason":
                    "High risk but evidence "
                    "remains conflicting."
            }


    # ========================================================
    # VERY HIGH RISK
    # ========================================================

    if evidence_verdict == "FRAUD_SUPPORTED":

        return {
            "decision": "BLOCK",
            "reason":
                "Very high risk with strong "
                "supporting fraud evidence."
        }


    # --------------------------------------------------------
    # VERY HIGH MODEL RISK BUT CONTRADICTORY EVIDENCE
    # --------------------------------------------------------

    if evidence_verdict == "LEGITIMACY_SUPPORTED":

        return {
            "decision": "STEP-UP",
            "reason":
                "Very high model risk conflicts "
                "with strong legitimacy evidence."
        }


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return {
        "decision": "STEP-UP",
        "reason":
            "Very high risk with conflicting evidence."
    }


# ============================================================
# 6. RUN COUNTER RISK
# ============================================================

results = []


for index, row in X_test.iterrows():

    risk = risk_scores[
        list(X_test.index).index(index)
    ]


    # ---------------------------------------------
    # Evidence analysis
    # ---------------------------------------------

    evidence = analyze_transaction(
        row
    )


    # ---------------------------------------------
    # Final decision
    # ---------------------------------------------

    decision = make_decision(
        risk,
        evidence
    )


    results.append({

        "transaction_id":
            df.loc[index, "transaction_id"],

        "scenario":
            df.loc[index, "scenario"],

        "risk":
            round(
                risk,
                4
            ),

        "fraud_evidence":
            evidence[
                "fraud_score"
            ],

        "legitimacy_evidence":
            evidence[
                "legitimacy_score"
            ],

        "evidence_balance":
            evidence[
                "evidence_balance"
            ],

        "evidence_verdict":
            evidence[
                "evidence_verdict"
            ],

        "decision":
            decision[
                "decision"
            ],

        "reason":
            decision[
                "reason"
            ],

        "actual":
            df.loc[index, "is_fraud"],

        "fraud_reasons":
            evidence[
                "evidence_for_fraud"
            ],

        "legitimate_reasons":
            evidence[
                "evidence_against_fraud"
            ]
    })


# ============================================================
# 7. DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "../data/counterrisk_decisions.csv",
    index=False
)


# ============================================================
# 9. SUMMARY
# ============================================================

print()
print("========================================")
print("       COUNTERRISK DECISION ENGINE")
print("========================================")


print()
print("Decision distribution:")
print(
    results_df[
        "decision"
    ].value_counts()
)


print()
print("Evidence verdict distribution:")
print(
    results_df[
        "evidence_verdict"
    ].value_counts()
)


print()
print("Scenario vs Decision:")
print(
    pd.crosstab(
        results_df["scenario"],
        results_df["decision"]
    )
)


# ============================================================
# 10. SHOW INTERESTING CASES
# ============================================================

print()
print("========================================")
print("       SAMPLE HIGH-RISK CASES")
print("========================================")


high_risk = results_df[
    results_df["risk"] >= 0.75
]


if len(high_risk) > 0:

    display_columns = [

        "transaction_id",

        "scenario",

        "risk",

        "fraud_evidence",

        "legitimacy_evidence",

        "evidence_balance",

        "evidence_verdict",

        "decision",

        "reason"
    ]


    print(
        high_risk[
            display_columns
        ]
        .sort_values(
            "risk",
            ascending=False
        )
        .head(15)
        .to_string(
            index=False
        )
    )


print()
print("========================================")
print(
    "Saved: ../data/counterrisk_decisions.csv"
)
print("========================================")