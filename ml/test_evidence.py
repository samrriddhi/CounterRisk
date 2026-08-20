import pandas as pd

from evidence_engine import analyze_transaction


df = pd.read_csv(
    "../data/transactions.csv"
)


# Pick one transaction
transaction = df.iloc[0]


result = analyze_transaction(
    transaction
)


print()
print("========================================")
print("        COUNTERRISK EVIDENCE")
print("========================================")

print()

print(
    "Fraud evidence score:",
    result["fraud_score"]
)

print(
    "Legitimacy evidence score:",
    result["legitimacy_score"]
)

print(
    "Evidence balance:",
    result["evidence_balance"]
)

print(
    "Verdict:",
    result["evidence_verdict"]
)


print()
print("EVIDENCE FOR FRAUD")
print("----------------------------------------")

for evidence in result[
    "evidence_for_fraud"
]:

    print("🚨", evidence)


print()
print("EVIDENCE AGAINST FRAUD")
print("----------------------------------------")

for evidence in result[
    "evidence_against_fraud"
]:

    print("🟢", evidence)


print()
print("========================================")