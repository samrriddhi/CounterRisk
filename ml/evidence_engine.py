import pandas as pd


def analyze_transaction(row):

    evidence_for_fraud = []
    evidence_against_fraud = []

    fraud_score = 0
    legitimacy_score = 0


    # ==========================================
    # TRANSACTION BEHAVIOUR
    # ==========================================

    if row["amount_ratio"] >= 5:

        fraud_score += 2

        evidence_for_fraud.append(
            "Transaction amount is significantly above the user's normal spending."
        )


    elif row["amount_ratio"] >= 3:

        fraud_score += 1

        evidence_for_fraud.append(
            "Transaction amount is unusually high for this user."
        )


    # ==========================================
    # DEVICE
    # ==========================================

    if row["is_new_device"] == 1:

        fraud_score += 2

        evidence_for_fraud.append(
            "Transaction originated from a new device."
        )

    else:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "Transaction originated from a known device."
        )


    # ==========================================
    # LOCATION
    # ==========================================

    if row["location_changed"] == 1:

        fraud_score += 1

        evidence_for_fraud.append(
            "Transaction location differs from recent behaviour."
        )

    else:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "Transaction location is consistent with recent behaviour."
        )


    # ==========================================
    # TRANSACTION VELOCITY
    # ==========================================

    if row["transaction_velocity"] >= 15:

        fraud_score += 3

        evidence_for_fraud.append(
            "Unusually high transaction velocity detected."
        )

    elif row["transaction_velocity"] >= 10:

        fraud_score += 2

        evidence_for_fraud.append(
            "Elevated transaction velocity detected."
        )


    else:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "Transaction velocity is within a normal range."
        )


    # ==========================================
    # RECEIVER
    # ==========================================

    if row["receiver_risk"] >= 0.75:

        fraud_score += 3

        evidence_for_fraud.append(
            "Receiver has a high historical risk score."
        )

    elif row["receiver_risk"] >= 0.50:

        fraud_score += 1

        evidence_for_fraud.append(
            "Receiver has elevated risk."
        )

    else:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "Receiver has relatively low historical risk."
        )


    # ==========================================
    # KNOWN RECEIVER
    # ==========================================

    if row["known_receiver"] == 1:

        legitimacy_score += 2

        evidence_against_fraud.append(
            "Receiver is known to the customer."
        )

    else:

        fraud_score += 2

        evidence_for_fraud.append(
            "Receiver is not previously known to the customer."
        )


    # ==========================================
    # PREVIOUS FRAUD
    # ==========================================

    if row["previous_fraud_count"] > 0:

        fraud_score += 3

        evidence_for_fraud.append(
            "Customer has previous fraud-related history."
        )

    else:

        legitimacy_score += 2

        evidence_against_fraud.append(
            "No previous fraud history detected."
        )


    # ==========================================
    # DEVICE NETWORK
    # ==========================================

    if row["shared_device_accounts"] >= 8:

        fraud_score += 3

        evidence_for_fraud.append(
            "Device is associated with many accounts."
        )

    elif row["shared_device_accounts"] <= 3:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "Device has limited account associations."
        )


    # ==========================================
    # IP NETWORK
    # ==========================================

    if row["shared_ip_accounts"] >= 10:

        fraud_score += 3

        evidence_for_fraud.append(
            "IP address is associated with many accounts."
        )

    elif row["shared_ip_accounts"] <= 4:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "IP has limited account associations."
        )


    # ==========================================
    # RECEIVER FRAUD HISTORY
    # ==========================================

    if row["receiver_fraud_rate"] >= 0.40:

        fraud_score += 3

        evidence_for_fraud.append(
            "Receiver has a high historical fraud rate."
        )

    elif row["receiver_fraud_rate"] <= 0.15:

        legitimacy_score += 2

        evidence_against_fraud.append(
            "Receiver has a low historical fraud rate."
        )


    # ==========================================
    # IMPOSSIBLE TRAVEL
    # ==========================================

    if row["impossible_travel"] == 1:

        fraud_score += 4

        evidence_for_fraud.append(
            "Impossible travel pattern detected."
        )

    else:

        legitimacy_score += 1

        evidence_against_fraud.append(
            "No impossible travel detected."
        )


    # ==========================================
    # FINAL EVIDENCE BALANCE
    # ==========================================

    evidence_balance = (
        legitimacy_score - fraud_score
    )


    if evidence_balance >= 4:

        evidence_verdict = "LEGITIMACY_SUPPORTED"

    elif evidence_balance <= -4:

        evidence_verdict = "FRAUD_SUPPORTED"

    else:

        evidence_verdict = "CONFLICTING_EVIDENCE"


    return {

        "fraud_score": fraud_score,

        "legitimacy_score": legitimacy_score,

        "evidence_balance": evidence_balance,

        "evidence_verdict": evidence_verdict,

        "evidence_for_fraud":
            evidence_for_fraud,

        "evidence_against_fraud":
            evidence_against_fraud
    }