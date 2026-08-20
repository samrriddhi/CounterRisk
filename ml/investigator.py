import os
import sys
import pandas as pd
import numpy as np
import joblib


# ============================================================
# COUNTERRISK — AI INVESTIGATOR
#
# Input:
#   transaction ID
#
# Output:
#   defender risk
#   historical network evidence
#   evidence summary
#   recommended action
# ============================================================

DATA_DIR = "../data/ellipticpp"

TX_FEATURE_FILE = os.path.join(
    DATA_DIR,
    "txs_features.csv"
)

NETWORK_FILE = (
    "../data/"
    "counterrisk_historical_fraud_network.csv"
)

MODEL_FILE = "counterrisk_v3.pkl"


# ============================================================
# 1. LOAD MODEL
# ============================================================

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]

feature_columns = bundle["features"]

train_medians = bundle["train_medians"]

time_column = bundle["time_column"]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading transaction data...")

tx_df = pd.read_csv(
    TX_FEATURE_FILE
)

tx_df["txId"] = (
    tx_df["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


print("Loading network evidence...")

network_df = pd.read_csv(
    NETWORK_FILE
)

network_df["txId"] = (
    network_df["txId"]
    .apply(
        lambda x:
        str(int(float(x)))
    )
)


# ============================================================
# 3. MERGE
# ============================================================

df = tx_df.merge(
    network_df,
    on="txId",
    how="left"
)


# ============================================================
# 4. CLEAN
# ============================================================

for column in feature_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# 5. INVESTIGATION FUNCTION
# ============================================================

def investigate(tx_id):

    tx_id = str(tx_id).strip()


    # --------------------------------------------------------
    # Find transaction
    # --------------------------------------------------------

    matches = df[
        df["txId"] == tx_id
    ]


    if len(matches) == 0:

        print()
        print(
            f"Transaction {tx_id} not found."
        )

        return


    row = matches.iloc[0]


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    X = pd.DataFrame(
        [
            row[feature_columns]
        ]
    )


    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        train_medians
    )


    # --------------------------------------------------------
    # DEFENDER
    # --------------------------------------------------------

    defender_probability = model.predict_proba(
        X
    )[0][1]


    # --------------------------------------------------------
    # HISTORICAL NETWORK
    # --------------------------------------------------------

    prior_connections = int(
        row.get(
            "prior_connected_transactions",
            0
        )
    )

    prior_illicit = int(
        row.get(
            "prior_illicit_connections",
            0
        )
    )

    connected_wallets = int(
        row.get(
            "connected_wallets",
            0
        )
    )

    prior_illicit_ratio = float(
        row.get(
            "prior_illicit_ratio",
            0
        )
    )


    # --------------------------------------------------------
    # NETWORK INTERPRETATION
    # --------------------------------------------------------

    network_reasons = []


    if prior_connections == 0:

        network_verdict = (
            "NO HISTORICAL NETWORK ACTIVITY"
        )

        network_reasons.append(
            "No prior connected transactions were available."
        )

    else:

        if prior_illicit_ratio >= 0.50:

            network_verdict = (
                "STRONG HISTORICAL FRAUD SIGNAL"
            )

            network_reasons.append(
                f"{prior_illicit} of "
                f"{prior_connections} prior connected "
                f"transactions were historically illicit."
            )

        elif prior_illicit_ratio >= 0.20:

            network_verdict = (
                "MODERATE HISTORICAL FRAUD SIGNAL"
            )

            network_reasons.append(
                f"Historical illicit activity represents "
                f"{prior_illicit_ratio:.1%} of prior "
                f"connected activity."
            )

        elif prior_illicit > 0:

            network_verdict = (
                "WEAK HISTORICAL FRAUD SIGNAL"
            )

            network_reasons.append(
                f"{prior_illicit} prior illicit connection(s) "
                "were observed."
            )

        else:

            network_verdict = (
                "NO CONFIRMED HISTORICAL FRAUD SIGNAL"
            )

            network_reasons.append(
                "Prior connected activity contains "
                "no confirmed illicit transactions."
            )


    if connected_wallets >= 10:

        network_reasons.append(
            f"Transaction is associated with "
            f"{connected_wallets} wallets."
        )

    elif connected_wallets > 0:

        network_reasons.append(
            f"Transaction is associated with "
            f"{connected_wallets} wallet(s)."
        )


    # --------------------------------------------------------
    # DECISION POLICY
    #
    # This is deliberately conservative.
    # Network evidence supports the decision;
    # it does not automatically override the ML model.
    # --------------------------------------------------------

    if defender_probability >= 0.85:

        if prior_illicit_ratio >= 0.20:

            decision = "BLOCK"

            decision_reason = (
                "Very high model risk is supported by "
                "historical illicit network evidence."
            )

        else:

            decision = "STEP-UP"

            decision_reason = (
                "High model risk, but historical "
                "network evidence is insufficient "
                "for an automatic block."
            )


    elif defender_probability >= 0.45:

        if prior_illicit_ratio >= 0.20:

            decision = "BLOCK"

            decision_reason = (
                "Borderline model risk is materially "
                "strengthened by historical illicit "
                "network activity."
            )

        elif prior_illicit > 0:

            decision = "STEP-UP"

            decision_reason = (
                "Borderline model risk with some "
                "historical illicit network evidence."
            )

        else:

            decision = "STEP-UP"

            decision_reason = (
                "Borderline model risk requires "
                "additional verification."
            )


    else:

        if prior_illicit_ratio >= 0.50:

            decision = "STEP-UP"

            decision_reason = (
                "Low model risk conflicts with strong "
                "historical network fraud evidence."
            )

        elif prior_illicit_ratio >= 0.20:

            decision = "STEP-UP"

            decision_reason = (
                "Low model risk but meaningful "
                "historical network concern."
            )

        else:

            decision = "ALLOW"

            decision_reason = (
                "Low model risk and limited "
                "historical fraud evidence."
            )


    # ========================================================
    # INVESTIGATION REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("                  COUNTERRISK INVESTIGATION")
    print("=" * 72)

    print()
    print(f"Transaction ID        : {tx_id}")

    print(
        f"Time step             : "
        f"{row[time_column]}"
    )

    print(
        f"Defender probability  : "
        f"{defender_probability:.2%}"
    )

    print()
    print("-" * 72)
    print("HISTORICAL NETWORK EVIDENCE")
    print("-" * 72)

    print(
        f"Prior connected transactions : "
        f"{prior_connections}"
    )

    print(
        f"Prior illicit connections    : "
        f"{prior_illicit}"
    )

    print(
        f"Connected wallets            : "
        f"{connected_wallets}"
    )

    print(
        f"Historical illicit ratio     : "
        f"{prior_illicit_ratio:.2%}"
    )

    print(
        f"Network verdict              : "
        f"{network_verdict}"
    )

    print()
    print("Evidence:")

    for reason in network_reasons:

        print(
            f"  • {reason}"
        )


    print()
    print("-" * 72)
    print("COUNTERRISK DECISION")
    print("-" * 72)

    print(
        f"Recommended action : "
        f"{decision}"
    )

    print(
        f"Reason             : "
        f"{decision_reason}"
    )

    print()
    print("=" * 72)


# ============================================================
# 6. COMMAND-LINE USAGE
# ============================================================

if len(sys.argv) > 1:

    investigate(
        sys.argv[1]
    )

else:

    print()
    print(
        "Usage:"
    )

    print(
        "python investigator.py <transaction_id>"
    )

    print()
    print(
        "Example:"
    )

    print(
        "python investigator.py 72637933"
    )