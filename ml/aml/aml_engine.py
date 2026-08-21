import pandas as pd
import numpy as np


class MoneyMuleDetector:

    def __init__(self):
        self.name = "CounterRisk Money Mule Detector"

    def detect(self, df):

        required_columns = [
            "user_id",
            "receiver_id",
            "amount",
            "is_fraud",
            "receiver_fraud_rate",
            "transaction_velocity"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        results = []

        # ----------------------------------------------------
        # USER-LEVEL AGGREGATION
        # ----------------------------------------------------

        grouped = df.groupby("user_id")

        for user_id, group in grouped:

            transactions = len(group)

            unique_receivers = (
                group["receiver_id"]
                .nunique()
            )

            total_amount = (
                group["amount"]
                .sum()
            )

            fraud_count = (
                group["is_fraud"]
                .sum()
            )

            fraud_ratio = (
                fraud_count / transactions
                if transactions > 0
                else 0
            )

            avg_receiver_fraud_rate = (
                group["receiver_fraud_rate"]
                .mean()
            )

            avg_velocity = (
                group["transaction_velocity"]
                .mean()
            )

            # ------------------------------------------------
            # SIGNAL 1 — COUNTERPARTY DIVERSITY
            # ------------------------------------------------

            diversity_score = min(
                unique_receivers / 7,
                1.0
            )

            # ------------------------------------------------
            # SIGNAL 2 — FRAUD CONNECTION
            # ------------------------------------------------

            fraud_connection_score = min(
                fraud_ratio / 0.50,
                1.0
            )

            # ------------------------------------------------
            # SIGNAL 3 — RECEIVER RISK
            # ------------------------------------------------

            receiver_risk_score = min(
                avg_receiver_fraud_rate / 0.50,
                1.0
            )

            # ------------------------------------------------
            # SIGNAL 4 — TRANSACTION VELOCITY
            #
            # We do NOT claim this proves rapid movement.
            # It is only a behavioural signal because the
            # current dataset does not give us exact timestamps.
            # ------------------------------------------------

            velocity_score = min(
                avg_velocity / 15,
                1.0
            )

            # ------------------------------------------------
            # COMBINED MULE SCORE
            # ------------------------------------------------

            mule_score = (
                0.30 * diversity_score
                + 0.30 * fraud_connection_score
                + 0.25 * receiver_risk_score
                + 0.15 * velocity_score
            )

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if mule_score >= 0.70:
                risk_level = "HIGH"

            elif mule_score >= 0.40:
                risk_level = "MEDIUM"

            else:
                risk_level = "LOW"

            # ------------------------------------------------
            # EXPLAINABLE EVIDENCE
            # ------------------------------------------------

            evidence = []

            if unique_receivers >= 7:
                evidence.append(
                    f"High counterparty diversity: "
                    f"{unique_receivers} unique receivers."
                )

            elif unique_receivers >= 5:
                evidence.append(
                    f"Elevated counterparty diversity: "
                    f"{unique_receivers} unique receivers."
                )

            if fraud_ratio >= 0.50:
                evidence.append(
                    f"High historical fraud connection: "
                    f"{fraud_ratio:.1%} of observed transactions "
                    f"were fraudulent."
                )

            elif fraud_ratio >= 0.30:
                evidence.append(
                    f"Meaningful fraud connection: "
                    f"{fraud_ratio:.1%} of observed transactions "
                    f"were fraudulent."
                )

            if avg_receiver_fraud_rate >= 0.30:
                evidence.append(
                    f"Connected receivers have elevated "
                    f"fraud rates (average "
                    f"{avg_receiver_fraud_rate:.1%})."
                )

            if avg_velocity >= 10:
                evidence.append(
                    f"Elevated transaction velocity: "
                    f"{avg_velocity:.1f}."
                )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            if risk_level == "HIGH":

                recommendation = (
                    "ENHANCED_REVIEW"
                )

            elif risk_level == "MEDIUM":

                recommendation = (
                    "MONITOR"
                )

            else:

                recommendation = (
                    "NO_ACTION"
                )

            results.append({

                "user_id": user_id,

                "mule_score": round(
                    mule_score,
                    4
                ),

                "risk_level":
                    risk_level,

                "transactions":
                    transactions,

                "unique_receivers":
                    unique_receivers,

                "total_amount":
                    round(total_amount, 2),

                "fraud_count":
                    int(fraud_count),

                "fraud_ratio":
                    round(fraud_ratio, 4),

                "avg_receiver_fraud_rate":
                    round(
                        avg_receiver_fraud_rate,
                        4
                    ),

                "avg_transaction_velocity":
                    round(
                        avg_velocity,
                        2
                    ),

                "evidence":
                    evidence,

                "recommendation":
                    recommendation
            })

        return pd.DataFrame(results)


# ============================================================
# TEST / DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("       COUNTERRISK — MONEY MULE DETECTOR")
    print("=" * 70)

    print("\nLoading transaction data...")

    df = pd.read_csv(
        "../data/transactions.csv"
    )

    print(
        f"Transactions loaded: {len(df):,}"
    )

    detector = MoneyMuleDetector()

    print("\nRunning Money Mule Detection...")

    results = detector.detect(df)

    print("\nDetection completed.")

    print("\n" + "=" * 70)
    print("TOP 10 MULE RISK ACCOUNTS")
    print("=" * 70)

    top = results.sort_values(
        "mule_score",
        ascending=False
    ).head(10)

    for _, row in top.iterrows():

        print(
            f"\nUser: {row['user_id']}"
        )

        print(
            f"Risk: {row['risk_level']}"
        )

        print(
            f"Mule Score: "
            f"{row['mule_score']:.2%}"
        )

        print(
            f"Transactions: "
            f"{row['transactions']}"
        )

        print(
            f"Unique receivers: "
            f"{row['unique_receivers']}"
        )

        print(
            f"Fraud ratio: "
            f"{row['fraud_ratio']:.2%}"
        )

        print("Evidence:")

        if row["evidence"]:

            for item in row["evidence"]:
                print(f"  - {item}")

        else:

            print(
                "  - No significant mule signals."
            )

        print(
            f"Recommendation: "
            f"{row['recommendation']}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_path = (
        "../data/aml_mule_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"Saved results to:\n"
        f"{output_path}"
    )

    print("=" * 70)