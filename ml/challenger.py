# ============================================================
# COUNTERRISK CHALLENGER
#
# The Challenger independently reviews the Defender's
# assessment using historical information available before
# the current transaction.
#
# It can challenge the Defender in either direction:
#
# 1. Defender may be OVERREACTING
#    High risk + clean historical network
#
# 2. Defender may be UNDERESTIMATING
#    Low/moderate risk + strong historical fraud network
# ============================================================


class CounterRiskChallenger:

    def __init__(self):

        self.name = (
            "CounterRisk Evidence Challenger"
        )


    # ========================================================
    # INVESTIGATE
    # ========================================================

    def investigate(
        self,
        defender_probability,
        prior_connected_transactions,
        prior_illicit_connections,
        prior_illicit_ratio,
        connected_wallets
    ):

        supporting_evidence = []

        counter_evidence = []

        challenger_reasons = []


        # ====================================================
        # 1. HISTORICAL FRAUD EVIDENCE
        # ====================================================

        if prior_illicit_connections > 0:

            supporting_evidence.append(
                f"{prior_illicit_connections} "
                f"prior connected transaction(s) "
                f"were historically illicit."
            )


        if prior_illicit_ratio >= 0.50:

            supporting_evidence.append(
                f"Historical illicit ratio is "
                f"{prior_illicit_ratio:.1%}, "
                f"indicating strong prior fraud exposure."
            )

        elif prior_illicit_ratio >= 0.20:

            supporting_evidence.append(
                f"Historical illicit ratio is "
                f"{prior_illicit_ratio:.1%}, "
                f"indicating meaningful prior fraud exposure."
            )


        # ====================================================
        # 2. LEGITIMACY / COUNTER-EVIDENCE
        # ====================================================

        if (
            prior_connected_transactions >= 2
            and
            prior_illicit_connections == 0
        ):

            counter_evidence.append(
                "Historical network activity exists "
                "without confirmed prior illicit activity."
            )


        if (
            prior_connected_transactions > 0
            and
            prior_illicit_ratio == 0
        ):

            counter_evidence.append(
                "No confirmed illicit activity exists "
                "within the prior connected history."
            )


        # ====================================================
        # 3. CHALLENGE: DEFENDER MAY BE OVERREACTING
        # ====================================================

        if (
            defender_probability >= 0.75
            and
            prior_connected_transactions >= 5
            and
            prior_illicit_ratio == 0
        ):

            challenger_reasons.append(
                "Defender risk is high, but the "
                "historical network contains no "
                "confirmed illicit activity."
            )


        # ====================================================
        # 4. CHALLENGE: DEFENDER MAY BE UNDERESTIMATING
        # ====================================================

        if (
            defender_probability < 0.50
            and
            prior_illicit_ratio >= 0.50
        ):

            challenger_reasons.append(
                "Defender risk is below 50%, but "
                "historical network evidence shows "
                "a strong illicit concentration."
            )


        # ====================================================
        # 5. MODERATE CONFLICT
        # ====================================================

        if (
            defender_probability < 0.65
            and
            prior_illicit_ratio >= 0.20
        ):

            challenger_reasons.append(
                "Network evidence materially conflicts "
                "with the Defender's moderate risk estimate."
            )


        # ====================================================
        # 6. DETERMINE OUTCOME
        # ====================================================

        if challenger_reasons:

            outcome = "DISPUTE"

        else:

            outcome = "SUPPORT"


        # ====================================================
        # 7. DETERMINE DIRECTION
        # ====================================================

        direction = "NONE"


        # Challenger thinks Defender is too aggressive.

        if (
            defender_probability >= 0.75
            and
            prior_illicit_ratio == 0
            and
            prior_connected_transactions >= 5
        ):

            direction = "DEFENDER_TOO_AGGRESSIVE"


        # Challenger thinks Defender is too optimistic.

        elif (
            defender_probability < 0.50
            and
            prior_illicit_ratio >= 0.50
        ):

            direction = "DEFENDER_UNDERESTIMATES_RISK"


        # ====================================================
        # 8. CONFIDENCE
        # ====================================================

        if prior_illicit_ratio >= 0.50:

            challenger_confidence = "HIGH"

        elif (
            prior_illicit_ratio >= 0.20
            or
            prior_connected_transactions >= 5
        ):

            challenger_confidence = "MEDIUM"

        else:

            challenger_confidence = "LOW"


        # ====================================================
        # 9. RESULT
        # ====================================================

        return {

            "outcome":
                outcome,

            "direction":
                direction,

            "confidence":
                challenger_confidence,

            "supporting_evidence":
                supporting_evidence,

            "counter_evidence":
                counter_evidence,

            "challenger_reasons":
                challenger_reasons,

            "disputes_defender":
                outcome == "DISPUTE"
        }