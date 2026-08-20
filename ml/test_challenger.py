from challenger import CounterRiskChallenger


challenger = CounterRiskChallenger()


def run_case(
    name,
    defender_probability,
    prior_connected_transactions,
    prior_illicit_connections,
    prior_illicit_ratio,
    connected_wallets
):

    result = challenger.investigate(

        defender_probability=defender_probability,

        prior_connected_transactions=
            prior_connected_transactions,

        prior_illicit_connections=
            prior_illicit_connections,

        prior_illicit_ratio=
            prior_illicit_ratio,

        connected_wallets=
            connected_wallets
    )


    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Defender probability : "
        f"{defender_probability:.2%}"
    )

    print(
        f"Prior connections    : "
        f"{prior_connected_transactions}"
    )

    print(
        f"Prior illicit        : "
        f"{prior_illicit_connections}"
    )

    print(
        f"Historical ratio    : "
        f"{prior_illicit_ratio:.2%}"
    )

    print()

    print(
        "Challenger outcome   :",
        result["outcome"]
    )

    print(
        "Direction            :",
        result["direction"]
    )

    print(
        "Confidence           :",
        result["confidence"]
    )

    print(
        "Disputes Defender    :",
        result["disputes_defender"]
    )


    if result["challenger_reasons"]:

        print("\nChallenge reasoning:")

        for reason in result[
            "challenger_reasons"
        ]:

            print(
                "  -",
                reason
            )


# ============================================================
# CASE 1
# DEFENDER CORRECTLY HIGH
# ============================================================

run_case(

    "CASE 1 — DEFENDER SUPPORTED",

    defender_probability=0.9999,

    prior_connected_transactions=95,

    prior_illicit_connections=94,

    prior_illicit_ratio=0.9895,

    connected_wallets=2
)


# ============================================================
# CASE 2
# DEFENDER TOO AGGRESSIVE
# ============================================================

run_case(

    "CASE 2 — CHALLENGER DISPUTES HIGH RISK",

    defender_probability=0.90,

    prior_connected_transactions=20,

    prior_illicit_connections=0,

    prior_illicit_ratio=0.0,

    connected_wallets=4
)


# ============================================================
# CASE 3
# DEFENDER TOO OPTIMISTIC
# ============================================================

run_case(

    "CASE 3 — CHALLENGER FINDS MISSED RISK",

    defender_probability=0.30,

    prior_connected_transactions=20,

    prior_illicit_connections=16,

    prior_illicit_ratio=0.80,

    connected_wallets=5
)


# ============================================================
# CASE 4
# BORDERLINE BUT NO CONTRADICTORY EVIDENCE
# ============================================================

run_case(

    "CASE 4 — BORDERLINE / NO CHALLENGE",

    defender_probability=0.4743,

    prior_connected_transactions=0,

    prior_illicit_connections=0,

    prior_illicit_ratio=0.0,

    connected_wallets=3
)