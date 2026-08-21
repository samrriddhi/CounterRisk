import os
import math
import pandas as pd
from collections import defaultdict


# ============================================================
# COUNTERRISK — LAYERING DETECTOR V2
#
# Detects suspicious multi-hop movement of funds.
#
# Design:
#   1. Build sender -> receiver transaction index
#   2. Discover temporal multi-hop paths
#   3. Score each path using multiple AML signals
#   4. Consolidate overlapping paths into networks
#   5. Export explainable results
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "aml_transactions_v2.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "aml_layering_results_v2.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_HOPS = 3
MAX_HOPS = 6

# A layering chain generally happens over a relatively short
# period. We allow a generous window because synthetic data
# should not be rejected too aggressively.
MAX_CHAIN_HOURS = 48

# Maximum number of candidate outgoing transactions examined
# from one node during path expansion.
MAX_BRANCHES = 25

# Amount can change considerably between layers.
# We therefore don't require exact equality.
MIN_AMOUNT_RETENTION = 0.25

# Avoid absurdly tiny transactions relative to the chain.
MIN_AMOUNT = 100


# ============================================================
# HELPERS
# ============================================================

def first_existing(df, candidates, required=True):

    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def clamp(value, low=0.0, high=1.0):

    return max(low, min(high, value))


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("       COUNTERRISK — LAYERING DETECTOR V2")
print("=" * 70)

print()
print("Loading AML transactions...")

df = pd.read_csv(DATA_PATH)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# RESOLVE COLUMN NAMES
# ============================================================

sender_col = first_existing(
    df,
    [
        "sender_id",
        "user_id",
        "source",
        "from_id"
    ]
)

receiver_col = first_existing(
    df,
    [
        "receiver_id",
        "recipient_id",
        "destination",
        "to_id"
    ]
)

amount_col = first_existing(
    df,
    [
        "amount",
        "transaction_amount",
        "value"
    ]
)

timestamp_col = first_existing(
    df,
    [
        "timestamp",
        "datetime",
        "date_time",
        "transaction_time",
        "time"
    ]
)

transaction_id_col = first_existing(
    df,
    [
        "transaction_id",
        "id",
        "txn_id"
    ],
    required=False
)

if transaction_id_col is None:

    df["_transaction_id"] = [
        f"TXN_{i}"
        for i in range(len(df))
    ]

    transaction_id_col = "_transaction_id"


# ============================================================
# NORMALIZE
# ============================================================

df["_sender"] = df[sender_col].astype(str)
df["_receiver"] = df[receiver_col].astype(str)

df["_amount"] = pd.to_numeric(
    df[amount_col],
    errors="coerce"
).fillna(0)

df["_timestamp"] = pd.to_datetime(
    df[timestamp_col],
    errors="coerce"
)

df["_transaction_id"] = df[
    transaction_id_col
].astype(str)


df = df[
    df["_sender"].notna()
    & df["_receiver"].notna()
    & df["_timestamp"].notna()
    & (df["_amount"] > 0)
].copy()


df = df.sort_values("_timestamp").reset_index(drop=True)


# ============================================================
# TRANSACTION INDEX
# ============================================================

print()
print("Building transaction index...")


# receiver -> outgoing transactions
outgoing = defaultdict(list)

# sender -> incoming transactions
incoming = defaultdict(list)


for idx, row in df.iterrows():

    sender = row["_sender"]
    receiver = row["_receiver"]

    outgoing[sender].append(idx)
    incoming[receiver].append(idx)


# Sort every outgoing list chronologically
for node in outgoing:

    outgoing[node].sort(
        key=lambda i: df.at[i, "_timestamp"]
    )


print(f"Graph entities: {len(set(df['_sender']) | set(df['_receiver'])):,}")


# ============================================================
# PATH EXPANSION
# ============================================================

print()
print("Searching for suspicious multi-hop flows...")


def discover_paths(start_idx):

    """
    Discover paths beginning with one transaction.

    Important:
    We do NOT require every hop to be perfect.

    A path is allowed when:

        previous receiver == next sender

    and the next transaction happens after the previous one.

    The final scoring stage decides whether the chain is
    genuinely suspicious.
    """

    start_row = df.loc[start_idx]

    start_sender = start_row["_sender"]
    start_receiver = start_row["_receiver"]

    start_time = start_row["_timestamp"]
    start_amount = start_row["_amount"]

    if start_amount < MIN_AMOUNT:
        return []

    paths = []

    # State:
    #
    # (
    #   current_node,
    #   transaction_indices,
    #   visited_nodes
    # )
    #
    stack = [
        (
            start_receiver,
            [start_idx],
            {start_sender, start_receiver}
        )
    ]

    while stack:

        current_node, chain, visited = stack.pop()

        last_idx = chain[-1]

        last_row = df.loc[last_idx]

        last_time = last_row["_timestamp"]

        # ----------------------------------------------------
        # Save chain once minimum hops reached
        # ----------------------------------------------------

        hops = len(chain)

        if hops >= MIN_HOPS:

            paths.append(
                tuple(chain)
            )

        if hops >= MAX_HOPS:
            continue

        # ----------------------------------------------------
        # Stop after time window
        # ----------------------------------------------------

        elapsed_hours = (
            last_time - start_time
        ).total_seconds() / 3600

        if elapsed_hours > MAX_CHAIN_HOURS:
            continue

        # ----------------------------------------------------
        # Find next transactions
        # ----------------------------------------------------

        candidates = outgoing.get(
            current_node,
            []
        )

        # Only transactions after current transaction
        valid = []

        for next_idx in candidates:

            if next_idx in chain:
                continue

            next_row = df.loc[next_idx]

            next_time = next_row["_timestamp"]

            if next_time <= last_time:
                continue

            elapsed = (
                next_time - start_time
            ).total_seconds() / 3600

            if elapsed > MAX_CHAIN_HOURS:
                continue

            next_receiver = next_row["_receiver"]

            # Prevent loops.
            if next_receiver in visited:
                continue

            valid.append(next_idx)

        # ----------------------------------------------------
        # Branch control
        # ----------------------------------------------------

        if len(valid) > MAX_BRANCHES:

            # Prioritize transactions closest in time.
            valid = sorted(
                valid,
                key=lambda i: df.at[
                    i,
                    "_timestamp"
                ]
            )[:MAX_BRANCHES]

        # ----------------------------------------------------
        # Expand
        # ----------------------------------------------------

        for next_idx in valid:

            next_receiver = df.at[
                next_idx,
                "_receiver"
            ]

            new_visited = set(visited)
            new_visited.add(next_receiver)

            stack.append(
                (
                    next_receiver,
                    chain + [next_idx],
                    new_visited
                )
            )

    return paths


# ============================================================
# DISCOVER
# ============================================================

all_paths = []

# To prevent enormous expansion on normal accounts,
# prioritize transactions involving repeated counterparties.
#
# We still inspect the whole dataset but cap the total paths.

MAX_TOTAL_PATHS = 15000

for idx in range(len(df)):

    if len(all_paths) >= MAX_TOTAL_PATHS:
        break

    paths = discover_paths(idx)

    for path in paths:

        all_paths.append(path)

        if len(all_paths) >= MAX_TOTAL_PATHS:
            break


print(
    f"Potential layering chains before scoring: "
    f"{len(all_paths):,}"
)


# ============================================================
# PATH SCORING
# ============================================================

def score_path(path):

    rows = df.loc[list(path)].copy()

    rows = rows.sort_values(
        "_timestamp"
    )

    amounts = rows["_amount"].tolist()

    timestamps = rows["_timestamp"].tolist()

    senders = rows["_sender"].tolist()

    receivers = rows["_receiver"].tolist()

    hops = len(rows)

    # --------------------------------------------------------
    # 1. HOP SCORE
    # --------------------------------------------------------

    # 3 hops = suspicious
    # 6 hops = extremely suspicious

    hop_score = clamp(
        (hops - 2) / 4
    )


    # --------------------------------------------------------
    # 2. TIME / RAPID MOVEMENT SCORE
    # --------------------------------------------------------

    gaps_hours = []

    for i in range(1, len(timestamps)):

        gap = (
            timestamps[i]
            - timestamps[i - 1]
        ).total_seconds() / 3600

        gaps_hours.append(gap)

    rapid_hops = sum(
        1
        for gap in gaps_hours
        if gap <= 6
    )

    rapid_score = (
        rapid_hops / max(
            1,
            len(gaps_hours)
        )
    )


    # --------------------------------------------------------
    # 3. AMOUNT RETENTION
    # --------------------------------------------------------

    initial_amount = amounts[0]
    final_amount = amounts[-1]

    if initial_amount > 0:

        retention = (
            final_amount /
            initial_amount
        )

    else:

        retention = 0

    retention_score = clamp(
        (retention - MIN_AMOUNT_RETENTION)
        / 0.75
    )


    # --------------------------------------------------------
    # 4. AMOUNT CONTINUITY
    # --------------------------------------------------------

    continuity_scores = []

    for i in range(1, len(amounts)):

        previous = amounts[i - 1]
        current = amounts[i]

        if previous <= 0:
            continue

        ratio = current / previous

        # Perfect preservation around 1.0.
        #
        # We deliberately allow significant variation.
        difference = abs(
            math.log(
                max(ratio, 0.01)
            )
        )

        similarity = math.exp(
            -difference
        )

        continuity_scores.append(
            similarity
        )

    if continuity_scores:

        amount_continuity = sum(
            continuity_scores
        ) / len(continuity_scores)

    else:

        amount_continuity = 0


    # --------------------------------------------------------
    # 5. DECREASING FLOW
    # --------------------------------------------------------

    decreasing_pairs = sum(
        1
        for i in range(1, len(amounts))
        if amounts[i] <= amounts[i - 1] * 1.10
    )

    decreasing_score = (
        decreasing_pairs /
        max(1, len(amounts) - 1)
    )


    # --------------------------------------------------------
    # 6. INTERMEDIATE NODE SCORE
    # --------------------------------------------------------

    intermediate_nodes = receivers[:-1]

    # A node acting as both receiver and sender is important.
    intermediate_count = 0

    for node in intermediate_nodes:

        if node in senders[1:]:

            intermediate_count += 1

    intermediate_score = (
        intermediate_count /
        max(1, hops - 1)
    )


    # --------------------------------------------------------
    # 7. DURATION SCORE
    # --------------------------------------------------------

    duration_hours = (
        timestamps[-1]
        - timestamps[0]
    ).total_seconds() / 3600

    if duration_hours <= 2:

        duration_score = 1.0

    elif duration_hours <= 6:

        duration_score = 0.9

    elif duration_hours <= 12:

        duration_score = 0.8

    elif duration_hours <= 24:

        duration_score = 0.65

    else:

        duration_score = 0.4


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (

        hop_score * 0.18

        + rapid_score * 0.18

        + retention_score * 0.16

        + amount_continuity * 0.16

        + decreasing_score * 0.10

        + intermediate_score * 0.12

        + duration_score * 0.10

    )

    score = clamp(score)


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    if score >= 0.72:

        risk = "HIGH"

    elif score >= 0.50:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # --------------------------------------------------------
    # Only keep meaningful chains
    # --------------------------------------------------------

    # Avoid reporting every random 3-hop connection.
    meaningful = (

        score >= 0.45
        and hops >= MIN_HOPS
        and (
            rapid_score >= 0.40
            or retention >= 0.35
            or amount_continuity >= 0.65
        )
    )

    if not meaningful:
        return None


    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    evidence = []

    evidence.append(
        f"{hops} transaction hops"
    )

    if rapid_hops > 0:

        evidence.append(
            f"{rapid_hops} rapid fund movements"
        )

    evidence.append(
        f"{retention * 100:.1f}% flow value retained"
    )

    if amount_continuity >= 0.70:

        evidence.append(
            "strong amount continuity across layers"
        )

    if decreasing_score >= 0.60:

        evidence.append(
            "progressive movement of funds through layers"
        )

    if duration_hours <= 6:

        evidence.append(
            f"rapid chain completion within {duration_hours:.2f} hours"
        )

    return {

        "score": score,

        "risk": risk,

        "hops": hops,

        "flow": " → ".join(
            [senders[0]]
            + receivers
        ),

        "amounts": [
            round(x, 2)
            for x in amounts
        ],

        "initial_amount":
            round(initial_amount, 2),

        "final_amount":
            round(final_amount, 2),

        "retention":
            round(retention * 100, 2),

        "duration_hours":
            round(duration_hours, 2),

        "rapid_hops":
            rapid_hops,

        "rapid_ratio":
            round(rapid_score, 3),

        "amount_continuity":
            round(amount_continuity, 3),

        "intermediate_score":
            round(intermediate_score, 3),

        "transaction_ids":
            "|".join(
                rows["_transaction_id"].astype(str)
            ),

        "evidence":
            "; ".join(evidence),

        "recommendation":
            "ENHANCED_REVIEW"
            if risk == "HIGH"
            else "STANDARD_REVIEW"
    }


# ============================================================
# SCORE ALL PATHS
# ============================================================

scored = []

for path in all_paths:

    result = score_path(path)

    if result is not None:

        scored.append(result)


print(
    f"Suspicious chains after scoring: "
    f"{len(scored):,}"
)


# ============================================================
# CONSOLIDATION
# ============================================================

print()
print("Consolidating overlapping layering chains...")


# We don't want this:

#
# A → B → C → D → E
# A → B → C → D
# B → C → D → E
# C → D → E
#
# to appear as four completely independent cases.
#
# Instead, we group highly overlapping transaction chains.


def transaction_set(result):

    return set(
        result["transaction_ids"].split("|")
    )


scored.sort(
    key=lambda x: x["score"],
    reverse=True
)


networks = []

used = set()


for i, result in enumerate(scored):

    if i in used:
        continue

    base_transactions = transaction_set(
        result
    )

    group = [result]

    used.add(i)

    for j in range(i + 1, len(scored)):

        if j in used:
            continue

        candidate_transactions = transaction_set(
            scored[j]
        )

        intersection = (
            len(
                base_transactions
                & candidate_transactions
            )
        )

        union = (
            len(
                base_transactions
                | candidate_transactions
            )
        )

        if union == 0:
            continue

        overlap = (
            intersection /
            union
        )

        if overlap >= 0.45:

            group.append(
                scored[j]
            )

            used.add(j)

    # --------------------------------------------------------
    # Pick strongest representative
    # --------------------------------------------------------

    representative = max(
        group,
        key=lambda x: x["score"]
    ).copy()

    representative["related_chains"] = len(
        group
    )

    representative["network_transactions"] = len(
        set().union(
            *[
                transaction_set(x)
                for x in group
            ]
        )
    )

    networks.append(
        representative
    )


# ============================================================
# SORT
# ============================================================

networks.sort(
    key=lambda x: (
        x["score"],
        x["hops"],
        x["retention"]
    ),
    reverse=True
)


# ============================================================
# OUTPUT
# ============================================================

print(
    f"Potential layering networks after consolidation: "
    f"{len(networks):,}"
)


print()
print("=" * 70)
print("TOP LAYERING NETWORKS")
print("=" * 70)


top = networks[:10]


for i, result in enumerate(
    top,
    start=1
):

    print()

    print(
        f"{i}. Risk: {result['risk']}"
    )

    print(
        f"   Layering Score: "
        f"{result['score'] * 100:.2f}%"
    )

    print(
        f"   Hops: {result['hops']}"
    )

    print(
        f"   Flow: {result['flow']}"
    )

    print(
        f"   Amounts: {result['amounts']}"
    )

    print(
        f"   Initial amount: "
        f"₹{result['initial_amount']:,.2f}"
    )

    print(
        f"   Final amount: "
        f"₹{result['final_amount']:,.2f}"
    )

    print(
        f"   Flow retention: "
        f"{result['retention']:.2f}%"
    )

    print(
        f"   Duration: "
        f"{result['duration_hours']:.2f} hours"
    )

    print(
        f"   Rapid hops: "
        f"{result['rapid_hops']}"
    )

    print(
        f"   Amount continuity: "
        f"{result['amount_continuity'] * 100:.2f}%"
    )

    print(
        f"   Related chains: "
        f"{result['related_chains']}"
    )

    print(
        f"   Network transactions: "
        f"{result['network_transactions']}"
    )

    print(
        f"   Evidence: "
        f"{result['evidence']}"
    )

    print(
        f"   Recommendation: "
        f"{result['recommendation']}"
    )


# ============================================================
# SAVE
# ============================================================

output_rows = []

for result in networks:

    output_rows.append({

        "risk":
            result["risk"],

        "layering_score":
            round(
                result["score"],
                4
            ),

        "hops":
            result["hops"],

        "flow":
            result["flow"],

        "initial_amount":
            result["initial_amount"],

        "final_amount":
            result["final_amount"],

        "flow_retention":
            result["retention"],

        "duration_hours":
            result["duration_hours"],

        "rapid_hops":
            result["rapid_hops"],

        "rapid_ratio":
            result["rapid_ratio"],

        "amount_continuity":
            result["amount_continuity"],

        "intermediate_score":
            result["intermediate_score"],

        "related_chains":
            result["related_chains"],

        "network_transactions":
            result["network_transactions"],

        "transaction_ids":
            result["transaction_ids"],

        "evidence":
            result["evidence"],

        "recommendation":
            result["recommendation"]
    })


result_df = pd.DataFrame(
    output_rows
)


result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 70)
print("Saved to:")
print(OUTPUT_PATH)
print("=" * 70)