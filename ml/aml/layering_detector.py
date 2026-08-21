import pandas as pd
from collections import defaultdict


DATA_PATH = "../data/aml_transactions.csv"


def build_graph(df):
    graph = defaultdict(list)

    for _, row in df.iterrows():
        graph[row["sender"]].append(
            (
                row["receiver"],
                row["amount"],
                row["time_step"],
                row["transaction_id"]
            )
        )

    return graph


def find_layering_chains(graph, min_hops=3, max_hops=6):
    chains = []

    def dfs(
        current,
        path,
        visited,
        amounts,
        transactions
    ):
        if len(path) - 1 >= min_hops:

            chains.append({
                "chain": path.copy(),
                "hops": len(path) - 1,
                "amounts": amounts.copy(),
                "transactions": transactions.copy()
            })

        if len(path) - 1 >= max_hops:
            return

        for (
            receiver,
            amount,
            time_step,
            transaction_id
        ) in graph.get(current, []):

            # Prevent cycles here.
            # Round-tripping gets handled separately.
            if receiver in visited:
                continue

            visited.add(receiver)
            path.append(receiver)

            amounts.append(amount)
            transactions.append(transaction_id)

            dfs(
                receiver,
                path,
                visited,
                amounts,
                transactions
            )

            transactions.pop()
            amounts.pop()
            path.pop()

            visited.remove(receiver)

    for sender in graph:
        dfs(
            sender,
            [sender],
            {sender},
            [],
            []
        )

    return chains


def calculate_layering_score(chain):
    hops = chain["hops"]
    amounts = chain["amounts"]

    # Longer chains are more suspicious.
    hop_score = min(
        hops / 5,
        1.0
    )

    # Check whether substantial value survives through the chain.
    if len(amounts) > 1:

        retention = (
            amounts[-1] /
            max(amounts[0], 1)
        )

    else:
        retention = 1.0

    retention = max(
        0,
        min(retention, 1)
    )

    score = (
        0.65 * hop_score +
        0.35 * retention
    )

    return round(
        score * 100,
        2
    )


def main():

    print("=" * 70)
    print("       COUNTERRISK — LAYERING DETECTOR")
    print("=" * 70)

    print("\nLoading AML transaction graph...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Transactions loaded: {len(df):,}"
    )

    print("\nBuilding transaction graph...")

    graph = build_graph(df)

    print(
        f"Graph entities: {len(graph):,}"
    )

    print("\nSearching for multi-hop transaction chains...")

    chains = find_layering_chains(
        graph,
        min_hops=3,
        max_hops=6
    )

    print(
        f"Potential chains found: {len(chains):,}"
    )

    if not chains:

        print("\nNo layering chains detected.")
        return

    results = []

    for chain in chains:

        score = calculate_layering_score(
            chain
        )

        results.append({

            "risk_score":
                score,

            "risk":
                (
                    "HIGH"
                    if score >= 70
                    else
                    "MEDIUM"
                    if score >= 40
                    else
                    "LOW"
                ),

            "hops":
                chain["hops"],

            "chain":
                " → ".join(
                    chain["chain"]
                ),

            "amounts":
                chain["amounts"],

            "transactions":
                chain["transactions"]
        })

    results = sorted(
        results,
        key=lambda x: x["risk_score"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP LAYERING CHAINS")
    print("=" * 70)

    for i, result in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"\n{i}. Risk: {result['risk']}"
        )

        print(
            f"   Layering Score: "
            f"{result['risk_score']:.2f}%"
        )

        print(
            f"   Hops: {result['hops']}"
        )

        print(
            f"   Flow: {result['chain']}"
        )

        print(
            f"   Amounts: "
            f"{[round(x, 2) for x in result['amounts']]}"
        )

        print(
            f"   Recommendation: "
            f"{'ENHANCED_REVIEW' if result['risk_score'] >= 70 else 'MONITOR'}"
        )

    output = pd.DataFrame(
        results
    )

    output.to_csv(
        "../data/aml_layering_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print(
        "Saved results to:"
    )
    print(
        "../../data/aml_layering_results.csv"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()