import os
import pandas as pd


# ============================================
# COUNTERRISK — ELLIPTIC++ DATA INSPECTOR
# ============================================

BASE = "../data/ellipticpp"


def show_file_info(filename, sample_rows=5):
    path = os.path.join(BASE, filename)

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not os.path.exists(path):
        print("❌ FILE NOT FOUND")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    try:
        df = pd.read_csv(path, nrows=sample_rows)

        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")

        print("\nSample:")
        print(df.to_string(index=False))

        print("\nData types:")
        print(df.dtypes)

    except Exception as e:
        print(f"❌ Could not read file: {e}")


def show_class_distribution():
    path = os.path.join(BASE, "txs_classes.csv")

    print("\n" + "=" * 70)
    print("TRANSACTION CLASS DISTRIBUTION")
    print("=" * 70)

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\nClass distribution:")

    # Find likely class column
    class_columns = [
        col for col in df.columns
        if "class" in col.lower()
    ]

    if class_columns:
        class_col = class_columns[0]
        print(f"\nUsing class column: {class_col}")
        print(df[class_col].value_counts(dropna=False))
    else:
        print("Could not automatically identify class column.")


def count_edges(filename):
    path = os.path.join(BASE, filename)

    print("\n" + "=" * 70)
    print(f"EDGE FILE: {filename}")
    print("=" * 70)

    if not os.path.exists(path):
        print("❌ FILE NOT FOUND")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        count = sum(1 for _ in f) - 1

    print(f"Number of edges: {count:,}")

    # Show first few rows
    df = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 edges:")
    print(df.to_string(index=False))


# ============================================
# START INSPECTION
# ============================================

print("=" * 70)
print("          COUNTERRISK — ELLIPTIC++ INSPECTOR")
print("=" * 70)

print(f"\nDataset location:")
print(os.path.abspath(BASE))


# --------------------------------------------
# TRANSACTION DATA
# --------------------------------------------

show_file_info(
    "txs_features.csv",
    sample_rows=5
)

show_class_distribution()

show_file_info(
    "txs_edgelist.csv",
    sample_rows=5
)


# --------------------------------------------
# WALLET / ACTOR DATA
# --------------------------------------------

show_file_info(
    "wallets_features.csv",
    sample_rows=5
)

show_file_info(
    "wallets_classes.csv",
    sample_rows=5
)


# --------------------------------------------
# NETWORK DATA
# --------------------------------------------

count_edges("AddrAddr_edgelist.csv")
count_edges("AddrTx_edgelist.csv")
count_edges("TxAddr_edgelist.csv")


# --------------------------------------------
# FINISHED
# --------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print("""
Next step:
We will design the real CounterRisk Defender
using the actual Elliptic++ columns.

Do NOT train the model yet.
""")