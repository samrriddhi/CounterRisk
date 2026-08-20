import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st


# ============================================================
# COUNTERRISK
# Risk Intelligence & Payment Decision Platform
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "ml")
ELLIPTIC_DIR = os.path.join(DATA_DIR, "ellipticpp")

MODEL_FILE = os.path.join(
    ML_DIR,
    "counterrisk_v3.pkl"
)

TRANSACTION_FILE = os.path.join(
    ELLIPTIC_DIR,
    "txs_features.csv"
)

NETWORK_FILE = os.path.join(
    DATA_DIR,
    "counterrisk_historical_fraud_network.csv"
)

SNAPSHOT_FILE = os.path.join(
    DATA_DIR,
    "counterrisk_network_snapshots.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CounterRisk | Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0a0f18;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #0d1420;
        border-right: 1px solid #202b3a;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.7rem;
    }

    h1, h2, h3, h4 {
        color: #f3f4f6 !important;
        letter-spacing: -0.01em;
    }

    p, label, span, div {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                     BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .eyebrow {
        color: #8090a5;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .page-title {
        color: #f8fafc;
        font-size: 2.45rem;
        line-height: 1.05;
        font-weight: 750;
        margin: 0;
    }

    .page-subtitle {
        color: #8fa0b5;
        font-size: 0.96rem;
        margin-top: 0.5rem;
    }

    .muted {
        color: #94a3b8;
    }

    .metric-wrap {
        background: #101824;
        border: 1px solid #202d3d;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        min-height: 118px;
    }

    .metric-label {
        color: #8190a5;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 0.4rem;
        line-height: 1;
    }

    .metric-caption {
        color: #7f8ea3;
        font-size: 0.78rem;
        margin-top: 0.55rem;
    }

    .status {
        display: inline-flex;
        align-items: center;
        padding: 0.32rem 0.62rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .status-block {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.28);
    }

    .status-review {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }

    .status-allow {
        background: rgba(34, 197, 94, 0.10);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.24);
    }

    .evidence-label {
        color: #7f8ea3;
        font-size: 0.74rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .evidence-value {
        color: #edf2f7;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.25rem;
        line-height: 1.45;
    }

    .decision-box {
        background: #0d1622;
        border: 1px solid #29384b;
        border-radius: 10px;
        padding: 1.2rem 1.3rem;
    }

    .decision-action {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }

    .decision-reason {
        color: #a9b5c4;
        line-height: 1.6;
        margin-top: 0.6rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_resource
def _load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def _load_transactions():
    df = pd.read_csv(TRANSACTION_FILE)
    df["txId"] = df["txId"].apply(
        lambda x: str(int(float(x)))
    )
    return df


@st.cache_data
def _load_network():
    df = pd.read_csv(NETWORK_FILE)
    df["txId"] = df["txId"].apply(
        lambda x: str(int(float(x)))
    )
    return df


@st.cache_data
def _load_snapshots():
    with open(SNAPSHOT_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


bundle = _load_model()
model = bundle["model"]
feature_columns = bundle["features"]
train_medians = bundle["train_medians"]

transactions = _load_transactions()
network = _load_network()
snapshots = _load_snapshots()

data = transactions.merge(
    network,
    on="txId",
    how="left",
)


# ============================================================
# DECISION ENGINE
# ============================================================

def make_decision(
    risk,
    network_ratio,
    high,
    medium,
    network_limit,
):
    if risk >= high:
        if network_ratio >= network_limit:
            return (
                "BLOCK",
                "High model risk is independently supported by historical illicit network evidence.",
            )
        return (
            "STEP-UP",
            "High model risk is present, but historical network evidence does not justify automatic blocking.",
        )

    if risk >= medium:
        if network_ratio >= network_limit:
            return (
                "BLOCK",
                "Borderline model risk is materially strengthened by historical illicit network evidence.",
            )
        return (
            "STEP-UP",
            "Model confidence is within the review range and requires additional verification.",
        )

    if network_ratio >= 0.50:
        return (
            "STEP-UP",
            "Low transaction-level model risk conflicts with strong historical network evidence.",
        )

    return (
        "ALLOW",
        "Low model risk and no strong historical fraud-network signal.",
    )


# ============================================================
# MODEL PREDICTION HELPERS
# ============================================================

def score_transaction(tx_id):
    matches = data[data["txId"] == str(tx_id).strip()]

    if len(matches) == 0:
        return None

    row = matches.iloc[0]
    X = pd.DataFrame([row[feature_columns]])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(train_medians)

    risk = float(model.predict_proba(X)[0][1])

    return {
        "row": row,
        "risk": risk,
        "network_ratio": float(row.get("prior_illicit_ratio", 0) or 0),
        "prior_connections": int(row.get("prior_connected_transactions", 0) or 0),
        "prior_illicit": int(row.get("prior_illicit_connections", 0) or 0),
        "connected_wallets": int(row.get("connected_wallets", 0) or 0),
    }


@st.cache_data
def _find_low_risk_transaction():
    candidates = data[data["Time step"] >= 35].head(2000).copy()

    if len(candidates) == 0:
        return "72637933"

    X = candidates[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(train_medians)
    risks = model.predict_proba(X)[:, 1]
    candidates["_risk"] = risks

    clean = candidates[
        candidates["prior_illicit_ratio"].fillna(0) == 0
    ]

    if len(clean) > 0:
        return str(clean.sort_values("_risk").iloc[0]["txId"])

    return str(candidates.sort_values("_risk").iloc[0]["txId"])


# ============================================================
# SIDEBAR WORKSPACE
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="eyebrow">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    workspace = st.radio(
        "Workspace",
        [
            "Customer Payment",
            "Risk Operations",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    demo_scenario = "Normal payment"

    if workspace == "Customer Payment":
        st.markdown(
            '<div class="eyebrow">DEMO SCENARIO</div>',
            unsafe_allow_html=True,
        )

        demo_scenario = st.selectbox(
            "Select scenario",
            [
                "Normal payment",
                "Borderline payment",
                "High-risk payment",
            ],
            label_visibility="collapsed",
        )

        st.caption(
            "Use these deterministic scenarios during the demo to "
            "show all three CounterRisk outcomes."
        )

        st.markdown("---")


# ============================================================
# CUSTOMER PAYMENT EXPERIENCE
# ============================================================

if workspace == "Customer Payment":

    st.markdown(
        '<div class="eyebrow">PAYMENT EXPERIENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Make a Payment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Secure payment authorization powered by CounterRisk'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.15])

    scenario_defaults = {
        "Normal payment": {
            "amount": 2500.0,
            "recipient": "ABC Store",
            "payment_method": "UPI",
            "reference": None,
        },
        "Borderline payment": {
            "amount": 7500.0,
            "recipient": "Unknown Merchant",
            "payment_method": "UPI",
            "reference": "72637933",
        },
        "High-risk payment": {
            "amount": 50000.0,
            "recipient": "Flagged Merchant",
            "payment_method": "Bank Transfer",
            "reference": "10000476",
        },
    }

    defaults = scenario_defaults[demo_scenario]

    with left:
        with st.container(border=True):
            st.markdown("### Payment details")

            amount = st.number_input(
                "Amount",
                min_value=1.0,
                max_value=500000.0,
                value=defaults["amount"],
                step=100.0,
                key=f"amount_{demo_scenario}",
            )

            recipient = st.text_input(
                "Recipient",
                value=defaults["recipient"],
                key=f"recipient_{demo_scenario}",
            )

            payment_method = st.selectbox(
                "Payment method",
                [
                    "UPI",
                    "Debit Card",
                    "Bank Transfer",
                ],
                index=[
                    "UPI",
                    "Debit Card",
                    "Bank Transfer",
                ].index(defaults["payment_method"]),
                key=f"method_{demo_scenario}",
            )

            pay_now = st.button(
                "Authorize Payment",
                type="secondary",
                use_container_width=True,
            )

            st.caption(
                "CounterRisk will assess the payment before authorization."
            )

    if pay_now:
        # Deterministic demo routing. This is intentionally isolated
        # from the production decision logic so the interviewer can
        # reliably demonstrate every payment outcome.
        representative_id = defaults["reference"]

        if representative_id is None:
            representative_id = _find_low_risk_transaction()

        result = score_transaction(representative_id)

        if result is not None:
            decision, reason = make_decision(
                result["risk"],
                result["network_ratio"],
                0.85,
                0.45,
                0.20,
            )

            st.session_state["payment_result"] = {
                "reference": representative_id,
                "amount": amount,
                "recipient": recipient,
                "payment_method": payment_method,
                "risk": result["risk"],
                "network_ratio": result["network_ratio"],
                "prior_connections": result["prior_connections"],
                "prior_illicit": result["prior_illicit"],
                "connected_wallets": result["connected_wallets"],
                "decision": decision,
                "reason": reason,
            }

    with right:
        with st.container(border=True):
            st.markdown("### Authorization status")

            payment = st.session_state.get("payment_result")

            if payment is None:
                st.markdown("### Awaiting authorization")
                st.caption(
                    "Submit a payment to run the CounterRisk assessment."
                )
            else:
                decision = payment["decision"]

                if decision == "ALLOW":
                    st.success("PAYMENT AUTHORIZED")
                    st.write(
                        f"Payment of **₹{payment['amount']:,.2f}** to "
                        f"**{payment['recipient']}** has been authorized."
                    )
                    st.caption(
                        f"Payment reference: {payment['reference']}"
                    )

                elif decision == "STEP-UP":
                    st.warning("VERIFICATION REQUIRED")
                    st.write(
                        "For your protection, this payment requires an additional verification step."
                    )
                    st.caption(
                        f"Payment reference: {payment['reference']}"
                    )

                    if st.button(
                        "Verify and Continue",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.success(
                            "Verification completed. Payment approved for this demonstration."
                        )

                else:
                    st.error("PAYMENT NOT AUTHORIZED")
                    st.write(
                        "The payment was blocked because CounterRisk identified a high-risk profile."
                    )
                    st.caption(
                        "Contact your financial institution if this payment was expected."
                    )

                with st.expander("View risk-operations evidence"):
                    st.write(
                        f"Representative transaction: {payment['reference']}"
                    )
                    st.write(
                        f"Prior connected transactions: {payment['prior_connections']}"
                    )
                    st.write(
                        f"Prior illicit connections: {payment['prior_illicit']}"
                    )
                    st.write(
                        f"Connected wallets: {payment['connected_wallets']}"
                    )
                    st.write(
                        payment["reason"]
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Demonstration environment"):
        st.caption(
            "Payment inputs are routed to representative transactions from the evaluated "
            "dataset. No real payment is initiated, charged, or transferred."
        )

    with st.container(border=True):
        st.markdown("### How CounterRisk protects the payment")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Assess**")
            st.caption(
                "Transaction behavior is evaluated by the Defender model."
            )

        with c2:
            st.markdown("**Challenge**")
            st.caption(
                "Historical network intelligence provides independent context."
            )

        with c3:
            st.markdown("**Decide**")
            st.caption(
                "The policy selects authorization, verification, or blocking."
            )

    st.stop()


# ============================================================
# RISK OPERATIONS CONSOLE
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="eyebrow">INVESTIGATION</div>',
        unsafe_allow_html=True,
    )

    transaction_id = st.text_input(
        "Transaction ID",
        value="72637933",
    ).strip()

    st.markdown("---")

    st.markdown(
        '<div class="eyebrow">DECISION POLICY</div>',
        unsafe_allow_html=True,
    )

    high_threshold = st.slider(
        "High-risk threshold",
        0.50,
        0.99,
        0.85,
        0.01,
    )

    medium_threshold = st.slider(
        "Review threshold",
        0.20,
        0.80,
        0.45,
        0.01,
    )

    network_threshold = st.slider(
        "Network evidence threshold",
        0.00,
        1.00,
        0.20,
        0.05,
    )

    st.markdown("---")
    st.caption(
        "Historical network evidence uses information from earlier time steps only."
    )


result = score_transaction(transaction_id)

if result is None:
    st.error(
        f"Transaction {transaction_id} was not found."
    )
    st.stop()

row = result["row"]
defender_probability = result["risk"]
prior_connections = result["prior_connections"]
prior_illicit = result["prior_illicit"]
connected_wallets = result["connected_wallets"]
prior_illicit_ratio = result["network_ratio"]

time_step = int(
    row.get(
        "Time step",
        row.get("time_step", 0),
    )
)

decision, decision_reason = make_decision(
    defender_probability,
    prior_illicit_ratio,
    high_threshold,
    medium_threshold,
    network_threshold,
)

status_class = (
    "status-block" if decision == "BLOCK"
    else "status-review" if decision == "STEP-UP"
    else "status-allow"
)


# ============================================================
# RISK OPS HEADER
# ============================================================

header_left, header_right = st.columns([7, 2])

with header_left:
    st.markdown(
        '<div class="eyebrow">RISK INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-title">CounterRisk</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">'
        'Evidence-driven transaction risk assessment and decision intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; padding-top:0.5rem;">
            <span class="status status-allow">SYSTEM ONLINE</span>
            <div class="muted" style="margin-top:0.45rem; font-size:0.74rem;">
                MODEL: COUNTERRISK v3
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TRANSACTION HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">TRANSACTION INVESTIGATION</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([7, 2])

with left:
    st.markdown(f"### {transaction_id}")
    st.caption(
        f"Time step {time_step} · Historical network context enabled"
    )

with right:
    st.markdown(
        f'<div style="text-align:right;"><span class="status {status_class}">{decision}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.container(border=True):
        st.markdown("**MODEL RISK**")
        st.markdown(f"### {defender_probability:.2%}")
        st.caption("Transaction-level illicit probability")

with c2:
    with st.container(border=True):
        st.markdown("**HISTORICAL ILLICIT RATIO**")
        st.markdown(f"### {prior_illicit_ratio:.2%}")
        st.caption("Confirmed illicit activity in prior network history")

with c3:
    with st.container(border=True):
        st.markdown("**PRIOR NETWORK ACTIVITY**")
        st.markdown(f"### {prior_connections:,}")
        st.caption("Connected transactions before current event")

with c4:
    with st.container(border=True):
        st.markdown("**DECISION**")
        st.markdown(f"### {decision}")
        st.caption("Current policy outcome")


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

with st.container(border=True):
    st.markdown("### Evidence Summary")
    st.caption("Independent signals considered in the current decision")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("**MODEL ASSESSMENT**")
        st.write(
            f"{defender_probability:.2%} transaction-level illicit probability"
        )

    with e2:
        st.markdown("**NETWORK ASSESSMENT**")
        if prior_connections > 0:
            st.write(
                f"{prior_illicit:,} of {prior_connections:,} prior connected "
                "transactions were historically illicit."
            )
        else:
            st.write("No prior connected transaction history identified.")

    with e3:
        st.markdown("**DECISION**")
        st.write(decision)


# ============================================================
# TABS
# ============================================================

tab_assessment, tab_network, tab_policy = st.tabs(
    [
        "Assessment",
        "Network Intelligence",
        "Policy Simulation",
    ]
)


# ============================================================
# ASSESSMENT
# ============================================================

with tab_assessment:
    a1, a2 = st.columns(2)

    with a1:
        with st.container(border=True):
            st.markdown("### Model Assessment")
            st.caption("Defender output from CounterRisk v3")
            st.progress(min(defender_probability, 1.0))
            st.write(f"Risk probability: **{defender_probability:.2%}**")

            if defender_probability >= high_threshold:
                st.warning("High model risk. Transaction-level evidence supports intervention.")
            elif defender_probability >= medium_threshold:
                st.warning("Borderline model risk. Additional evidence should inform the final action.")
            else:
                st.success("Low transaction-level model risk.")

    with a2:
        with st.container(border=True):
            st.markdown("### Historical Network Assessment")
            st.caption("Evidence available before the current transaction")

            st.write(f"Prior connected transactions: **{prior_connections:,}**")
            st.write(f"Prior illicit connections: **{prior_illicit:,}**")
            st.write(f"Connected wallets: **{connected_wallets:,}**")
            st.write(f"Historical illicit ratio: **{prior_illicit_ratio:.2%}**")

            if prior_illicit_ratio >= 0.50:
                st.error("Strong historical fraud-network signal.")
            elif prior_illicit_ratio >= 0.20:
                st.warning("Meaningful historical fraud-network signal.")
            elif prior_connections > 0:
                st.info("Historical network activity exists without a strong illicit concentration.")
            else:
                st.success("No prior connected transaction history is available.")

    with st.container(border=True):
        st.markdown("### Decision Rationale")
        st.markdown(
            f"**{decision}** — {decision_reason}"
        )


# ============================================================
# NETWORK INTELLIGENCE
# ============================================================

with tab_network:
    snapshot = snapshots.get(transaction_id, {})
    wallets = snapshot.get("wallets", [])
    previous_transactions = snapshot.get("previous_transactions", [])

    st.subheader("Network Intelligence")
    st.caption(
        "Historical transaction and wallet relationships available before the current event."
    )

    n1, n2, n3 = st.columns(3)

    with n1:
        with st.container(border=True):
            st.markdown("**Wallet Associations**")
            st.markdown(f"### {len(wallets):,}")
            st.caption("Wallets associated with the current transaction")

    with n2:
        with st.container(border=True):
            st.markdown("**Historical Transactions**")
            st.markdown(f"### {prior_connections:,}")
            st.caption("Prior connected transactions")

    with n3:
        with st.container(border=True):
            st.markdown("**Historical Illicit Ratio**")
            st.markdown(f"### {prior_illicit_ratio:.2%}")
            st.caption("Confirmed illicit activity in prior network history")

    with st.container(border=True):
        st.markdown("### Transaction Relationship Map")
        st.caption("Compact historical investigation view")

        st.markdown(
            f"**Current transaction:** `{transaction_id}`"
        )

        if wallets:
            st.markdown("**Associated wallets**")
            wallet_cols = st.columns(min(len(wallets[:6]), 6))
            for idx, wallet in enumerate(wallets[:6]):
                short_wallet = wallet[:16] + "..." if len(wallet) > 16 else wallet
                with wallet_cols[idx]:
                    st.info(short_wallet)
        else:
            st.caption("No wallet associations available.")

        st.divider()
        st.markdown("**Prior transactions**")

        if previous_transactions:
            hist_cols = st.columns(4)
            for idx, tx in enumerate(previous_transactions[:20]):
                with hist_cols[idx % 4]:
                    st.write(f"`{tx}`")
        else:
            st.caption("No prior transactions available for this snapshot.")

    if prior_illicit_ratio >= 0.50:
        st.error(
            f"Strong historical network evidence: {prior_illicit:,} of "
            f"{prior_connections:,} prior connected transactions were historically illicit."
        )
    elif prior_illicit_ratio >= 0.20:
        st.warning(
            f"Meaningful historical network evidence: {prior_illicit:,} of "
            f"{prior_connections:,} prior connected transactions were historically illicit."
        )
    elif prior_connections > 0:
        st.info(
            f"Historical network activity exists, but only {prior_illicit:,} of "
            f"{prior_connections:,} prior connected transactions were historically illicit."
        )
    else:
        st.success("No prior connected transaction history was available.")

    with st.container(border=True):
        st.markdown("### Historical Transaction Context")
        if previous_transactions:
            history_df = pd.DataFrame({"Transaction ID": previous_transactions})
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No prior transactions are available for this network snapshot.")


# ============================================================
# POLICY SIMULATION
# ============================================================

with tab_policy:
    with st.container(border=True):
        st.markdown("### Policy Simulation")
        st.caption("Evaluate how intervention thresholds affect the current transaction.")

        sim_high = st.slider(
            "High-risk threshold",
            0.50,
            0.99,
            high_threshold,
            0.01,
            key="ops_policy_high",
        )

        sim_medium = st.slider(
            "Review threshold",
            0.20,
            0.80,
            medium_threshold,
            0.01,
            key="ops_policy_medium",
        )

        sim_network = st.slider(
            "Network evidence threshold",
            0.00,
            1.00,
            network_threshold,
            0.05,
            key="ops_policy_network",
        )

        simulated_action, simulated_reason = make_decision(
            defender_probability,
            prior_illicit_ratio,
            sim_high,
            sim_medium,
            sim_network,
        )

        if simulated_action == "BLOCK":
            st.error("SIMULATED OUTCOME · BLOCK")
        elif simulated_action == "STEP-UP":
            st.warning("SIMULATED OUTCOME · STEP-UP")
        else:
            st.success("SIMULATED OUTCOME · ALLOW")

        st.caption(simulated_reason)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander("Technical Details"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Dataset**")
        st.write("Elliptic++")
        st.write("**Model**")
        st.write("Random Forest · CounterRisk v3")
        st.write("**Transaction features**")
        st.write("183")

    with col_b:
        st.write("**Training period**")
        st.write("Time steps 1–34")
        st.write("**Evaluation period**")
        st.write("Time steps 35–49")
        st.write("**Network intelligence**")
        st.write("Leakage-safe historical wallet activity")