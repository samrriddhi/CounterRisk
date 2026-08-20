import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st


# ============================================================
# COUNTERRISK
# Enterprise Risk Intelligence Console
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

ML_DIR = os.path.join(
    BASE_DIR,
    "ml"
)

ELLIPTIC_DIR = os.path.join(
    DATA_DIR,
    "ellipticpp"
)

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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CounterRisk | Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ENTERPRISE STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ----------------------------------------------------- */
    /* Global */
    /* ----------------------------------------------------- */

    .stApp {
        background: #0a0f18;
        color: #e5e7eb;
    }

    .main {
        background: #0a0f18;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    /* ----------------------------------------------------- */
    /* Sidebar */
    /* ----------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #0d1420;
        border-right: 1px solid #202b3a;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ----------------------------------------------------- */
    /* Typography */
    /* ----------------------------------------------------- */

    h1, h2, h3, h4 {
        color: #f3f4f6 !important;
        letter-spacing: -0.01em;
    }

    p, label, span, div {
        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .muted {
        color: #94a3b8;
    }

    .eyebrow {
        color: #7f8ea3;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .page-title {
        color: #f8fafc;
        font-size: 2.4rem;
        line-height: 1.05;
        font-weight: 750;
        margin: 0;
    }

    .page-subtitle {
        color: #8fa0b5;
        font-size: 0.96rem;
        margin-top: 0.5rem;
    }

    /* ----------------------------------------------------- */
    /* Panels */
    /* ----------------------------------------------------- */

    .panel {
        background: #101824;
        border: 1px solid #202d3d;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: #e5e7eb;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }

    .panel-subtitle {
        color: #7f8ea3;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    /* ----------------------------------------------------- */
    /* Metric cards */
    /* ----------------------------------------------------- */

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

    /* ----------------------------------------------------- */
    /* Status badges */
    /* ----------------------------------------------------- */

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

    /* ----------------------------------------------------- */
    /* Evidence */
    /* ----------------------------------------------------- */

    .evidence-row {
        padding: 0.8rem 0;
        border-bottom: 1px solid #1f2a38;
    }

    .evidence-row:last-child {
        border-bottom: none;
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
    }

    /* ----------------------------------------------------- */
    /* Decision box */
    /* ----------------------------------------------------- */

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

    /* ----------------------------------------------------- */
    /* Network visual */
    /* ----------------------------------------------------- */

    .network {
        background: #0c131d;
        border: 1px solid #202d3d;
        border-radius: 10px;
        padding: 1.3rem;
        text-align: center;
    }

    .network-node {
        display: inline-block;
        padding: 0.55rem 0.8rem;
        border: 1px solid #334155;
        border-radius: 7px;
        background: #111b28;
        margin: 0.25rem;
        color: #dbe4ef;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .network-node-risk {
        border-color: #9f3a3a;
        background: rgba(127, 29, 29, 0.20);
        color: #fecaca;
    }

    .network-line {
        color: #53657b;
        font-size: 0.9rem;
        margin: 0.15rem 0;
    }

    /* ----------------------------------------------------- */
    /* Streamlit controls */
    /* ----------------------------------------------------- */

    div[data-baseweb="input"] {
        background: #0b111a;
    }

    div[data-baseweb="select"] > div {
        background: #0b111a;
        border-color: #2a3849;
    }

    .stSlider {
        margin-bottom: 0.8rem;
    }

    /* ----------------------------------------------------- */
    /* Tabs */
    /* ----------------------------------------------------- */

    button[data-baseweb="tab"] {
        color: #8fa0b5;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc;
    }

    /* ----------------------------------------------------- */
    /* Hide Streamlit branding */
    /* ----------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_FILE
    )


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

@st.cache_data
def load_transactions():

    df = pd.read_csv(
        TRANSACTION_FILE
    )

    df["txId"] = (
        df["txId"]
        .apply(
            lambda x:
            str(int(float(x)))
        )
    )

    return df


# ============================================================
# LOAD NETWORK FEATURES
# ============================================================

@st.cache_data
def load_network():

    df = pd.read_csv(
        NETWORK_FILE
    )

    df["txId"] = (
        df["txId"]
        .apply(
            lambda x:
            str(int(float(x)))
        )
    )

    return df


# ============================================================
# LOAD
# ============================================================

bundle = load_model()

model = bundle["model"]

feature_columns = bundle["features"]

train_medians = bundle["train_medians"]

transactions = load_transactions()

network = load_network()


# ============================================================
# MERGE
# ============================================================

data = transactions.merge(
    network,
    on="txId",
    how="left"
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [7, 2]
)

with header_left:

    st.markdown(
        '<div class="eyebrow">'
        'RISK INTELLIGENCE PLATFORM'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">'
        'CounterRisk'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Evidence-driven transaction risk assessment and decision intelligence'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        """
        <div style="text-align:right; padding-top:0.5rem;">
            <span class="status status-allow">
                SYSTEM ONLINE
            </span>
            <div class="muted"
                 style="margin-top:0.45rem; font-size:0.74rem;">
                MODEL: COUNTERRISK v3
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="eyebrow">'
        'INVESTIGATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Transaction Review"
    )

    transaction_id = st.text_input(
        "Transaction ID",
        value="72637933"
    ).strip()


    st.markdown("---")

    st.markdown(
        '<div class="eyebrow">'
        'DECISION POLICY'
        '</div>',
        unsafe_allow_html=True
    )


    high_threshold = st.slider(
        "High-risk threshold",
        0.50,
        0.99,
        0.85,
        0.01
    )


    medium_threshold = st.slider(
        "Review threshold",
        0.20,
        0.80,
        0.45,
        0.01
    )


    network_threshold = st.slider(
        "Network evidence threshold",
        0.00,
        1.00,
        0.20,
        0.05
    )


    st.markdown("---")

    st.markdown(
        '<div class="muted" style="font-size:0.78rem; line-height:1.6;">'
        'Historical network evidence uses information from '
        'earlier time steps only.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FIND TRANSACTION
# ============================================================

matches = data[
    data["txId"] == transaction_id
]


if len(matches) == 0:

    st.error(
        f"Transaction {transaction_id} was not found."
    )

    st.stop()


row = matches.iloc[0]


# ============================================================
# MODEL INPUT
# ============================================================

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


# ============================================================
# DEFENDER
# ============================================================

defender_probability = float(
    model.predict_proba(
        X
    )[0][1]
)


# ============================================================
# NETWORK EVIDENCE
# ============================================================

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


time_step = int(
    row.get(
        "Time step",
        row.get(
            "time_step",
            0
        )
    )
)


# ============================================================
# DECISION
# ============================================================

def make_decision(
    risk,
    network_ratio,
    high,
    medium,
    network_limit
):

    if risk >= high:

        if network_ratio >= network_limit:

            return (
                "BLOCK",
                "High model risk is independently supported "
                "by historical illicit network evidence."
            )

        return (
            "STEP-UP",
            "High model risk is present, but historical "
            "network evidence does not justify automatic blocking."
        )


    if risk >= medium:

        if network_ratio >= network_limit:

            return (
                "BLOCK",
                "Borderline model risk is materially "
                "strengthened by historical illicit network evidence."
            )

        return (
            "STEP-UP",
            "Model confidence is within the review range "
            "and requires additional verification."
        )


    if network_ratio >= 0.50:

        return (
            "STEP-UP",
            "Low transaction-level model risk conflicts "
            "with strong historical network evidence."
        )


    return (
        "ALLOW",
        "Low model risk and no strong historical fraud-network signal."
    )


decision, decision_reason = make_decision(
    defender_probability,
    prior_illicit_ratio,
    high_threshold,
    medium_threshold,
    network_threshold
)


# ============================================================
# STATUS CLASS
# ============================================================

if decision == "BLOCK":

    status_class = "status-block"

elif decision == "STEP-UP":

    status_class = "status-review"

else:

    status_class = "status-allow"


# ============================================================
# TRANSACTION HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">'
    'TRANSACTION INVESTIGATION'
    '</div>',
    unsafe_allow_html=True
)

transaction_header_left, transaction_header_right = st.columns(
    [7, 2]
)

with transaction_header_left:

    st.markdown(
        f"### {transaction_id}"
    )

    st.caption(
        f"Time step {time_step} · "
        f"Historical network context enabled"
    )


with transaction_header_right:

    st.markdown(
        f"""
        <div style="text-align:right;">
            <span class="status {status_class}">
                {decision}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="metric-wrap">
            <div class="metric-label">
                Model Risk
            </div>
            <div class="metric-value">
                {defender_probability:.2%}
            </div>
            <div class="metric-caption">
                Transaction-level illicit probability
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-wrap">
            <div class="metric-label">
                Historical Illicit Ratio
            </div>
            <div class="metric-value">
                {prior_illicit_ratio:.2%}
            </div>
            <div class="metric-caption">
                Confirmed illicit activity in prior network history
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-wrap">
            <div class="metric-label">
                Prior Network Activity
            </div>
            <div class="metric-value">
                {prior_connections:,}
            </div>
            <div class="metric-caption">
                Connected transactions before current event
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-wrap">
            <div class="metric-label">
                Decision
            </div>
            <div class="metric-value"
                 style="font-size:1.7rem;">
                {decision}
            </div>
            <div class="metric-caption">
                Current policy outcome
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="panel">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="panel-title">Evidence Summary</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="panel-subtitle">'
    'Independent signals considered in the current decision'
    '</div>',
    unsafe_allow_html=True
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        f"""
        <div class="evidence-row">
            <div class="evidence-label">Model Assessment</div>
            <div class="evidence-value">
                {defender_probability:.2%} transaction-level illicit probability
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_b:

    if prior_connections > 0:
        network_summary = (
            f"{prior_illicit:,} of "
            f"{prior_connections:,} prior connected "
            f"transactions were historically illicit."
        )
    else:
        network_summary = (
            "No prior connected transaction history identified."
        )

    st.markdown(
        f"""
        <div class="evidence-row">
            <div class="evidence-label">Network Assessment</div>
            <div class="evidence-value">
                {network_summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_c:
    st.markdown(
        f"""
        <div class="evidence-row">
            <div class="evidence-label">Decision</div>
            <div class="evidence-value">
                {decision}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)
# ============================================================
# MAIN TABS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

tab_overview, tab_network, tab_policy = st.tabs(
    [
        "Assessment",
        "Network Intelligence",
        "Policy Simulation"
    ]
)


# ============================================================
# ASSESSMENT
# ============================================================

with tab_overview:

    left, right = st.columns(
        [1, 1]
    )


    # --------------------------------------------------------
    # MODEL ASSESSMENT
    # --------------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    Model Assessment
                </div>
                <div class="panel-subtitle">
                    Defender output from CounterRisk v3
                </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            min(
                defender_probability,
                1.0
            )
        )

        st.markdown(
            f"""
            <div style="
                color:#dbe4ef;
                font-size:1.2rem;
                font-weight:700;
                margin-top:0.6rem;">
                {defender_probability:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )


        if defender_probability >= high_threshold:

            st.markdown(
                '<div class="muted" style="margin-top:0.7rem;">'
                'High model risk. Transaction-level evidence '
                'supports intervention.'
                '</div>',
                unsafe_allow_html=True
            )

        elif defender_probability >= medium_threshold:

            st.markdown(
                '<div class="muted" style="margin-top:0.7rem;">'
                'Borderline model risk. Additional evidence '
                'should inform the final action.'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="muted" style="margin-top:0.7rem;">'
                'Low transaction-level model risk.'
                '</div>',
                unsafe_allow_html=True
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # NETWORK ASSESSMENT
    # --------------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    Historical Network Assessment
                </div>
                <div class="panel-subtitle">
                    Evidence available before the current transaction
                </div>
            """,
            unsafe_allow_html=True
        )


        rows = [
            (
                "Prior connected transactions",
                f"{prior_connections:,}"
            ),
            (
                "Prior illicit connections",
                f"{prior_illicit:,}"
            ),
            (
                "Connected wallets",
                f"{connected_wallets:,}"
            ),
            (
                "Historical illicit ratio",
                f"{prior_illicit_ratio:.2%}"
            )
        ]


        for label, value in rows:

            st.markdown(
                f"""
                <div class="evidence-row">
                    <div class="evidence-label">
                        {label}
                    </div>
                    <div class="evidence-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DECISION RATIONALE
    # --------------------------------------------------------

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'Decision Rationale'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-subtitle">'
        'How the two evidence layers were reconciled'
        '</div>',
        unsafe_allow_html=True
    )


    if decision == "BLOCK":

        rationale = (
            f"CounterRisk recommends blocking the transaction. "
            f"The Defender assigns {defender_probability:.2%} "
            f"fraud probability and historical network intelligence "
            f"shows {prior_illicit:,} illicit connections across "
            f"{prior_connections:,} prior connected transactions."
        )

    elif decision == "STEP-UP":

        rationale = (
            f"CounterRisk recommends additional verification. "
            f"The Defender assigns {defender_probability:.2%} "
            f"fraud probability, while historical network evidence "
            f"does not provide sufficient independent support for "
            f"an automatic block."
        )

    else:

        rationale = (
            f"CounterRisk recommends allowing the transaction. "
            f"The Defender assigns {defender_probability:.2%} "
            f"fraud probability and no strong historical illicit "
            f"network signal was identified."
        )


    st.markdown(
        f"""
        <div class="decision-box">
            <div class="decision-action">
                {decision}
            </div>
            <div class="decision-reason">
                {rationale}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# NETWORK INTELLIGENCE
# ============================================================

with tab_network:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'Network Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-subtitle">'
        'Historical wallet context associated with the transaction'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="network">

            <div class="network-node network-node-risk">
                Current Transaction<br>
                {transaction_id}
            </div>

            <div class="network-line">
                │
            </div>

            <div>
                <span class="network-node">
                    {connected_wallets} connected wallet(s)
                </span>
            </div>

            <div class="network-line">
                │
            </div>

            <div>
                <span class="network-node">
                    {prior_connections:,} prior connected transactions
                </span>
            </div>

            <div class="network-line">
                │
            </div>

            <div>
                <span class="network-node network-node-risk">
                    {prior_illicit:,} historically illicit connections
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    if prior_illicit_ratio >= 0.50:

        st.error(
            "Strong historical network signal. "
            "A substantial proportion of prior connected activity "
            "was historically confirmed illicit."
        )

    elif prior_illicit_ratio >= 0.20:

        st.warning(
            "Meaningful historical network signal detected."
        )

    elif prior_connections > 0:

        st.info(
            "Historical network activity exists, but no strong "
            "confirmed illicit pattern is present."
        )

    else:

        st.success(
            "No prior connected transaction history is available."
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# POLICY SIMULATION
# ============================================================

with tab_policy:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'Policy Simulation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-subtitle">'
        'Evaluate how intervention thresholds affect the current transaction'
        '</div>',
        unsafe_allow_html=True
    )


    sim_high = st.slider(
        "High-risk threshold",
        0.50,
        0.99,
        high_threshold,
        0.01,
        key="policy_high"
    )

    sim_medium = st.slider(
        "Review threshold",
        0.20,
        0.80,
        medium_threshold,
        0.01,
        key="policy_medium"
    )

    sim_network = st.slider(
        "Network evidence threshold",
        0.00,
        1.00,
        network_threshold,
        0.05,
        key="policy_network"
    )


    sim_decision, sim_reason = make_decision(
        defender_probability,
        prior_illicit_ratio,
        sim_high,
        sim_medium,
        sim_network
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    if sim_decision == "BLOCK":

        st.error(
            f"SIMULATED OUTCOME  ·  BLOCK"
        )

    elif sim_decision == "STEP-UP":

        st.warning(
            f"SIMULATED OUTCOME  ·  STEP-UP"
        )

    else:

        st.success(
            f"SIMULATED OUTCOME  ·  ALLOW"
        )


    st.markdown(
        f"""
        <div class="decision-reason">
            {sim_reason}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "Technical Details"
):

    col_a, col_b = st.columns(2)

    with col_a:

        st.write(
            "**Dataset**"
        )

        st.write(
            "Elliptic++"
        )

        st.write(
            "**Model**"
        )

        st.write(
            "Random Forest · CounterRisk v3"
        )

        st.write(
            "**Transaction features**"
        )

        st.write(
            "183"
        )

    with col_b:

        st.write(
            "**Training period**"
        )

        st.write(
            "Time steps 1–34"
        )

        st.write(
            "**Evaluation period**"
        )

        st.write(
            "Time steps 35–49"
        )

        st.write(
            "**Network intelligence**"
        )

        st.write(
            "Leakage-safe historical wallet activity"
        )