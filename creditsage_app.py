"""
CreditSage Loan Advisory Agent — Streamlit UI

This is the main application file that provides:
  - A polished sidebar for applicant selection and snapshot display
  - An applicant dashboard with charts (credit score gauge, income breakdown,
    risk profile visualization)
  - A chat interface for conversational loan advisory
  - Integration with the LLM-based router and tool-calling layer
  - Session-based conversation memory

Launch via: python run.py  (which calls streamlit run creditsage_app.py)
"""

import streamlit as st
import pandas as pd
import os

from agent.memory import ConversationMemory
from agent.tools import (
    get_applicant_summary,
    check_eligibility,
    assess_risk_profile,
    calculate_emi,
    CREDIT_SCORE_THRESHOLDS,
    INCOME_THRESHOLDS,
)
from agent.router import route_query

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="CreditSage Loan Advisory Agent",
    page_icon="💳",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for polished look
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(120deg, #1e88e5, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk badge */
    .risk-low { color: #4caf50; font-weight: 700; }
    .risk-medium { color: #ff9800; font-weight: 700; }
    .risk-high { color: #f44336; font-weight: 700; }

    /* Eligibility badge */
    .eligible-yes {
        background: #1b5e20;
        color: #a5d6a7;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .eligible-no {
        background: #b71c1c;
        color: #ef9a9a;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }

    /* Chat message area spacing */
    .stChatMessage {
        margin-bottom: 0.5rem;
    }

    /* Quick action buttons */
    .quick-btn {
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Dataset Loading (cached to avoid re-reading CSV on every interaction)
# ──────────────────────────────────────────────

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the loan applications CSV once and cache it.

    Uses @st.cache_data to avoid re-reading the CSV file on every
    Streamlit rerun. This is important for performance.

    Returns:
        pd.DataFrame with all 25 applicant records.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "creditsage_loan_applications.csv")
    return pd.read_csv(csv_path)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize session_state variables on first load.

    Creates the ConversationMemory instance and tracking variables
    so they persist across Streamlit reruns.
    """
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "current_applicant_id" not in st.session_state:
        st.session_state.current_applicant_id = None
    if "applicant_data" not in st.session_state:
        st.session_state.applicant_data = None
    if "show_dashboard" not in st.session_state:
        st.session_state.show_dashboard = True


init_session_state()


# ──────────────────────────────────────────────
# Helper: Format INR values (Indian number system)
# ──────────────────────────────────────────────

def format_inr(value: float) -> str:
    """Format a number in Indian Rupee notation.

    Indian number system groups: last 3 digits, then groups of 2.
    Example: 35,00,000 (thirty-five lakh)

    Args:
        value: Numeric value to format.

    Returns:
        Formatted string with Indian comma grouping.
    """
    if value < 0:
        return f"-{format_inr(-value)}"

    parts = f"{value:.0f}"

    if len(parts) <= 3:
        return f"{parts}"
    else:
        last_three = parts[-3:]
        remaining = parts[:-3]
        groups = []
        while remaining:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        groups.reverse()
        return ",".join(groups) + "," + last_three


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with applicant selection, snapshot, and controls."""
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='margin:0; color: #4fc3f7;'>💳 CreditSage</h2>
            <p style='color: #888; font-size: 0.9rem; margin-top: 0.2rem;'>AI Loan Advisory Agent</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # Load dataset for applicant list
        df = load_dataset()
        applicant_ids = df["applicant_id"].tolist()
        applicant_names = df["name"].tolist()

        # Build display options: "ID — Name"
        options = [f"{aid} — {name}" for aid, name in zip(applicant_ids, applicant_names)]

        selected = st.selectbox(
            "Select Applicant",
            options=options,
            index=None,
            placeholder="Choose an applicant...",
        )

        if selected:
            selected_id = int(selected.split(" — ")[0])

            # Only reload if the applicant changed
            if selected_id != st.session_state.current_applicant_id:
                st.session_state.current_applicant_id = selected_id
                result = get_applicant_summary(selected_id)
                if result.get("found"):
                    st.session_state.applicant_data = result["summary"]
                    st.session_state.memory.set_applicant(selected_id, result["summary"])
                    st.session_state.memory.history = []  # Reset chat when switching
                    st.session_state.show_dashboard = True
                else:
                    st.error("Applicant not found.")
                    st.session_state.applicant_data = None

            # Show applicant info
            if st.session_state.applicant_data:
                data = st.session_state.applicant_data
                st.success(f"**{data['Name']}** (ID: {data['Applicant ID']})")

                # Expandable Applicant Snapshot
                with st.expander("📋 Applicant Snapshot", expanded=True):
                    st.markdown(f"**Age:** {data['Age']} | **Gender:** {data['Gender']}")
                    st.markdown(f"**City:** {data['City']}")
                    st.markdown(f"**Employment:** {data['Employment Type']}")
                    st.markdown(f"**Employer:** {data['Employer']}")
                    st.markdown(f"**Years at Job:** {data['Years at Current Job']}")
                    st.markdown("---")
                    st.markdown(f"**Monthly Income:** ₹{format_inr(data['Monthly Income (INR)'])}")
                    st.markdown(f"**Credit Score:** {data['Credit Score']}")
                    st.markdown(f"**Existing EMI:** ₹{format_inr(data['Existing EMI (INR)'])}/mo")
                    st.markdown("---")
                    st.markdown(f"**Loan Purpose:** {data['Loan Purpose']}")
                    st.markdown(f"**Requested:** ₹{format_inr(data['Requested Amount (INR)'])}")
                    st.markdown(f"**Tenure:** {data['Preferred Tenure (months)']} months")
                    st.markdown(f"**Down Payment:** ₹{format_inr(data['Down Payment (INR)'])}")
                    st.markdown(f"**Collateral:** {data['Collateral']}")

        st.markdown("---")

        # Toggle dashboard view
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.show_dashboard = True
                st.rerun()
        with col2:
            if st.button("💬 Chat", use_container_width=True):
                st.session_state.show_dashboard = False
                st.rerun()

        st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.memory.history = []
            st.rerun()

        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.75rem;'>"
            "Powered by Groq LLaMA 3.3 70B<br/>"
            "CreditSage Financial Technologies</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Dashboard View — Charts & Visual Analytics
# ──────────────────────────────────────────────

def render_dashboard():
    """Render the applicant analytics dashboard with charts and metrics."""

    data = st.session_state.applicant_data
    applicant_id = st.session_state.current_applicant_id

    # ── Header Metrics Row ──
    st.markdown("### Applicant Overview")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Monthly Income", f"₹{format_inr(data['Monthly Income (INR)'])}")
    with m2:
        score = data['Credit Score']
        score_delta = "Good" if score >= 700 else ("Fair" if score >= 650 else "Poor")
        st.metric("Credit Score", str(score), delta=score_delta,
                  delta_color="normal" if score >= 700 else ("off" if score >= 650 else "inverse"))
    with m3:
        st.metric("Requested Loan", f"₹{format_inr(data['Requested Amount (INR)'])}")
    with m4:
        st.metric("Net Income After EMI", f"₹{format_inr(data['Net Monthly Income After EMI (INR)'])}")

    st.markdown("---")

    # ── Row 2: Eligibility + Risk Profile ──
    col_elig, col_risk = st.columns(2)

    with col_elig:
        st.markdown("#### Eligibility Check")
        elig_result = check_eligibility(applicant_id)

        if elig_result.get("eligible"):
            st.markdown('<span class="eligible-yes">ELIGIBLE</span>', unsafe_allow_html=True)
            st.success(f"Meets all criteria for a **{elig_result['loan_purpose']}** loan.")
        else:
            st.markdown('<span class="eligible-no">NOT ELIGIBLE</span>', unsafe_allow_html=True)
            for fc in elig_result.get("failed_criteria", []):
                st.error(f"{fc}")

        # Show thresholds table
        purpose = data['Loan Purpose']
        threshold_data = {
            "Criterion": ["Age Range", "Min Credit Score", "Min Monthly Income"],
            "Required": [
                "21 – 60 years",
                str(CREDIT_SCORE_THRESHOLDS.get(purpose, 650)),
                f"₹{format_inr(INCOME_THRESHOLDS.get(purpose, 25000))}",
            ],
            "Applicant": [
                f"{data['Age']} years",
                str(data['Credit Score']),
                f"₹{format_inr(data['Monthly Income (INR)'])}",
            ],
            "Status": [
                "Pass" if 21 <= data['Age'] <= 60 else "Fail",
                "Pass" if data['Credit Score'] >= CREDIT_SCORE_THRESHOLDS.get(purpose, 650) else "Fail",
                "Pass" if data['Monthly Income (INR)'] >= INCOME_THRESHOLDS.get(purpose, 25000) else "Fail",
            ],
        }
        st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)

    with col_risk:
        st.markdown("#### Risk Profile")
        risk_result = assess_risk_profile(applicant_id)

        tier = risk_result["risk_tier"]
        tier_class = f"risk-{tier.lower()}"
        st.markdown(f'Risk Tier: <span class="{tier_class}">{tier.upper()}</span> '
                    f'(Score: {risk_result["score"]}/3.0)', unsafe_allow_html=True)

        # Risk factor breakdown as a bar chart
        factors = risk_result["factor_breakdown"]
        factor_names = []
        factor_scores = []
        factor_colors = []
        for name, info in factors.items():
            display_name = name.replace("_", " ").title()
            factor_names.append(display_name)
            factor_scores.append(info["score"])

        risk_df = pd.DataFrame({
            "Factor": factor_names,
            "Risk Score": factor_scores,
        })
        st.bar_chart(risk_df.set_index("Factor"), color="#4fc3f7", height=250)

        # Factor details
        for name, info in factors.items():
            display = name.replace("_", " ").title()
            level_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(info["level"], "⚪")
            st.caption(f"{level_icon} **{display}**: {info['detail']}")

    st.markdown("---")

    # ── Row 3: Financial Breakdown Charts ──
    col_income, col_credit = st.columns(2)

    with col_income:
        st.markdown("#### Income & Obligations Breakdown")
        monthly_income = data['Monthly Income (INR)']
        existing_emi = data['Existing EMI (INR)']
        available = monthly_income - existing_emi

        income_df = pd.DataFrame({
            "Category": ["Available Income", "Existing EMI"],
            "Amount (INR)": [available, existing_emi],
        })
        st.bar_chart(income_df.set_index("Category"), color="#7c4dff", height=280)

        dti = data['Debt-to-Income Ratio']
        dti_status = "Healthy" if dti < 0.2 else ("Caution" if dti <= 0.4 else "High Risk")
        st.markdown(f"**Debt-to-Income Ratio:** {dti:.1%} ({dti_status})")
        st.progress(min(dti, 1.0))

    with col_credit:
        st.markdown("#### Credit Score Analysis")
        score = data['Credit Score']

        # Credit score gauge visualization
        score_pct = (score - 300) / 600  # Normalize 300-900 to 0-1
        st.progress(score_pct)

        score_range = (
            "Excellent (750+)" if score >= 750 else
            "Good (700-749)" if score >= 700 else
            "Fair (650-699)" if score >= 650 else
            "Poor (<650)"
        )
        st.markdown(f"**{score}** / 900 — {score_range}")

        # Loan-to-Income Ratio
        lti = data['Loan-to-Income Ratio']
        st.markdown(f"**Loan-to-Income Ratio:** {lti:.1f}x annual income")
        lti_status = "Comfortable" if lti < 3 else ("Stretched" if lti <= 5 else "Over-ambitious")
        st.caption(f"Assessment: {lti_status}")

        # Employment stability
        years = data['Years at Current Job']
        st.markdown(f"**Employment Stability:** {years} years at {data['Employer']}")
        stability = "Stable" if years > 3 else ("Moderate" if years >= 1 else "Short tenure")
        st.caption(f"Assessment: {stability}")

    st.markdown("---")

    # ── Row 4: EMI Estimate Preview ──
    st.markdown("#### EMI Quick Estimate")
    req_amount = data['Requested Amount (INR)'] - data['Down Payment (INR)']
    tenure = data['Preferred Tenure (months)']

    # Calculate EMI at different sample rates
    rates = [8.5, 10.0, 11.5, 13.0]
    emi_data = []
    for rate in rates:
        result = calculate_emi(req_amount, rate, tenure)
        emi_data.append({
            "Interest Rate": f"{rate}%",
            "Monthly EMI": f"₹{format_inr(result['emi'])}",
            "Total Interest": f"₹{format_inr(result['total_interest'])}",
            "Total Payable": f"₹{format_inr(result['total_payable'])}",
        })

    st.dataframe(
        pd.DataFrame(emi_data),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Estimates based on principal ₹{format_inr(req_amount)} "
        f"(requested ₹{format_inr(data['Requested Amount (INR)'])} "
        f"- down payment ₹{format_inr(data['Down Payment (INR)'])}) "
        f"for {tenure} months."
    )


# ──────────────────────────────────────────────
# Chat View
# ──────────────────────────────────────────────

def render_chat():
    """Render the main chat interface with message history and input."""

    # Quick action buttons
    st.markdown("##### Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("Check Eligibility", use_container_width=True):
            _send_query("Am I eligible for this loan?")
    with qa2:
        if st.button("Show Loan Products", use_container_width=True):
            _send_query("What loan products are available for me?")
    with qa3:
        if st.button("Calculate EMI", use_container_width=True):
            _send_query("Calculate the EMI for my loan request.")
    with qa4:
        if st.button("Risk Assessment", use_container_width=True):
            _send_query("What is my risk profile?")

    st.markdown("---")

    # Display conversation history
    for msg in st.session_state.memory.get_history():
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask Alex about eligibility, EMI, loan options...")

    if user_input:
        _process_message(user_input)


def _send_query(query: str):
    """Handle quick action button clicks by processing the query.

    Args:
        query: Pre-defined query string from quick action buttons.
    """
    _process_message(query)


def _process_message(user_input: str):
    """Process a user message: display it, call router, show response.

    Args:
        user_input: The user's query text.
    """
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Add to memory
    st.session_state.memory.add_message("user", user_input)

    # Generate response with spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Alex is analyzing your query..."):
            try:
                response = route_query(user_input, st.session_state.memory)
            except Exception as e:
                response = (
                    f"I apologize, but I encountered an error: {str(e)}. "
                    "Please check your API key in the .env file and try again."
                )

        st.markdown(response)

    # Add response to memory
    st.session_state.memory.add_message("assistant", response)


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def main():
    """Application entry point — renders sidebar, header, and active view."""

    render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>CreditSage Loan Advisory Agent</h1>
        <p>AI-powered loan guidance for smarter financial decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if an applicant is loaded
    if st.session_state.current_applicant_id is None:
        st.markdown("---")
        st.info("👈 **Select an Applicant ID from the sidebar to begin.**")

        # Show a preview of the dataset
        st.markdown("### 📊 Dataset Preview")
        df = load_dataset()
        st.dataframe(
            df[["applicant_id", "name", "age", "city", "employment_type",
                "monthly_income", "credit_score", "loan_purpose", "requested_amount"]],
            use_container_width=True,
            hide_index=True,
        )
        return

    st.markdown("---")

    # Toggle between Dashboard and Chat views
    if st.session_state.show_dashboard:
        render_dashboard()

        # Always show chat below dashboard
        st.markdown("---")
        st.markdown("### 💬 Chat with Alex")
        render_chat()
    else:
        render_chat()


if __name__ == "__main__":
    main()
