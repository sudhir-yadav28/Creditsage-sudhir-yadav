"""
CreditSage Loan Advisory Agent — Tool Definitions

This module implements the 5 core tools that the LLM can call:
  1. check_eligibility   — Validates applicant against loan-type-specific rules
  2. get_loan_products    — Returns matching loan products from hardcoded catalog
  3. calculate_emi        — Reducing-balance EMI formula
  4. assess_risk_profile  — 4-factor risk scoring
  5. get_applicant_summary— Full applicant profile with derived metrics

Each tool has a corresponding Groq/OpenAI-compatible tool schema in TOOL_SCHEMAS.
"""

import os
import pandas as pd

# ──────────────────────────────────────────────
# Dataset loader (cached externally via st.cache_data in app.py)
# ──────────────────────────────────────────────

# Resolve CSV path relative to project root (one level up from agent/)
_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "creditsage_loan_applications.csv")


def _load_data() -> pd.DataFrame:
    """Load the loan applications CSV into a DataFrame.

    Returns:
        pd.DataFrame: All 25 applicant records.
    """
    return pd.read_csv(_CSV_PATH)


def _get_applicant(applicant_id: int) -> dict | None:
    """Retrieve a single applicant row as a dict.

    Args:
        applicant_id: Unique identifier (1–25).

    Returns:
        dict of applicant fields, or None if not found.
    """
    df = _load_data()
    row = df[df["applicant_id"] == applicant_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ──────────────────────────────────────────────
# Tool 1: check_eligibility
# ──────────────────────────────────────────────

# Eligibility thresholds — chosen based on Indian banking norms:
#   Age 21–60: RBI mandates minimum 18; most lenders require 21+.
#              Upper bound 60 aligns with retirement-age risk limits.
#   Credit score:
#     Personal / Vehicle: >= 650 — moderate risk tolerance for smaller,
#       shorter-term loans.
#     Home / Business: >= 700 — stricter because of larger exposure and
#       longer tenure; aligns with CIBIL "good" band.
#   Minimum income:
#     Personal: ₹25,000 — covers basic repayment capacity for unsecured loans.
#     Home:     ₹40,000 — higher to sustain large, long-tenure EMIs.
#     Vehicle:  ₹20,000 — lower as vehicle loans are asset-backed.
#     Business: ₹50,000 — highest; business loans carry cash-flow risk.

CREDIT_SCORE_THRESHOLDS = {
    "Personal": 650,
    "Home": 700,
    "Vehicle": 650,
    "Business": 700,
}

INCOME_THRESHOLDS = {
    "Personal": 25000,
    "Home": 40000,
    "Vehicle": 20000,
    "Business": 50000,
}


def check_eligibility(applicant_id: int) -> dict:
    """Check whether an applicant meets minimum eligibility criteria.

    Evaluates three criteria based on loan type:
      1. Age between 21 and 60 (inclusive)
      2. Credit score above loan-type-specific floor
      3. Monthly income above loan-type-specific minimum

    Args:
        applicant_id: Unique applicant identifier from the CSV.

    Returns:
        dict with keys: eligible (bool), reason (str),
        failed_criteria (list), applicant_name (str), loan_purpose (str).
    """
    applicant = _get_applicant(applicant_id)
    if applicant is None:
        return {"error": f"Applicant with ID {applicant_id} not found."}

    failed = []
    loan_purpose = applicant["loan_purpose"]
    name = applicant["name"]
    age = int(applicant["age"])
    credit_score = int(applicant["credit_score"])
    income = float(applicant["monthly_income"])

    # --- Criterion 1: Age must be 21–60 inclusive ---
    if age < 21 or age > 60:
        failed.append(f"Age {age} is outside the eligible range of 21–60 years.")

    # --- Criterion 2: Credit score floor per loan type ---
    min_score = CREDIT_SCORE_THRESHOLDS.get(loan_purpose, 650)
    if credit_score < min_score:
        failed.append(
            f"Credit score {credit_score} is below the minimum {min_score} "
            f"required for {loan_purpose} loans."
        )

    # --- Criterion 3: Minimum monthly income per loan type ---
    min_income = INCOME_THRESHOLDS.get(loan_purpose, 25000)
    if income < min_income:
        failed.append(
            f"Monthly income ₹{income:,.0f} is below the minimum "
            f"₹{min_income:,.0f} required for {loan_purpose} loans."
        )

    eligible = len(failed) == 0
    reason = (
        f"{name} meets all eligibility criteria for a {loan_purpose} loan."
        if eligible
        else f"{name} does not meet the following criteria: " + "; ".join(failed)
    )

    return {
        "eligible": eligible,
        "reason": reason,
        "failed_criteria": failed,
        "applicant_name": name,
        "loan_purpose": loan_purpose,
    }


# ──────────────────────────────────────────────
# Tool 2: get_loan_products
# ──────────────────────────────────────────────

# Hardcoded loan product catalog — modeled after real Indian bank products.
# Each category has 4–5 products spanning different lender tiers
# (public-sector banks, private banks, NBFCs) to give applicants variety.

LOAN_PRODUCTS = {
    "Personal": [
        {"name": "SBI Xpress Personal Loan", "lender": "SBI", "interest_rate_pa": 10.75, "max_tenure_months": 72, "processing_fee_pct": 1.0, "min_amount": 50000, "max_amount": 2000000},
        {"name": "HDFC Personal Loan", "lender": "HDFC Bank", "interest_rate_pa": 10.50, "max_tenure_months": 60, "processing_fee_pct": 1.5, "min_amount": 100000, "max_amount": 4000000},
        {"name": "ICICI Instant Personal Loan", "lender": "ICICI Bank", "interest_rate_pa": 10.85, "max_tenure_months": 60, "processing_fee_pct": 2.0, "min_amount": 50000, "max_amount": 2500000},
        {"name": "Bajaj Finserv Flexi Loan", "lender": "Bajaj Finserv", "interest_rate_pa": 11.00, "max_tenure_months": 84, "processing_fee_pct": 1.75, "min_amount": 25000, "max_amount": 3500000},
    ],
    "Home": [
        {"name": "SBI Home Loan", "lender": "SBI", "interest_rate_pa": 8.40, "max_tenure_months": 360, "processing_fee_pct": 0.35, "min_amount": 500000, "max_amount": 50000000},
        {"name": "HDFC Home Loan", "lender": "HDFC Ltd", "interest_rate_pa": 8.50, "max_tenure_months": 360, "processing_fee_pct": 0.50, "min_amount": 500000, "max_amount": 50000000},
        {"name": "ICICI Home Loan", "lender": "ICICI Bank", "interest_rate_pa": 8.60, "max_tenure_months": 360, "processing_fee_pct": 0.50, "min_amount": 300000, "max_amount": 50000000},
        {"name": "LIC HFL Home Loan", "lender": "LIC HFL", "interest_rate_pa": 8.35, "max_tenure_months": 360, "processing_fee_pct": 0.25, "min_amount": 200000, "max_amount": 50000000},
        {"name": "Axis Home Loan", "lender": "Axis Bank", "interest_rate_pa": 8.75, "max_tenure_months": 300, "processing_fee_pct": 1.0, "min_amount": 500000, "max_amount": 50000000},
    ],
    "Vehicle": [
        {"name": "SBI Car Loan", "lender": "SBI", "interest_rate_pa": 8.65, "max_tenure_months": 84, "processing_fee_pct": 0.50, "min_amount": 100000, "max_amount": 5000000},
        {"name": "HDFC Car Loan", "lender": "HDFC Bank", "interest_rate_pa": 8.75, "max_tenure_months": 84, "processing_fee_pct": 0.50, "min_amount": 100000, "max_amount": 5000000},
        {"name": "ICICI Auto Loan", "lender": "ICICI Bank", "interest_rate_pa": 8.90, "max_tenure_months": 84, "processing_fee_pct": 0.75, "min_amount": 150000, "max_amount": 5000000},
        {"name": "Mahindra Finance Vehicle Loan", "lender": "Mahindra Finance", "interest_rate_pa": 9.50, "max_tenure_months": 60, "processing_fee_pct": 1.0, "min_amount": 50000, "max_amount": 3000000},
    ],
    "Business": [
        {"name": "SBI SME Loan", "lender": "SBI", "interest_rate_pa": 11.00, "max_tenure_months": 60, "processing_fee_pct": 1.0, "min_amount": 200000, "max_amount": 50000000},
        {"name": "HDFC Business Growth Loan", "lender": "HDFC Bank", "interest_rate_pa": 11.50, "max_tenure_months": 48, "processing_fee_pct": 1.5, "min_amount": 500000, "max_amount": 40000000},
        {"name": "ICICI Business Loan", "lender": "ICICI Bank", "interest_rate_pa": 12.00, "max_tenure_months": 48, "processing_fee_pct": 2.0, "min_amount": 300000, "max_amount": 50000000},
        {"name": "Bajaj Finserv Business Loan", "lender": "Bajaj Finserv", "interest_rate_pa": 13.00, "max_tenure_months": 60, "processing_fee_pct": 2.5, "min_amount": 100000, "max_amount": 30000000},
        {"name": "Tata Capital Business Loan", "lender": "Tata Capital", "interest_rate_pa": 12.50, "max_tenure_months": 48, "processing_fee_pct": 2.0, "min_amount": 200000, "max_amount": 50000000},
    ],
}


def get_loan_products(loan_purpose: str, requested_amount: float) -> dict:
    """Return up to 3 matching loan products for a given purpose and amount.

    Filters the hardcoded product catalog by loan_purpose, then keeps
    only products whose [min_amount, max_amount] range includes the
    requested_amount. Results are sorted by interest_rate_pa ascending
    so the cheapest option appears first.

    Args:
        loan_purpose: One of Personal, Home, Vehicle, Business.
        requested_amount: Loan amount the applicant wants (INR).

    Returns:
        dict with keys: products (list[dict]), loan_purpose (str),
        requested_amount (float).
    """
    products = LOAN_PRODUCTS.get(loan_purpose, [])
    if not products:
        return {
            "products": [],
            "loan_purpose": loan_purpose,
            "requested_amount": requested_amount,
            "message": f"No products found for loan purpose '{loan_purpose}'.",
        }

    # Filter by amount eligibility
    matching = [
        p for p in products
        if p["min_amount"] <= requested_amount <= p["max_amount"]
    ]

    # Sort by interest rate (lowest first) and take top 3
    matching.sort(key=lambda x: x["interest_rate_pa"])
    top_products = matching[:3]

    return {
        "products": top_products,
        "loan_purpose": loan_purpose,
        "requested_amount": requested_amount,
    }


# ──────────────────────────────────────────────
# Tool 3: calculate_emi
# ──────────────────────────────────────────────

def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """Calculate monthly EMI using the standard reducing-balance formula.

    Formula: EMI = P × r × (1+r)^n / ((1+r)^n – 1)
    where:
      P = principal loan amount
      r = monthly interest rate (annual_rate / 12 / 100)
      n = tenure in months

    Args:
        principal: Loan principal amount in INR.
        annual_rate: Annual interest rate as a percentage (e.g., 10.5 for 10.5%).
        tenure_months: Repayment period in months.

    Returns:
        dict with keys: emi, total_interest, total_payable,
        principal, annual_rate, tenure_months.
    """
    if principal <= 0 or annual_rate <= 0 or tenure_months <= 0:
        return {"error": "Principal, annual_rate, and tenure_months must all be positive."}

    r = annual_rate / (12 * 100)  # monthly interest rate as decimal
    n = tenure_months

    # Standard reducing-balance EMI formula
    emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)

    total_payable = emi * n
    total_interest = total_payable - principal

    return {
        "emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payable": round(total_payable, 2),
        "principal": round(principal, 2),
        "annual_rate": annual_rate,
        "tenure_months": tenure_months,
    }


# ──────────────────────────────────────────────
# Tool 4: assess_risk_profile
# ──────────────────────────────────────────────

# Risk-scoring thresholds — justified by standard underwriting norms:
#
# 1. Credit score (CIBIL):
#    >= 750 = Low risk  — "excellent" band; default rate < 2 %
#    650–749 = Medium   — "good/fair" band; default rate ~5 %
#    < 650 = High       — "poor" band; default rate >10 %
#
# 2. Debt-to-income (DTI) ratio (existing_emi / monthly_income):
#    < 0.2 = Low  — borrower retains 80 %+ of income; strong buffer
#    0.2–0.4 = Medium — industry standard caution zone
#    > 0.4 = High — over-leveraged; most banks reject above 50 %
#
# 3. Employment stability (years_at_current_job):
#    > 3 years = Low   — demonstrates career stability, steady income
#    1–3 years = Medium — probationary risk window
#    < 1 year = High    — job-hop risk; income continuity uncertain
#
# 4. Loan-to-income (LTI) ratio (requested_amount / annual_income):
#    < 3 = Low   — comfortable repayment horizon
#    3–5 = Medium — stretched but manageable
#    > 5 = High   — over-ambitious ask relative to income

def assess_risk_profile(applicant_id: int) -> dict:
    """Compute a risk tier (Low / Medium / High) for an applicant.

    Uses a 4-factor weighted scoring model. Each factor scores 1 (Low),
    2 (Medium), or 3 (High). The average score determines the final tier.

    Args:
        applicant_id: Unique applicant identifier from the CSV.

    Returns:
        dict with keys: risk_tier (str), score (float),
        factor_breakdown (dict), applicant_name (str).
    """
    applicant = _get_applicant(applicant_id)
    if applicant is None:
        return {"error": f"Applicant with ID {applicant_id} not found."}

    credit_score = int(applicant["credit_score"])
    monthly_income = float(applicant["monthly_income"])
    existing_emi = float(applicant["existing_emi"])
    years_at_job = float(applicant["years_at_current_job"])
    requested_amount = float(applicant["requested_amount"])
    annual_income = monthly_income * 12

    factors = {}

    # Factor 1: Credit score
    if credit_score >= 750:
        factors["credit_score"] = {"score": 1, "level": "Low", "value": credit_score, "detail": "Excellent credit (≥750)"}
    elif credit_score >= 650:
        factors["credit_score"] = {"score": 2, "level": "Medium", "value": credit_score, "detail": "Fair credit (650–749)"}
    else:
        factors["credit_score"] = {"score": 3, "level": "High", "value": credit_score, "detail": "Poor credit (<650)"}

    # Factor 2: Debt-to-income ratio
    dti = existing_emi / monthly_income if monthly_income > 0 else 1.0
    if dti < 0.2:
        factors["debt_to_income"] = {"score": 1, "level": "Low", "value": round(dti, 3), "detail": "DTI < 20% — strong buffer"}
    elif dti <= 0.4:
        factors["debt_to_income"] = {"score": 2, "level": "Medium", "value": round(dti, 3), "detail": "DTI 20–40% — caution zone"}
    else:
        factors["debt_to_income"] = {"score": 3, "level": "High", "value": round(dti, 3), "detail": "DTI > 40% — over-leveraged"}

    # Factor 3: Employment stability
    if years_at_job > 3:
        factors["employment_stability"] = {"score": 1, "level": "Low", "value": years_at_job, "detail": "Stable employment (>3 years)"}
    elif years_at_job >= 1:
        factors["employment_stability"] = {"score": 2, "level": "Medium", "value": years_at_job, "detail": "Moderate stability (1–3 years)"}
    else:
        factors["employment_stability"] = {"score": 3, "level": "High", "value": years_at_job, "detail": "Short tenure (<1 year)"}

    # Factor 4: Loan-to-income ratio
    lti = requested_amount / annual_income if annual_income > 0 else 10.0
    if lti < 3:
        factors["loan_to_income"] = {"score": 1, "level": "Low", "value": round(lti, 2), "detail": "LTI < 3x — comfortable"}
    elif lti <= 5:
        factors["loan_to_income"] = {"score": 2, "level": "Medium", "value": round(lti, 2), "detail": "LTI 3–5x — stretched"}
    else:
        factors["loan_to_income"] = {"score": 3, "level": "High", "value": round(lti, 2), "detail": "LTI > 5x — over-ambitious"}

    # Average score determines final tier
    avg_score = sum(f["score"] for f in factors.values()) / len(factors)
    if avg_score < 1.75:
        risk_tier = "Low"
    elif avg_score <= 2.25:
        risk_tier = "Medium"
    else:
        risk_tier = "High"

    return {
        "risk_tier": risk_tier,
        "score": round(avg_score, 2),
        "factor_breakdown": factors,
        "applicant_name": applicant["name"],
    }


# ──────────────────────────────────────────────
# Tool 5: get_applicant_summary
# ──────────────────────────────────────────────

def get_applicant_summary(applicant_id: int) -> dict:
    """Retrieve and format all available data for an applicant.

    In addition to raw CSV fields, computes derived metrics:
      - debt_to_income_ratio: existing_emi / monthly_income
      - loan_to_income_ratio: requested_amount / (monthly_income * 12)
      - net_monthly_income_after_emi: monthly_income - existing_emi

    Args:
        applicant_id: Unique applicant identifier from the CSV.

    Returns:
        dict with keys: found (bool), summary (dict with human-readable labels).
    """
    applicant = _get_applicant(applicant_id)
    if applicant is None:
        return {"found": False, "error": f"Applicant with ID {applicant_id} not found."}

    monthly_income = float(applicant["monthly_income"])
    existing_emi = float(applicant["existing_emi"])
    requested_amount = float(applicant["requested_amount"])
    annual_income = monthly_income * 12

    summary = {
        "Applicant ID": int(applicant["applicant_id"]),
        "Name": applicant["name"],
        "Age": int(applicant["age"]),
        "Gender": applicant["gender"],
        "City": applicant["city"],
        "Employment Type": applicant["employment_type"],
        "Employer": applicant["employer_name"],
        "Monthly Income (INR)": monthly_income,
        "Credit Score": int(applicant["credit_score"]),
        "Existing EMI (INR)": existing_emi,
        "Loan Purpose": applicant["loan_purpose"],
        "Requested Amount (INR)": requested_amount,
        "Preferred Tenure (months)": int(applicant["preferred_tenure_months"]),
        "Down Payment (INR)": float(applicant["down_payment"]),
        "Collateral": applicant["collateral"],
        "Years at Current Job": float(applicant["years_at_current_job"]),
        # Derived metrics
        "Debt-to-Income Ratio": round(existing_emi / monthly_income, 3) if monthly_income > 0 else 0,
        "Loan-to-Income Ratio": round(requested_amount / annual_income, 2) if annual_income > 0 else 0,
        "Net Monthly Income After EMI (INR)": round(monthly_income - existing_emi, 2),
    }

    return {"found": True, "summary": summary}


# ──────────────────────────────────────────────
# Groq / OpenAI-Compatible Tool Schemas
# ──────────────────────────────────────────────
# These schemas follow the OpenAI function-calling format used by Groq.
# Passed to client.chat.completions.create(tools=TOOL_SCHEMAS).

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": (
                "Check whether a loan applicant meets minimum eligibility criteria "
                "based on age (21–60), credit score floor per loan type (>=650 for "
                "Personal/Vehicle, >=700 for Home/Business), and minimum monthly "
                "income thresholds per loan type. Returns eligibility verdict with "
                "reasons for any failed criteria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "Unique applicant identifier (1–25) from the loan applications dataset.",
                    },
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_products",
            "description": (
                "Return a list of up to 3 matching loan products with interest rates, "
                "max tenure, and processing fees based on the applicant's loan purpose "
                "(Personal/Home/Vehicle/Business) and requested amount. Products are "
                "sorted by interest rate ascending (cheapest first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_purpose": {
                        "type": "string",
                        "description": "Loan category: 'Personal', 'Home', 'Vehicle', or 'Business'.",
                        "enum": ["Personal", "Home", "Vehicle", "Business"],
                    },
                    "requested_amount": {
                        "type": "number",
                        "description": "Loan amount requested by the applicant in INR.",
                    },
                },
                "required": ["loan_purpose", "requested_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_emi",
            "description": (
                "Calculate monthly EMI (Equated Monthly Installment) using the standard "
                "reducing balance formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1). "
                "Returns EMI amount, total interest payable, and total amount payable. "
                "Use this for single calculations or call multiple times for tenure/rate comparisons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Loan principal amount in INR.",
                    },
                    "annual_rate": {
                        "type": "number",
                        "description": "Annual interest rate as a percentage (e.g., 10.5 for 10.5%).",
                    },
                    "tenure_months": {
                        "type": "integer",
                        "description": "Repayment period in months (e.g., 36 for 3 years, 60 for 5 years).",
                    },
                },
                "required": ["principal", "annual_rate", "tenure_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_risk_profile",
            "description": (
                "Assess an applicant's risk tier (Low / Medium / High) based on 4 factors: "
                "credit score, debt-to-income ratio, employment stability, and loan-to-income "
                "ratio. Returns the risk tier, average score, and detailed factor breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "Unique applicant identifier (1–25) from the loan applications dataset.",
                    },
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_applicant_summary",
            "description": (
                "Retrieve the complete profile for a given applicant from the CSV dataset, "
                "including demographics, financial details, loan request info, existing "
                "obligations, and derived metrics (debt-to-income ratio, loan-to-income ratio, "
                "net monthly income after EMI)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "Unique applicant identifier (1–25) from the loan applications dataset.",
                    },
                },
                "required": ["applicant_id"],
            },
        },
    },
]

# Mapping from tool name to callable — used by the router to dispatch calls
TOOL_FUNCTIONS = {
    "check_eligibility": check_eligibility,
    "get_loan_products": get_loan_products,
    "calculate_emi": calculate_emi,
    "assess_risk_profile": assess_risk_profile,
    "get_applicant_summary": get_applicant_summary,
}
