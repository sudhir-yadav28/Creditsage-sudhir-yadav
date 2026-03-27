"""
CreditSage Loan Advisory Agent Package

This package contains the core components of the CreditSage AI-powered
loan advisory system, including tools, router, memory, and prompts.
"""

from agent.tools import (
    check_eligibility,
    get_loan_products,
    calculate_emi,
    assess_risk_profile,
    get_applicant_summary,
    TOOL_SCHEMAS,
)
from agent.router import route_query
from agent.memory import ConversationMemory
from agent.prompts import ROUTER_SYSTEM_PROMPT, ADVISOR_SYSTEM_PROMPT
