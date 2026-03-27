# CreditSage Loan Advisory Agent

> AI-powered loan advisory system built for CreditSage Financial Technologies.
> Replaces manual 15–20 minute loan advisory sessions with instant, consistent, compliance-ready AI guidance.

---

## Architecture Overview

The system uses a **Router Pattern** with LLM-based intent classification. Every user query flows through a central router that classifies intent using Groq (LLaMA 3.3 70B), then dispatches to specialised tool handlers.

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Chat UI                     │
│  (Sidebar: Applicant Selector + Snapshot)                │
└────────────────────┬────────────────────────────────────┘
                     │ User Query
                     ▼
          ┌─────────────────────┐
          │   LLM Router Agent  │  ← Claude classifies intent
          │  (temperature=0.0)  │    (deterministic routing)
          └──────┬──────────────┘
                 │
     ┌───────────┼───────────────┬──────────────┐
     ▼           ▼               ▼              ▼
┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────┐
│ELIGIBILITY│ │PRODUCT_   │ │ EMI_CALC │ │   GENERAL    │
│           │ │MATCH      │ │          │ │              │
│ Tools:    │ │ Tools:    │ │ Tools:   │ │ No tools —   │
│ • check_  │ │ • get_    │ │ • calc_  │ │ LLM answers  │
│  eligibi- │ │  loan_    │ │  emi()   │ │ from context │
│  lity()   │ │  products │ │          │ │ + knowledge  │
│ • assess_ │ │  ()       │ │          │ │              │
│  risk_    │ │ • check_  │ │          │ │              │
│  profile()│ │  eligibi- │ │          │ │              │
│           │ │  lity()   │ │          │ │              │
└─────┬─────┘ └─────┬─────┘ └────┬─────┘ └──────┬───────┘
      │             │            │               │
      └─────────────┴────────────┴───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  LLM Response Gen   │  ← Claude synthesises tool
              │  (temperature=0.3)  │    results into natural
              │  (top_p=0.9)        │    language advisory
              └──────┬──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ Session Memory│  ← Streamlit session_state
              │ (last 10 msgs)│    stores full conversation
              └──────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| `creditsage_app.py` | Streamlit UI — sidebar, chat, applicant snapshot |
| `agent/router.py` | LLM-based intent classifier + tool dispatcher |
| `agent/tools.py` | 5 tool functions + Groq/OpenAI-compatible tool schemas |
| `agent/memory.py` | ConversationMemory class (session_state backed) |
| `agent/prompts.py` | All LLM prompts centralised as string constants |
| `run.py` | Entry point — launches Streamlit app |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Groq API key (free at https://console.groq.com — no credit card needed)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/creditsage-agent-ROLLNUMBER
cd creditsage-agent-ROLLNUMBER
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Dataset

Place `creditsage_loan_applications.csv` in the project root directory. The file contains 25 loan applicant records with fields like income, credit score, loan purpose, and more.

### Run

```bash
python run.py
```

This launches the Streamlit app at `http://localhost:8501`.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (free at https://console.groq.com) |

Create a `.env` file from the provided `.env.example`:
```
GROQ_API_KEY=gsk_your_key_here
```

---

## Tool Reference

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `check_eligibility` | Validates age (21–60), income, credit score against loan-type-specific rules | `applicant_id` |
| `get_loan_products` | Returns up to 3 matching loan products by purpose and amount, sorted by interest rate | `loan_purpose`, `requested_amount` |
| `calculate_emi` | Reducing balance EMI calculation with total interest payable | `principal`, `annual_rate`, `tenure_months` |
| `assess_risk_profile` | 4-factor risk scoring: credit score, DTI, employment stability, LTI | `applicant_id` |
| `get_applicant_summary` | Full applicant profile from CSV with derived metrics (DTI, LTI, net income) | `applicant_id` |

### Eligibility Rules

| Loan Type | Min Credit Score | Min Monthly Income |
|-----------|-----------------|-------------------|
| Personal | 650 | ₹25,000 |
| Home | 700 | ₹40,000 |
| Vehicle | 650 | ₹20,000 |
| Business | 700 | ₹50,000 |

All applicants must be aged 21–60.

### Risk Scoring Factors

| Factor | Low (1) | Medium (2) | High (3) |
|--------|---------|------------|----------|
| Credit Score | ≥750 | 650–749 | <650 |
| Debt-to-Income | <20% | 20–40% | >40% |
| Employment Stability | >3 years | 1–3 years | <1 year |
| Loan-to-Income | <3x | 3–5x | >5x |

Final tier: Average score <1.75 = Low, 1.75–2.25 = Medium, >2.25 = High.

---

## Agentic Design Decisions

1. **Router Pattern over direct tool-calling**: The router classifies intent first, then dispatches to the appropriate handler. This ensures each query type gets the right combination of tools and context, rather than relying on the LLM to always pick optimally from all 5 tools simultaneously.

2. **LLM classification over keyword matching**: Keyword-based routing is brittle — it breaks on paraphrased queries ("Am I qualified?" vs "Check my eligibility" vs "Can I get this loan?"). LLM classification handles semantic equivalence, sarcasm, multi-intent queries, and unseen phrasings without maintenance.

3. **temperature=0.0 for routing, 0.3 for responses**: The router needs deterministic, reproducible classification — the same query should always map to the same intent. Response generation benefits from slight creativity (0.3) to produce natural, varied advisory language while staying factual.

4. **session_state for memory**: For a proof-of-concept demo, session_state is the simplest approach — zero infrastructure, instant setup, and it naturally scopes memory to each browser session. A production system would use Redis or a database.

5. **Groq's native tool-use API**: Instead of manually parsing JSON from the LLM, we use Groq's built-in OpenAI-compatible tool_use mechanism with proper schemas. This gives us structured, validated tool calls with typed parameters — and it's completely free.

---

## Business Value

CreditSage's human advisors currently take 15–20 minutes per applicant inquiry, handling 50–80 daily. This AI agent delivers the same advisory — eligibility checks, product matching, EMI calculations, and risk assessments — in under 10 seconds with zero manual errors. With the mobile app launch expected to surge volume 5x to 250–400 daily inquiries, the agent scales instantly without additional hiring, while enforcing compliance rules consistently across every interaction. Human advisors are freed to handle only edge-case escalations, improving both throughput and job satisfaction.
