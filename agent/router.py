"""
CreditSage Loan Advisory Agent — Router (LLM-Based Intent Classification)

Implements the Router Pattern using Groq (LLaMA 3.3 70B):
  1. Classify the user's query intent using the LLM (NOT keyword matching).
  2. Route to the appropriate handler that calls the right tool(s).
  3. Pass tool results + conversation history to the LLM for a final
     natural-language response.

This design was chosen over keyword matching because:
  - LLM classification handles paraphrased, ambiguous, and multi-intent queries
    far better than fragile regex/keyword rules.
  - It generalises to unseen phrasings without constant rule maintenance.
  - Groq provides free, ultra-fast inference — ideal for real-time advisory.
"""

import json
import os

from groq import Groq
from dotenv import load_dotenv

from agent.prompts import (
    ADVISOR_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    TOOL_CALLING_SYSTEM_PROMPT,
)
from agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

# Initialise the Groq client once at module level
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model used for all LLM calls — LLaMA 3.3 70B on Groq is free, fast, and
# excellent at tool calling and instruction following.
MODEL = "llama-3.3-70b-versatile"


# ──────────────────────────────────────────────
# Step 1: Intent Classification
# ──────────────────────────────────────────────

def classify_intent(user_message: str) -> str:
    """Use the LLM to classify user query intent.

    Sends the user's message to Groq with ROUTER_SYSTEM_PROMPT
    and returns one of: ELIGIBILITY, PRODUCT_MATCH, EMI_CALC, GENERAL.

    Args:
        user_message: The raw text the user typed.

    Returns:
        Intent label string (one of the four valid intents).
    """
    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=20,       # Intent label is a single word — keep tokens minimal
        temperature=0.0,     # Deterministic output for consistent routing; we want
                             # the same query to always map to the same intent.
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    intent = response.choices[0].message.content.strip().upper()

    # Validate the intent is one of our known routes
    valid_intents = {"ELIGIBILITY", "PRODUCT_MATCH", "EMI_CALC", "GENERAL"}
    if intent not in valid_intents:
        # Fallback: if the LLM returns something unexpected, default to GENERAL
        intent = "GENERAL"

    return intent


# ──────────────────────────────────────────────
# Step 2 & 3: Route + Generate Response
# ──────────────────────────────────────────────

def route_query(user_message: str, memory) -> str:
    """Route a user query to the appropriate handler and return a response.

    This is the main entry point called by the Streamlit UI. It:
      1. Classifies the intent via LLM.
      2. Calls the appropriate tool(s) using Groq's tool-use API.
      3. Generates a final natural-language response grounded in tool results.

    Args:
        user_message: The raw text the user typed.
        memory: ConversationMemory instance with session history + applicant context.

    Returns:
        The assistant's response string to display in the chat.
    """
    # Get applicant context for prompt injection
    applicant_context = memory.get_applicant_context()

    # Classify intent (Step 1)
    intent = classify_intent(user_message)

    # For GENERAL queries, no tool call is needed — respond directly
    if intent == "GENERAL":
        return _generate_general_response(user_message, memory, applicant_context)

    # For tool-requiring intents, use Groq's tool-use flow
    return _generate_tool_response(user_message, memory, applicant_context, intent)


def _generate_general_response(user_message: str, memory, applicant_context: str) -> str:
    """Handle GENERAL intent queries without tool calls.

    Uses LLM knowledge + applicant context from memory to answer
    general questions about loan processes, documents, trends, etc.

    Args:
        user_message: The user's query.
        memory: ConversationMemory instance.
        applicant_context: Formatted applicant profile string.

    Returns:
        LLM-generated response string.
    """
    system_prompt = ADVISOR_SYSTEM_PROMPT.format(applicant_context=applicant_context)

    # Build messages from conversation history + current query
    messages = [{"role": "system", "content": system_prompt}]
    messages += memory.get_context_window(max_messages=10)
    # Ensure the latest user message is included
    if not messages or messages[-1].get("content") != user_message:
        messages.append({"role": "user", "content": user_message})

    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,     # Sufficient for detailed advisory responses without
                             # being wasteful. Most answers fit in 300–600 tokens.
        temperature=0.3,     # Slightly creative for natural conversation, but low
                             # enough to stay factual and avoid hallucination.
        top_p=0.9,           # Nucleus sampling at 0.9 trims the long tail of unlikely
                             # tokens, keeping responses focused and coherent.
        messages=messages,
    )

    return response.choices[0].message.content


def _generate_tool_response(user_message: str, memory, applicant_context: str, intent: str) -> str:
    """Handle tool-requiring intents via Groq's tool-use API.

    Sends the user message with tool schemas to the LLM, lets it decide
    which tools to call, executes those tools, and then generates a
    final response grounded in the tool results.

    Args:
        user_message: The user's query.
        memory: ConversationMemory instance.
        applicant_context: Formatted applicant profile string.
        intent: Classified intent (ELIGIBILITY, PRODUCT_MATCH, or EMI_CALC).

    Returns:
        LLM-generated response string grounded in tool results.
    """
    system_prompt = TOOL_CALLING_SYSTEM_PROMPT.format(applicant_context=applicant_context)

    # Build conversation messages
    messages = [{"role": "system", "content": system_prompt}]
    messages += memory.get_context_window(max_messages=10)
    if not messages or messages[-1].get("content") != user_message:
        messages.append({"role": "user", "content": user_message})

    # First LLM call — let the model decide which tools to call
    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.0,     # Deterministic for tool selection — we want reliable,
                             # reproducible tool calls, not creative ones.
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",  # Let the LLM decide which tools to invoke
    )

    response_message = response.choices[0].message

    # Process tool calls in a loop until the model stops requesting tools
    # Max 5 iterations to prevent infinite loops
    max_iterations = 5
    iteration = 0

    while response_message.tool_calls and iteration < max_iterations:
        iteration += 1

        # Append the assistant's message (with tool calls) to the conversation
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ],
        })

        # Execute each tool call and append results
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Execute the tool
            if tool_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[tool_name](**tool_args)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Append tool result as a "tool" role message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # Send tool results back to the LLM for the next step
        # Switch to the advisor prompt for the final response generation
        messages[0] = {"role": "system", "content": ADVISOR_SYSTEM_PROMPT.format(applicant_context=applicant_context)}

        response = _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.3,     # Slightly creative for natural final response
            top_p=0.9,           # Focused nucleus sampling
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

    # Extract the final text response
    final_text = response_message.content or ""

    return final_text if final_text else "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
