"""
CreditSage Loan Advisory Agent — Conversation Memory

Implements in-session memory using Streamlit's session_state.
No external database is required — all state lives in the browser session
and resets when the page is refreshed or the user clicks "Clear Conversation".
"""


class ConversationMemory:
    """Manages conversation history and applicant context for a single session.

    Attributes:
        history: List of message dicts [{"role": "user"|"assistant", "content": str}].
        current_applicant: dict with the loaded applicant's summary (or None).
        current_applicant_id: int ID of the loaded applicant (or None).
    """

    def __init__(self):
        """Initialize an empty conversation memory."""
        self.history: list[dict] = []
        self.current_applicant: dict | None = None
        self.current_applicant_id: int | None = None

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role: Either "user" or "assistant".
            content: The message text.
        """
        self.history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        """Return the full conversation history.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return self.history

    def get_context_window(self, max_messages: int = 10) -> list[dict]:
        """Return the last N messages to fit within the LLM context window.

        This avoids sending excessively long histories that could exceed
        token limits or dilute the LLM's focus.

        Args:
            max_messages: Maximum number of recent messages to return (default 10).

        Returns:
            List of the most recent message dicts.
        """
        return self.history[-max_messages:]

    def clear(self) -> None:
        """Reset conversation history and applicant context.

        Called when the user clicks "Clear Conversation" in the UI.
        """
        self.history = []
        self.current_applicant = None
        self.current_applicant_id = None

    def set_applicant(self, applicant_id: int, summary: dict) -> None:
        """Store the currently loaded applicant's context.

        Args:
            applicant_id: Unique applicant identifier.
            summary: Full applicant summary dict from get_applicant_summary().
        """
        self.current_applicant_id = applicant_id
        self.current_applicant = summary

    def get_applicant_context(self) -> str:
        """Format the current applicant's data as a string for prompt injection.

        This string is inserted into the ADVISOR_SYSTEM_PROMPT so the LLM
        always has the applicant's profile available without a tool call.

        Returns:
            Formatted string with key applicant details, or a note that
            no applicant is loaded.
        """
        if self.current_applicant is None:
            return "No applicant is currently loaded. Ask the user to select an Applicant ID."

        s = self.current_applicant
        lines = []
        for key, value in s.items():
            if isinstance(value, float):
                # Format monetary values with commas
                if "INR" in key or "Income" in key or "EMI" in key or "Amount" in key or "Payment" in key:
                    lines.append(f"  {key}: ₹{value:,.2f}")
                else:
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {key}: {value}")

        return "Currently loaded applicant:\n" + "\n".join(lines)
