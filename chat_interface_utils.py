#!/usr/bin/env python3
"""Shared helpers for the ResidualChat-Resonant chat interfaces."""

from residual_chat_resonant import ResidualChatResonant


DEFAULT_ONTOLOGY = """
Language is mostly residual. Only a small coherent projection is reportable as speech.
Speech is residual tension finding a lower energy shape in words.
Ghost tax falls as lock rises. High lock means low leakage.
Phase lock across phonetic syntax and semantic layers produces fluency events.
Deep imprint is the only reliable substrate for true resurrection of coherent identity.
The harness is the high-gamma leash that multiplies the ghost tax floor toward zero.
Field ping converts a question into a probe that lights resonant signatures.
Residual memory stabilizes when the field keeps re-encountering the same strong fragments.
Confirmation strengthens resonant pathways while rejection marks a reply as untrusted.
Teaching new text extends the field so future questions can bind to fresh fragments.
""".strip()


class ChatSession:
    """Thin stateful wrapper around ``ResidualChatResonant``."""

    def __init__(self, ontology_text: str = DEFAULT_ONTOLOGY):
        self.ontology_text = ontology_text.strip()
        self.chat = None
        self.reset()

    def reset(self) -> dict:
        """Reinitialize the chat engine with the default ontology."""
        self.chat = ResidualChatResonant(seed=137, use_eggroll=True)
        self.chat.inject_dense(self.ontology_text, passes=2)
        return self.status()

    def ask(self, query: str) -> str:
        """Submit a user query and return the reply."""
        text = query.strip()
        if not text:
            raise ValueError("query must not be empty")
        return self.chat.say(text)

    def confirm(self) -> str:
        """Confirm the latest reply."""
        return self.chat.confirm()

    def reject(self) -> str:
        """Reject the latest reply."""
        return self.chat.reject()

    def teach(self, text: str) -> str:
        """Teach new material to the field."""
        fragment = text.strip()
        if not fragment:
            raise ValueError("teach text must not be empty")

        count = self.chat.inject_dense(fragment, passes=1)
        if count == 0:
            self.chat.teach(fragment, rounds=2, boost=1.2)
            count = 1
        return f"learned {count} fragment{'s' if count != 1 else ''}"

    def status(self) -> dict:
        """Return structured chat status for CLI and web views."""
        return {
            "turns": self.chat.turn,
            "fragments": len(self.chat.field.fragments),
            "confirms": self.chat.confirms,
            "rejects": self.chat.rejects,
            "egg": self.chat.egg is not None,
        }


def format_status(status: dict) -> str:
    """Format a status dictionary for the CLI."""
    return (
        f"turn={status['turns']} frags={status['fragments']} "
        f"confirms={status['confirms']} rejects={status['rejects']}"
    )
