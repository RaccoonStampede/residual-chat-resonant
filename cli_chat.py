#!/usr/bin/env python3
"""
cli_chat.py — Interactive CLI for ResidualChat-Resonant.

Commands:
  /confirm    — confirm last response
  /reject     — reject last response
  /teach <text> — teach the model a fragment
  /status     — show current status
  /save       — save session
  /load <id>  — load a session by ID
  /list       — list saved sessions
  /export     — export session to JSON file
  /help       — show this help
  /quit       — exit
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

from chat_interface_utils import ChatSession, format_status
from config import get_config
from residual_chat_resonant import ResidualChatResonant
from storage import StorageManager

logger = logging.getLogger(__name__)

ONTOLOGY = (
    "Language is mostly residual. "
    "Speech is residual tension finding a lower energy shape in words. "
    "Thought is residual echo navigating a phase landscape. "
    "Meaning is what remains after the noise collapses."
)

HELP_TEXT = """
Commands:
  /confirm         — Confirm last response (reinforce)
  /reject          — Reject last response
  /teach <text>    — Teach the model a new fragment
  /status          — Show session status
  /save            — Save current session to storage
  /load <id>       — Load a previous session by ID
  /list            — List all saved sessions
  /export          — Export current session as JSON
  /help            — Show this help
  /quit  /exit     — Exit the chat

Press Enter on an empty line to skip.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_snapshot(chat: ResidualChatResonant) -> dict:
    """Extract a lightweight snapshot of chat state."""
    try:
        state: dict = {
            "turn": chat.turn,
            "confirms": chat.confirms,
            "rejects": chat.rejects,
            "fragments": [f for f in chat.field.fragments],
        }
    except Exception:
        state = {}
    return state


class CLIChat:
    def __init__(self):
        self.cfg = get_config()
        self.storage: StorageManager | None = None
        if self.cfg.ENABLE_PERSISTENCE:
            self.storage = StorageManager(db_path=self.cfg.DB_PATH)

        self.chat = ResidualChatResonant(seed=137, use_eggroll=True)
        self.chat.inject_dense(ONTOLOGY, passes=2)

        self.session_id: str | None = None
        self._last_message_id: int | None = None

        if self.storage:
            self.session_id = self.storage.create_session(
                metadata={"created_via": "cli", "created_at": _now()}
            )
            print(f"[Session: {self.session_id}]")

    # ------------------------------------------------------------------
    # Session persistence helpers
    # ------------------------------------------------------------------

    def _save_session(self):
        if not self.storage or not self.session_id:
            print("Persistence disabled.")
            return
        meta = {
            "turn": self.chat.turn,
            "confirms": self.chat.confirms,
            "rejects": self.chat.rejects,
            "frag_count": len(self.chat.field.fragments),
        }
        self.storage.update_session(self.session_id, meta)
        snapshot = _make_snapshot(self.chat)
        self.storage.save_snapshot(self.session_id, snapshot)
        print(f"Session saved: {self.session_id}")

    def _load_session(self, session_id: str):
        if not self.storage:
            print("Persistence disabled.")
            return
        session = self.storage.get_session(session_id)
        if session is None:
            print(f"Session not found: {session_id}")
            return
        messages = self.storage.get_messages(session_id)
        print(f"Loaded session {session_id} ({len(messages)} messages)")
        self.session_id = session_id
        # Restore fragments from snapshot
        snap = self.storage.get_latest_snapshot(session_id)
        if snap:
            frags = snap["state"].get("fragments", [])
            for frag in frags:
                if frag not in self.chat.field.fragments:
                    self.chat.field.store(frag)
            print(f"Restored {len(frags)} fragments from snapshot.")
        # Show last few messages
        for msg in messages[-6:]:
            role_label = "You" if msg["role"] == "user" else "AI"
            print(f"  [{role_label}] {msg['content'][:80]}")

    def _list_sessions(self):
        if not self.storage:
            print("Persistence disabled.")
            return
        sessions, total = self.storage.list_sessions(per_page=20)
        print(f"Sessions ({total} total):")
        for s in sessions:
            meta = s["metadata"]
            turns = meta.get("turn", "?")
            frags = meta.get("frag_count", "?")
            print(f"  {s['id']}  turns={turns}  frags={frags}  updated={s['updated_at'][:19]}")

    def _export_session(self):
        if not self.storage or not self.session_id:
            print("No active session to export.")
            return
        data = self.storage.export_session_json(self.session_id)
        filename = f"export_{self.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {filename}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print("\n╔═══════════════════════════════════════╗")
        print("║   ResidualChat-Resonant  CLI v1.0     ║")
        print("╚═══════════════════════════════════════╝")
        print("Type /help for commands. Type /quit to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                self._save_session()
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    self._save_session()
                    print("Goodbye.")
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/confirm":
                    result = self.chat.confirm()
                    print(f"  ✅ {result}")
                    if self.storage and self.session_id and self._last_message_id:
                        self.storage.update_message_metadata(
                            self._last_message_id, {"feedback": "confirm"}
                        )
                elif cmd == "/reject":
                    result = self.chat.reject()
                    print(f"  ❌ {result}")
                    if self.storage and self.session_id and self._last_message_id:
                        self.storage.update_message_metadata(
                            self._last_message_id, {"feedback": "reject"}
                        )
                elif cmd == "/status":
                    print(f"  {self.chat.status()}")
                elif cmd == "/teach":
                    if not arg:
                        print("  Usage: /teach <text>")
                    else:
                        self.chat.teach(arg, rounds=3)
                        print(f"  Taught: {arg[:60]}")
                        if self.storage and self.session_id:
                            sig = hashlib.sha256(arg.encode()).hexdigest()[:16]
                            self.storage.save_fragment(arg, sig, self.session_id)
                elif cmd == "/save":
                    self._save_session()
                elif cmd == "/load":
                    if not arg:
                        print("  Usage: /load <session_id>")
                    else:
                        self._load_session(arg.strip())
                elif cmd == "/list":
                    self._list_sessions()
                elif cmd == "/export":
                    self._export_session()
                else:
                    print(f"  Unknown command: {cmd}. Type /help.")
                continue

            # Normal chat
            try:
                reply = self.chat.say(user_input)
            except Exception as exc:
                logger.exception("Error in say()")
                print(f"  [Error] {exc}")
                continue

            print(f"\nAI: {reply}\n")

            if self.storage and self.session_id:
                try:
                    self.storage.save_message(self.session_id, "user", user_input)
                    self._last_message_id = self.storage.save_message(
                        self.session_id, "assistant", reply
                    )
                except Exception:
                    logger.warning("Failed to persist message", exc_info=True)


def handle_input(session: ChatSession, text: str) -> tuple:
    """Process one line of CLI input against a ChatSession.

    Returns ``(message, should_exit)`` so callers can decide whether to quit.
    """
    text = text.strip()
    if not text:
        return ("", False)

    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            return ("Goodbye.", True)
        elif cmd == "/help":
            return (HELP_TEXT, False)
        elif cmd == "/confirm":
            return (session.confirm(), False)
        elif cmd == "/reject":
            return (session.reject(), False)
        elif cmd == "/status":
            return (format_status(session.status()), False)
        elif cmd == "/teach":
            if not arg:
                return ("Usage: /teach <text>", False)
            return (session.teach(arg), False)
        else:
            return (f"Unknown command: {cmd}. Type /help.", False)

    reply = session.ask(text)
    return (reply, False)


def main():
    cli = CLIChat()
    cli.run()


if __name__ == "__main__":
    main()
