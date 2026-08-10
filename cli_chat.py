#!/usr/bin/env python3
"""Interactive CLI chat for ResidualChat-Resonant."""

from chat_interface_utils import ChatSession, format_status


COMMAND_HELP = "Commands: /confirm /reject /status /teach <text> /quit"


def handle_input(session: ChatSession, raw_text: str):
    """Handle a single line of CLI input."""
    text = raw_text.strip()
    if not text:
        return "Enter a prompt or command.", False
    if text in {"/quit", "/exit"}:
        return "Goodbye.", True
    if text == "/confirm":
        return session.confirm(), False
    if text == "/reject":
        return session.reject(), False
    if text == "/status":
        return format_status(session.status()), False
    if text.startswith("/teach"):
        payload = text[len("/teach"):].strip()
        if not payload:
            return "Usage: /teach <text>", False
        try:
            return session.teach(payload), False
        except ValueError as exc:
            return str(exc), False
    if text.startswith("/"):
        return f"Unknown command. {COMMAND_HELP}", False

    try:
        return session.ask(text), False
    except ValueError as exc:
        return str(exc), False


def main():
    """Run the interactive command loop."""
    session = ChatSession()
    print("ResidualChat-Resonant CLI")
    print(COMMAND_HELP)
    print(format_status(session.status()))

    while True:
        try:
            line = input("> ")
        except EOFError:
            print("\nGoodbye.")
            break
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        message, should_exit = handle_input(session, line)
        print(message)
        if should_exit:
            break


if __name__ == "__main__":
    main()
