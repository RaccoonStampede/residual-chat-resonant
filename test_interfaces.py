#!/usr/bin/env python3
"""Focused tests for the CLI and web chat interfaces."""

from chat_interface_utils import ChatSession
from cli_chat import handle_input
from web_chat import Flask, create_app


def assert_ok(condition, message):
    if not condition:
        raise AssertionError(message)


def test_cli_flow():
    print("=" * 64)
    print("1. CLI flow")
    print("=" * 64)
    session = ChatSession()

    reply, should_exit = handle_input(session, "What is residual tension?")
    print(f"  reply     : {reply}")
    assert_ok(not should_exit, "question should not exit")
    assert_ok(isinstance(reply, str) and len(reply) > 0, "reply should be non-empty")

    message, _ = handle_input(session, "/confirm")
    print(f"  confirm   : {message}")
    assert_ok("confirmed" in message, "confirm should succeed after a reply")

    message, _ = handle_input(session, "/teach Residual harmony grows when the field stabilizes.")
    print(f"  teach     : {message}")
    assert_ok("learned" in message, "teach should acknowledge learning")

    message, _ = handle_input(session, "/status")
    print(f"  status    : {message}")
    assert_ok("turn=" in message and "frags=" in message, "status output should be formatted")

    message, should_exit = handle_input(session, "/quit")
    print(f"  quit      : {message}")
    assert_ok(should_exit, "quit should exit")


def test_web_flow():
    print("\n" + "=" * 64)
    print("2. Web API flow")
    print("=" * 64)
    if Flask is None:
        print("  SKIP: Flask not installed")
        return

    app = create_app(ChatSession())
    client = app.test_client()

    status_response = client.get("/api/status")
    status_data = status_response.get_json()
    print(f"  status    : {status_data}")
    assert_ok(status_response.status_code == 200, "status endpoint should succeed")
    assert_ok(status_data["status"]["fragments"] > 0, "ontology should load on startup")

    chat_response = client.post("/api/chat", json={"query": "What is the harness?"})
    chat_data = chat_response.get_json()
    print(f"  chat      : {chat_data}")
    assert_ok(chat_response.status_code == 200, "chat endpoint should succeed")
    assert_ok(len(chat_data["response"]) > 0, "chat response should be non-empty")

    confirm_response = client.post("/api/confirm", json={})
    confirm_data = confirm_response.get_json()
    print(f"  confirm   : {confirm_data}")
    assert_ok("confirmed" in confirm_data["message"], "confirm endpoint should succeed")

    teach_response = client.post("/api/teach", json={"text": "Residual anchors improve future recall."})
    teach_data = teach_response.get_json()
    print(f"  teach     : {teach_data}")
    assert_ok("learned" in teach_data["message"], "teach endpoint should acknowledge learning")

    reset_response = client.post("/api/reset", json={})
    reset_data = reset_response.get_json()
    print(f"  reset     : {reset_data}")
    assert_ok(reset_data["status"]["turns"] == 0, "reset should clear turns")

    error_response = client.post("/api/chat", json={"query": ""})
    error_data = error_response.get_json()
    print(f"  error     : {error_data}")
    assert_ok(error_response.status_code == 400, "empty query should fail")


if __name__ == "__main__":
    test_cli_flow()
    test_web_flow()
    print("\nAll interface checks passed.")
