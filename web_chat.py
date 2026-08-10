#!/usr/bin/env python3
"""Flask web chat interface for ResidualChat-Resonant."""

from threading import Lock

from chat_interface_utils import ChatSession

try:
    from flask import Flask, jsonify, render_template, request
except ImportError as exc:  # pragma: no cover - exercised when Flask is unavailable
    Flask = None
    _FLASK_IMPORT_ERROR = exc
else:
    _FLASK_IMPORT_ERROR = None


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def create_app(session: ChatSession = None, port: int = DEFAULT_PORT):
    """Create the Flask application."""
    if Flask is None:
        raise RuntimeError(
            "Flask is required to run the web interface. Install it with: pip install Flask"
        ) from _FLASK_IMPORT_ERROR

    app = Flask(__name__)
    state = {"session": session or ChatSession(), "lock": Lock()}
    allowed_origins = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}

    def ok_response(payload=None):
        data = {"ok": True}
        if payload is not None:
            data.update(payload)
        return jsonify(data)

    def error_response(message: str, status_code: int = 400):
        return jsonify({"ok": False, "error": message}), status_code

    def run_action(action):
        with state["lock"]:
            return action(state["session"])

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api/chat", methods=["POST", "OPTIONS"])
    def api_chat():
        if request.method == "OPTIONS":
            return ("", 204)
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return error_response("query is required")
        try:
            response, status = run_action(lambda current: (current.ask(query), current.status()))
        except ValueError as exc:
            return error_response(str(exc))
        return ok_response({"response": response, "status": status})

    @app.route("/api/confirm", methods=["POST", "OPTIONS"])
    def api_confirm():
        if request.method == "OPTIONS":
            return ("", 204)
        message, status = run_action(lambda current: (current.confirm(), current.status()))
        return ok_response({"message": message, "status": status})

    @app.route("/api/reject", methods=["POST", "OPTIONS"])
    def api_reject():
        if request.method == "OPTIONS":
            return ("", 204)
        message, status = run_action(lambda current: (current.reject(), current.status()))
        return ok_response({"message": message, "status": status})

    @app.route("/api/teach", methods=["POST", "OPTIONS"])
    def api_teach():
        if request.method == "OPTIONS":
            return ("", 204)
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return error_response("text is required")
        try:
            message, status = run_action(lambda current: (current.teach(text), current.status()))
        except ValueError as exc:
            return error_response(str(exc))
        return ok_response({"message": message, "status": status})

    @app.route("/api/status", methods=["GET", "OPTIONS"])
    def api_status():
        if request.method == "OPTIONS":
            return ("", 204)
        return ok_response({"status": run_action(lambda current: current.status())})

    @app.route("/api/reset", methods=["POST", "OPTIONS"])
    def api_reset():
        if request.method == "OPTIONS":
            return ("", 204)
        status = run_action(lambda current: current.reset())
        return ok_response({"message": "chat reset", "status": status})

    @app.errorhandler(404)
    def not_found(_error):
        return error_response("not found", 404)

    @app.errorhandler(500)
    def internal_error(_error):
        return error_response("internal server error", 500)

    return app


if __name__ == "__main__":
    if Flask is None:
        raise SystemExit("Flask is required to run this server. Install it with: pip install Flask")
    app = create_app(port=DEFAULT_PORT)
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT)
