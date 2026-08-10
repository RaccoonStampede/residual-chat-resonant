#!/usr/bin/env python3
"""
web_chat.py — Flask web server for ResidualChat-Resonant.

Endpoints:
  GET  /                     — Web UI
  POST /api/chat             — Send a message
  POST /api/confirm          — Confirm last response
  POST /api/reject           — Reject last response
  POST /api/teach            — Teach a fragment
  GET  /api/status           — Session status
  POST /api/sessions         — List sessions (paginated)
  GET  /api/sessions/<id>    — Load session
  DELETE /api/sessions/<id>  — Soft-delete session
  POST /api/export           — Export chat (json/csv/txt)
  POST /api/export-fragments — Export fragments as JSON
  GET  /api/fragments        — Search/list fragments
  POST /api/fragments/bulk-teach — Teach multiple texts
  GET  /api/metrics          — Basic stats
"""

import csv
import hashlib
import io
import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional

from flask import Flask, g, jsonify, render_template, request, send_file

from config import get_config
from residual_chat_resonant import ResidualChatResonant
from storage import StorageManager

cfg = get_config()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------
_storage: Optional[StorageManager] = None


def get_storage() -> Optional[StorageManager]:
    global _storage
    if _storage is None and cfg.ENABLE_PERSISTENCE:
        _storage = StorageManager(db_path=cfg.DB_PATH)
    return _storage


# ------------------------------------------------------------------
# In-memory rate limiter (per IP)
# ------------------------------------------------------------------
_rate_buckets: Dict[str, list] = defaultdict(list)
_rate_lock = threading.Lock()

RATE_LIMIT = cfg.RATE_LIMIT          # requests
RATE_WINDOW = cfg.RATE_LIMIT_WINDOW  # seconds


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[ip]
        _rate_buckets[ip] = [t for t in bucket if now - t < RATE_WINDOW]
        if len(_rate_buckets[ip]) >= RATE_LIMIT:
            return False
        _rate_buckets[ip].append(now)
    return True


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.path.startswith("/static"):
            return f(*args, **kwargs)
        ip = request.remote_addr or "unknown"
        if not _check_rate_limit(ip):
            retry_after = RATE_WINDOW
            resp = jsonify(
                _err("Too many requests. Please slow down.", 429)
            )
            resp.status_code = 429
            resp.headers["Retry-After"] = str(retry_after)
            return resp
        return f(*args, **kwargs)
    return decorated


# ------------------------------------------------------------------
# Session / chat instance (single in-process session)
# ------------------------------------------------------------------
_chat_instance: Optional[ResidualChatResonant] = None
_chat_session_id: Optional[str] = None
_last_message_id: Optional[int] = None

ONTOLOGY = (
    "Language is mostly residual. "
    "Speech is residual tension finding a lower energy shape in words. "
    "Thought is residual echo navigating a phase landscape. "
    "Meaning is what remains after the noise collapses."
)


def get_chat() -> ResidualChatResonant:
    global _chat_instance, _chat_session_id
    if _chat_instance is None:
        _chat_instance = ResidualChatResonant(seed=137, use_eggroll=True)
        _chat_instance.inject_dense(ONTOLOGY, passes=2)
        storage = get_storage()
        if storage:
            _chat_session_id = storage.create_session(
                metadata={"created_via": "web", "created_at": _now()}
            )
            logger.info("Web session created: %s", _chat_session_id)
    return _chat_instance


# ------------------------------------------------------------------
# Response helpers
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: Any = None, message: str = "OK") -> Dict:
    return {
        "ok": True,
        "data": data,
        "message": message,
        "error": None,
        "timestamp": _now(),
    }


def _err(error: str, status: int = 400) -> Dict:
    return {
        "ok": False,
        "data": None,
        "message": None,
        "error": error,
        "timestamp": _now(),
    }


def _validate_text(value: Any, field: str, min_len: int = 1, max_len: int = 4096) -> Optional[str]:
    if not isinstance(value, str):
        return f"{field} must be a string"
    stripped = value.strip()
    if len(stripped) < min_len:
        return f"{field} is too short (min {min_len} chars)"
    if len(stripped) > max_len:
        return f"{field} is too long (max {max_len} chars)"
    return None


# ------------------------------------------------------------------
# Web UI
# ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------
# Chat API
# ------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
@rate_limit
def api_chat():
    global _last_message_id
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "")
        err = _validate_text(query, "query", min_len=1, max_len=2048)
        if err:
            return jsonify(_err(err, 400)), 400

        chat = get_chat()
        reply = chat.say(query.strip())

        storage = get_storage()
        if storage and _chat_session_id:
            try:
                storage.save_message(_chat_session_id, "user", query.strip())
                _last_message_id = storage.save_message(
                    _chat_session_id, "assistant", reply
                )
            except Exception:
                logger.warning("Failed to persist message", exc_info=True)

        return jsonify(_ok({"reply": reply, "session_id": _chat_session_id}))
    except Exception as exc:
        logger.exception("Error in /api/chat")
        return jsonify(_err("Internal error processing your message.", 500)), 500


@app.route("/api/confirm", methods=["POST"])
@rate_limit
def api_confirm():
    global _last_message_id
    try:
        chat = get_chat()
        result = chat.confirm()
        storage = get_storage()
        if storage and _last_message_id:
            try:
                storage.update_message_metadata(_last_message_id, {"feedback": "confirm"})
            except Exception:
                logger.warning("Failed to update message metadata", exc_info=True)
        return jsonify(_ok({"result": result}))
    except Exception as exc:
        logger.exception("Error in /api/confirm")
        return jsonify(_err("Failed to confirm.", 500)), 500


@app.route("/api/reject", methods=["POST"])
@rate_limit
def api_reject():
    global _last_message_id
    try:
        chat = get_chat()
        result = chat.reject()
        storage = get_storage()
        if storage and _last_message_id:
            try:
                storage.update_message_metadata(_last_message_id, {"feedback": "reject"})
            except Exception:
                logger.warning("Failed to update message metadata", exc_info=True)
        return jsonify(_ok({"result": result}))
    except Exception as exc:
        logger.exception("Error in /api/reject")
        return jsonify(_err("Failed to reject.", 500)), 500


@app.route("/api/teach", methods=["POST"])
@rate_limit
def api_teach():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        err = _validate_text(text, "text", min_len=4, max_len=4096)
        if err:
            return jsonify(_err(err, 400)), 400

        rounds = int(data.get("rounds", 3))
        rounds = max(1, min(rounds, 10))

        chat = get_chat()
        chat.teach(text.strip(), rounds=rounds)

        storage = get_storage()
        if storage and _chat_session_id:
            try:
                sig = hashlib.sha256(text.encode()).hexdigest()[:16]
                if not storage.fragment_exists(sig):
                    storage.save_fragment(text.strip(), sig, _chat_session_id)
            except Exception:
                logger.warning("Failed to persist fragment", exc_info=True)

        return jsonify(_ok({"taught": text.strip()[:80]}))
    except Exception as exc:
        logger.exception("Error in /api/teach")
        return jsonify(_err("Failed to teach fragment.", 500)), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        chat = get_chat()
        status_str = chat.status()
        return jsonify(_ok({
            "status": status_str,
            "session_id": _chat_session_id,
            "fragments": len(chat.field.fragments),
            "turn": chat.turn,
            "confirms": chat.confirms,
            "rejects": chat.rejects,
        }))
    except Exception as exc:
        logger.exception("Error in /api/status")
        return jsonify(_err("Failed to get status.", 500)), 500


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------

@app.route("/api/sessions", methods=["POST"])
def api_list_sessions():
    try:
        data = request.get_json(silent=True) or {}
        page = int(data.get("page", 1))
        per_page = int(data.get("per_page", 20))
        storage = get_storage()
        if not storage:
            return jsonify(_err("Persistence disabled.", 503)), 503
        sessions, total = storage.list_sessions(page=page, per_page=per_page)
        return jsonify(_ok({"sessions": sessions, "total": total, "page": page, "per_page": per_page}))
    except Exception as exc:
        logger.exception("Error listing sessions")
        return jsonify(_err("Failed to list sessions.", 500)), 500


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_get_session(session_id: str):
    try:
        storage = get_storage()
        if not storage:
            return jsonify(_err("Persistence disabled.", 503)), 503
        session = storage.get_session(session_id)
        if not session:
            return jsonify(_err("Session not found.", 404)), 404
        messages = storage.get_messages(session_id)
        return jsonify(_ok({"session": session, "messages": messages}))
    except Exception as exc:
        logger.exception("Error getting session")
        return jsonify(_err("Failed to get session.", 500)), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id: str):
    try:
        storage = get_storage()
        if not storage:
            return jsonify(_err("Persistence disabled.", 503)), 503
        storage.delete_session(session_id)
        return jsonify(_ok(message="Session deleted."))
    except Exception as exc:
        logger.exception("Error deleting session")
        return jsonify(_err("Failed to delete session.", 500)), 500


@app.route("/api/sessions/<session_id>/export", methods=["POST"])
def api_export_session(session_id: str):
    try:
        storage = get_storage()
        if not storage:
            return jsonify(_err("Persistence disabled.", 503)), 503
        data = storage.export_session_json(session_id)
        return jsonify(_ok(data))
    except ValueError as exc:
        return jsonify(_err(str(exc), 404)), 404
    except Exception as exc:
        logger.exception("Error exporting session")
        return jsonify(_err("Failed to export session.", 500)), 500


# ------------------------------------------------------------------
# Export (current session)
# ------------------------------------------------------------------

def _build_csv(messages):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "session_id", "role", "content", "timestamp"])
    for msg in messages:
        writer.writerow([
            msg.get("id", ""),
            msg.get("session_id", ""),
            msg.get("role", ""),
            msg.get("content", ""),
            msg.get("timestamp", ""),
        ])
    return output.getvalue()


def _build_txt(messages):
    lines = []
    for msg in messages:
        label = "You" if msg["role"] == "user" else "AI"
        lines.append(f"[{msg['timestamp'][:19]}] {label}: {msg['content']}")
    return "\n".join(lines)


@app.route("/api/export", methods=["POST"])
@rate_limit
def api_export():
    try:
        fmt = request.args.get("format", "json").lower()
        storage = get_storage()
        if not storage or not _chat_session_id:
            return jsonify(_err("No active session to export.", 400)), 400

        messages = storage.get_messages(_chat_session_id)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sid_short = (_chat_session_id or "session")[:8]

        if fmt == "json":
            payload = json.dumps(storage.export_session_json(_chat_session_id), indent=2)
            buf = io.BytesIO(payload.encode())
            return send_file(buf, mimetype="application/json",
                             as_attachment=True,
                             download_name=f"chat_{sid_short}_{ts}.json")
        elif fmt == "csv":
            payload = _build_csv(messages)
            buf = io.BytesIO(payload.encode())
            return send_file(buf, mimetype="text/csv",
                             as_attachment=True,
                             download_name=f"chat_{sid_short}_{ts}.csv")
        elif fmt == "txt":
            payload = _build_txt(messages)
            buf = io.BytesIO(payload.encode())
            return send_file(buf, mimetype="text/plain",
                             as_attachment=True,
                             download_name=f"chat_{sid_short}_{ts}.txt")
        else:
            return jsonify(_err(f"Unknown format: {fmt}. Use json, csv, or txt.", 400)), 400
    except Exception as exc:
        logger.exception("Error in /api/export")
        return jsonify(_err("Export failed.", 500)), 500


@app.route("/api/export-fragments", methods=["POST"])
@rate_limit
def api_export_fragments():
    try:
        fmt = request.args.get("format", "json").lower()
        storage = get_storage()
        if not storage:
            return jsonify(_err("Persistence disabled.", 503)), 503
        fragments = storage.get_fragments(session_id=_chat_session_id)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = json.dumps({"fragments": fragments, "exported_at": _now()}, indent=2)
        buf = io.BytesIO(payload.encode())
        return send_file(buf, mimetype="application/json",
                         as_attachment=True,
                         download_name=f"fragments_{ts}.json")
    except Exception as exc:
        logger.exception("Error exporting fragments")
        return jsonify(_err("Export failed.", 500)), 500


# ------------------------------------------------------------------
# Fragment API
# ------------------------------------------------------------------

@app.route("/api/fragments", methods=["GET"])
def api_fragments():
    try:
        search = request.args.get("search", None)
        session_id = request.args.get("session_id", None)
        limit = int(request.args.get("limit", 100))
        storage = get_storage()
        if not storage:
            return jsonify(_err("Persistence disabled.", 503)), 503
        fragments = storage.get_fragments(session_id=session_id, search=search, limit=limit)
        return jsonify(_ok({"fragments": fragments, "count": len(fragments)}))
    except Exception as exc:
        logger.exception("Error listing fragments")
        return jsonify(_err("Failed to list fragments.", 500)), 500


@app.route("/api/fragments/bulk-teach", methods=["POST"])
@rate_limit
def api_bulk_teach():
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify(_err("'texts' must be a non-empty list.", 400)), 400
        if len(texts) > 50:
            return jsonify(_err("Maximum 50 texts per bulk-teach request.", 400)), 400

        chat = get_chat()
        storage = get_storage()
        taught = []
        for text in texts:
            err = _validate_text(text, "text item", min_len=4, max_len=2048)
            if err:
                continue
            chat.teach(text.strip(), rounds=2)
            taught.append(text.strip()[:80])
            if storage and _chat_session_id:
                sig = hashlib.sha256(text.encode()).hexdigest()[:16]
                if not storage.fragment_exists(sig):
                    storage.save_fragment(text.strip(), sig, _chat_session_id)

        return jsonify(_ok({"taught_count": len(taught), "taught": taught}))
    except Exception as exc:
        logger.exception("Error in bulk-teach")
        return jsonify(_err("Bulk teach failed.", 500)), 500


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    try:
        chat = get_chat()
        base = {
            "fragments_in_memory": len(chat.field.fragments),
            "turn": chat.turn,
            "confirms": chat.confirms,
            "rejects": chat.rejects,
        }
        storage = get_storage()
        if storage:
            base.update(storage.get_metrics())
        return jsonify(_ok(base))
    except Exception as exc:
        logger.exception("Error in /api/metrics")
        return jsonify(_err("Failed to get metrics.", 500)), 500


# ------------------------------------------------------------------
# Error handlers
# ------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify(_err("Endpoint not found.", 404)), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(_err("Method not allowed.", 405)), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify(_err("Internal server error.", 500)), 500


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    port = cfg.FLASK_PORT
    debug = cfg.FLASK_DEBUG
    logger.info("Starting ResidualChat-Resonant web server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
