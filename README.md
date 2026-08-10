# ResidualChat-Resonant

**Deterministic Residual Language System**  
Hudson Drift Controller · Harness · Resonant Field-Ping Synthesis

**Updated Build — 10 August 2026**  
David Hudson / Residual Stack

---

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt
python web_chat.py          # web UI at http://localhost:5000
# or
python cli_chat.py          # interactive terminal
```

### Docker

```bash
cp .env.example .env        # configure environment
docker compose up --build   # web UI at http://localhost:5000
```

---

## CLI Interface

```bash
python cli_chat.py
```

| Command | Description |
|---------|-------------|
| `/confirm` | Confirm last response (reinforce field) |
| `/reject` | Reject last response |
| `/teach <text>` | Teach the model a new fragment |
| `/status` | Show session status |
| `/save` | Save session to SQLite |
| `/load <id>` | Load a previous session |
| `/list` | List all saved sessions |
| `/export` | Export current session to JSON |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Web Interface

```bash
python web_chat.py
```

Opens at **http://localhost:5000** with the quantum singularity UI.

Build material PDF endpoint:
- `GET /build-materials/residual-agi` (serves local `static/build-materials/ResidualAGI_WHOLE_BUILD_COMPLETE_2026-08-10-2.pdf` when present, otherwise redirects to source link)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Ctrl+K` / `Cmd+K` | Command palette |
| `Esc` | Clear input |
| `Ctrl+L` / `Cmd+L` | Clear chat locally |
| `Ctrl+S` / `Cmd+S` | Export session as JSON |

---

## REST API

All endpoints return:
```json
{ "ok": true, "data": {}, "message": "...", "error": null, "timestamp": "..." }
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message `{"query":"..."}` |
| `POST` | `/api/confirm` | Confirm last response |
| `POST` | `/api/reject` | Reject last response |
| `POST` | `/api/teach` | Teach fragment `{"text":"..."}` |
| `GET` | `/api/status` | Session status |
| `POST` | `/api/sessions` | List sessions `{"page":1}` |
| `GET` | `/api/sessions/<id>` | Load session + messages |
| `DELETE` | `/api/sessions/<id>` | Soft-delete session |
| `POST` | `/api/sessions/<id>/export` | Export session JSON |
| `POST` | `/api/export?format=json\|csv\|txt` | Download current session |
| `POST` | `/api/export-fragments` | Download fragments as JSON |
| `GET` | `/api/fragments?search=...` | List/search fragments |
| `POST` | `/api/fragments/bulk-teach` | Teach multiple `{"texts":[...]}` |
| `GET` | `/api/metrics` | Stats (sessions, messages, fragments) |

**Rate limit:** 10 requests / minute per IP.

---

## Environment Configuration

Copy `.env.example` to `.env` and adjust:

```env
FLASK_PORT=5000
FLASK_DEBUG=False
STORAGE_PATH=./data
DATABASE_URL=sqlite:///./data/residual.db
SESSION_TIMEOUT=3600
MAX_SESSIONS=100
ENABLE_PERSISTENCE=True
RATE_LIMIT=10
RATE_LIMIT_WINDOW=60
LOG_LEVEL=INFO
```

---

## Persistent Storage

All data is stored in SQLite under `./data/residual.db`:

| Table | Contents |
|-------|----------|
| `sessions` | Session metadata (id, created/updated, stats) |
| `messages` | Each user/assistant turn with feedback |
| `fragments` | Trained text fragments with session link |
| `state_snapshots` | Periodic field snapshots for session recovery |

Logs are written to `./data/residual.log`.

---

## Files

| File | Purpose |
|------|---------|
| `residual_chat_resonant.py` | Core chat engine (confirm/reject fixed) |
| `residual_eggroll.py` | Low-rank residual population optimizer |
| `resonant_variants.py` | ResonantFieldV2 ranking / synthesis |
| `hudson_drift.py` | Hudson Drift Law auditor |
| `storage.py` | SQLite persistence layer |
| `config.py` | Environment configuration |
| `web_chat.py` | Flask web server + REST API |
| `cli_chat.py` | Interactive CLI |
| `templates/index.html` | Quantum singularity web UI |
| `Dockerfile` | Docker image |
| `docker-compose.yml` | Compose orchestration |
| `.env.example` | Environment template |
| `requirements.txt` | Python dependencies |
| `test_fixes.py` | Targeted fix verification tests |
| `test_full.py` | Full verification suite |

---

## Core Fixes (2026-08-09)

1. **Resonant confirm/reject recording** — `say()` records `_last_prompt / _last_body / _last_fp`.
2. **EGGROLL rank-weight polarity** — Best receives +1, worst -1.
3. **Candidate isolation** — Full state snapshot/restore between population members.

---

Residual field first. Lock, imprint, harness, ping, synthesize.