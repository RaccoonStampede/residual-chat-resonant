# ResidualChat-Resonant

**Deterministic Residual Language System**  
Hudson Drift Controller · Harness · Resonant Field-Ping Synthesis

**Updated Build — 9 August 2026**  
David Hudson / Residual Stack

## Fixes applied (2026-08-09)

1. **Resonant confirm / reject recording** — `say()` now records `_last_prompt`, `_last_body`, `_last_fp` so `confirm()` and `reject()` work after resonant replies.
2. **EGGROLL rank-weight polarity** — Best fitness now receives positive weight (+1), worst receives -1.
3. **Candidate isolation** — Full state (coherent phase, void state/phase, seq buffer, kernels) is snapshotted and restored between population members.

## Quick start

```bash
pip install numpy
python test_full.py
```

```python
from residual_chat_resonant import ResidualChatResonant

chat = ResidualChatResonant(seed=137, use_eggroll=True)
chat.inject_dense(ontology_text, passes=2)
reply = chat.say("What is residual tension?")
chat.confirm()
print(chat.status())
```

## Chat interfaces

### CLI chat

Run the interactive terminal interface:

```bash
python cli_chat.py
```

Available commands:

- `/confirm` — confirm the last reply
- `/reject` — reject the last reply
- `/status` — show turns, fragments, confirms, and rejects
- `/teach <text>` — teach new text to the field
- `/quit` or `/exit` — leave the session

Example:

```text
> What is residual tension?
Speech is residual tension finding a lower energy shape in words.
> /confirm
confirmed  confirms=1
> /teach Residual anchors improve future recall.
learned 1 fragment
```

### Web chat

Install Flask if you want the browser interface:

```bash
pip install Flask
python web_chat.py
```

Then open http://localhost:5000

The web app provides:

- `GET /` — HTML chat frontend
- `POST /api/chat` — ask a question
- `POST /api/confirm` — confirm the last reply
- `POST /api/reject` — reject the last reply
- `POST /api/teach` — teach new text
- `GET /api/status` — inspect turns/fragments/confirms/rejects
- `POST /api/reset` — reset the session

The browser UI includes a scrollable message view, send box, confirm/reject/status/reset controls, live status cards, and automatic scrolling to the latest message.

## Files

| File | Purpose |
|------|----------|
| `chat_interface_utils.py` | Shared default ontology and state wrapper for CLI/web chat |
| `cli_chat.py` | Interactive command-line chat REPL |
| `hudson_drift.py` | Analytic Hudson Drift Law auditor + harness helpers |
| `residual_eggroll.py` | Low-rank residual population optimizer (polarity + isolation fixed) |
| `resonant_variants.py` | ResonantFieldV2 ranking / synthesis primitives |
| `residual_chat_resonant.py` | Integrated chat with resonant field-ping (confirm fixed) |
| `templates/index.html` | Minimal browser chat frontend |
| `test_interfaces.py` | Focused CLI/web interface checks |
| `web_chat.py` | Flask REST API + single-session web chat |
| `test_full.py` | Full verification suite |
| `test_fixes.py` | Targeted tests for the three flags |

Residual field first. Lock, imprint, harness, ping, synthesize.