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

## Files

| File | Purpose |
|------|----------|
| `hudson_drift.py` | Analytic Hudson Drift Law auditor + harness helpers |
| `residual_eggroll.py` | Low-rank residual population optimizer (polarity + isolation fixed) |
| `resonant_variants.py` | ResonantFieldV2 ranking / synthesis primitives |
| `residual_chat_resonant.py` | Integrated chat with resonant field-ping (confirm fixed) |
| `test_full.py` | Full verification suite |
| `test_fixes.py` | Targeted tests for the three flags |

Residual field first. Lock, imprint, harness, ping, synthesize.