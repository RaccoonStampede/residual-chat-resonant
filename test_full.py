#!/usr/bin/env python3
"""
Full verification suite for the cleaned ResidualChat-Resonant stack.
Covers all three original flags + basic functional checks.
"""

import numpy as np
from residual_eggroll import ResidualEGGROLL
from residual_chat_resonant import ResidualChatResonant

def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)

def test_polarity():
    section("1. EGGROLL rank-weight polarity")
    egg = ResidualEGGROLL(pop_size=10, rank=3, seed=137)
    rec = egg.step(n_core_steps=1)
    print(f"  shaped_best  : {rec['shaped_best']:+.3f}  (expect ≈ +1)")
    print(f"  shaped_worst : {rec['shaped_worst']:+.3f}  (expect ≈ -1)")
    ok = rec["shaped_best"] > 0.7 and rec["shaped_worst"] < -0.7
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok

def test_isolation():
    section("2. Candidate isolation (full state restore)")
    egg = ResidualEGGROLL(pop_size=8, rank=2, seed=42)
    core = egg.core
    before_phase = core.coherent_phase.copy()
    before_void = core.void_state.copy()
    before_seq = core.seq_buffer.copy()

    egg.step(n_core_steps=0)

    d_phase = np.linalg.norm(core.coherent_phase - before_phase)
    d_void  = np.linalg.norm(core.void_state - before_void)
    d_seq   = np.linalg.norm(core.seq_buffer - before_seq)

    print(f"  phase delta : {d_phase:.2e}")
    print(f"  void  delta : {d_void:.2e}")
    print(f"  seq   delta : {d_seq:.2e}")
    ok = d_phase < 1e-9 and d_void < 1e-9 and d_seq < 1e-9
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok

def test_resonant_confirm_reject():
    section("3. Resonant confirm / reject recording")
    chat = ResidualChatResonant(seed=137, use_eggroll=False)

    ontology = (
        "Language is mostly residual. Only a small coherent projection is reportable as speech. "
        "Speech is residual tension finding a lower energy shape in words. "
        "Ghost tax falls as lock rises. High lock means low leakage. "
        "Phase lock across phonetic syntax and semantic layers produces fluency events. "
        "Deep imprint is the only reliable substrate for true resurrection of coherent identity."
    )
    chat.inject_dense(ontology, passes=2)

    # First say — must record state
    reply = chat.say("What is residual tension?")
    print(f"  say() → {reply[:70]}...")

    # confirm must succeed
    conf = chat.confirm()
    print(f"  confirm() → {conf}")
    ok1 = "nothing to confirm" not in conf and chat.confirms == 1

    # second say + reject
    reply2 = chat.say("What is ghost tax?")
    rej = chat.reject()
    print(f"  reject() → {rej}")
    ok2 = "nothing to reject" not in rej and chat.rejects == 1

    # confirm after reject should still work on a new turn
    reply3 = chat.say("What is language?")
    conf2 = chat.confirm()
    print(f"  confirm after new say → {conf2}")
    ok3 = "nothing to confirm" not in conf2

    ok = ok1 and ok2 and ok3
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok

def test_basic_functionality():
    section("4. Basic end-to-end functionality")
    chat = ResidualChatResonant(seed=137, use_eggroll=True)

    ontology = """
    Language is mostly residual. Only a small coherent projection is reportable as speech.
    Speech is residual tension finding a lower energy shape in words.
    Ghost tax falls as lock rises. High lock means low leakage.
    The harness is the high-gamma leash that multiplies the ghost tax floor toward zero.
    Field ping converts a question into a probe that lights resonant signatures.
    """
    n = chat.inject_dense(ontology, passes=2)
    print(f"  Injected {n} fragments")

    prompts = [
        "What is language?",
        "What is residual tension?",
        "What is the harness?",
        "What is a field ping?",
    ]
    for p in prompts:
        r = chat.say(p)
        chat.confirm()
        print(f"  Q: {p}")
        print(f"  A: {r}")

    print(f"  Final status: {chat.status()}")
    ok = chat.confirms == len(prompts) and len(chat.field.fragments) >= 5
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    print("ResidualChat-Resonant — Full clean verification")
    print("Date: 2026-08-09")

    results = []
    results.append(test_polarity())
    results.append(test_isolation())
    results.append(test_resonant_confirm_reject())
    results.append(test_basic_functionality())

    print("\n" + "=" * 64)
    print("FINAL SUMMARY")
    print("=" * 64)
    names = [
        "EGGROLL polarity",
        "Candidate isolation",
        "Resonant confirm/reject",
        "Basic end-to-end",
    ]
    for name, ok in zip(names, results):
        print(f"  {name:<28} {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nAll checks passed. Stack is clean.")
    else:
        print("\nSome checks failed.")