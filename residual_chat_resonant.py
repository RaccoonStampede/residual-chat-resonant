#!/usr/bin/env python3
"""
ResidualChat-Resonant — Fixed version (2026-08-09)

Fixes applied:
1. Resonant say() now correctly records _last_prompt / _last_body / _last_fp
   so that inherited confirm() and reject() work after a resonant reply.
2. Optional light reinforcement of the resonant field on confirm.
3. Cleaner status reporting.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional

from residual_eggroll import ResidualEGGROLL, LanguageCore, det_vector
from resonant_variants import ResonantFieldV2, content_tokens, residual_signature, cos, tokenize, STOP

class ResidualChatResonant:
    """
    Resonant field-ping synthesis as primary reply path.
    Includes the confirm/reject recording fix.
    """

    def __init__(
        self,
        seed: int = 137,
        use_eggroll: bool = True,
        rank: int = 4,
        pop_size: int = 16,
        sigma: float = 0.05,
        egg_every: int = 4,
        resonant_dim: int = 64,
    ):
        self.seed = seed
        self.core = LanguageCore(coherent_dim=48, seed=seed)
        self.egg = None
        self.egg_every = egg_every
        self._action_count = 0

        if use_eggroll:
            self.egg = ResidualEGGROLL(
                core=self.core, rank=rank, pop_size=pop_size, sigma=sigma, lr=0.02, seed=seed
            )
            for _ in range(4):
                self.egg.core.step()
            for _ in range(6):
                self.egg.step(n_core_steps=1)

        self.field = ResonantFieldV2(dim=resonant_dim)
        self.use_resonant = True
        self.min_resonance = 0.10

        # Required for confirm / reject
        self._last_prompt = None
        self._last_body = None
        self._last_fp = None
        self._last_feedback = None
        self.confirms = 0
        self.rejects = 0
        self.turn = 0
        self.metrics_log: List[Dict] = []

    def _fp(self) -> np.ndarray:
        v = self.core.coherent_state[:32].copy()
        return v / (np.linalg.norm(v) + 1e-9)

    def _maybe_eggroll(self):
        if self.egg is None:
            return
        self._action_count += 1
        if self._action_count % self.egg_every == 0:
            self.egg.step(n_core_steps=1)

    def teach(self, text: str, rounds: int = 3, boost: float = 1.0):
        for _ in range(rounds):
            self.core.step()
        self.field.store(text)
        self._maybe_eggroll()

    def inject_dense(self, text: str, passes: int = 2):
        parts = [p.strip() for p in re.split(r"[.!?]", text) if len(tokenize(p)) >= 4]
        for _ in range(passes):
            for part in parts:
                self.teach(part, rounds=2, boost=1.3)
            self.teach(text, rounds=2, boost=1.6)
        return len(parts)

    def _rank(self, query: str) -> List[Tuple[str, float]]:
        return self.field.rank_keyterm_boost(query)

    def _synthesize(self, query: str) -> str:
        ranked = self._rank(query)
        hits = [(f, s) for f, s in ranked if s >= self.min_resonance]
        if not hits:
            return "No strong resonance yet. Teach me more."

        lead, lead_score = hits[0]
        lead = lead.rstrip(".")
        q_content = set(content_tokens(query))
        lead_content = set(content_tokens(lead))
        lead_covers = len(q_content & lead_content) / max(1, len(q_content)) if q_content else 0.0

        if lead_score >= 0.75 or lead_covers >= 0.70 or len(hits) == 1:
            return lead + "."

        # lead + one low-overlap support
        lead_toks = set(tokenize(lead))
        for f, s in hits[1:4]:
            if s < lead_score * 0.55:
                break
            ftoks = set(tokenize(f))
            overlap = len(lead_toks & ftoks) / max(1, len(ftoks))
            if overlap < 0.70:
                return f"{lead}. {f.rstrip('.')}."
        return lead + "."

    def say(self, prompt: str, verbose: bool = False) -> str:
        """
        Primary path: resonant field-ping synthesis.
        CRITICAL FIX: always record _last_* so confirm/reject work.
        """
        self.turn += 1
        self._last_feedback = None

        if self.use_resonant and len(self.field.fragments) >= 2:
            reply = self._synthesize(prompt)

            # --- FIX: record state for confirm / reject ---
            fp = self._fp()
            self._last_prompt = prompt
            self._last_body = reply
            self._last_fp = fp.copy()

            self.metrics_log.append({
                "mode": "resonant",
                "bind": 1.0,
                "residual_cov": 1.0,
                "ttr": len(set(tokenize(reply))) / max(1, len(tokenize(reply))),
            })
            self._maybe_eggroll()
            return reply

        # Fallback (should rarely be hit once ontology is injected)
        reply = "Field is still forming."
        fp = self._fp()
        self._last_prompt = prompt
        self._last_body = reply
        self._last_fp = fp.copy()
        return reply

    def confirm(self) -> str:
        if self._last_body is None or self._last_fp is None:
            return "nothing to confirm"
        if self._last_feedback is not None:
            return f"already {self._last_feedback}"

        self._last_feedback = "confirm"
        self.confirms += 1

        # Strengthen residual pathways
        if self._last_body:
            self.field.store(self._last_body)          # reinforce the resonant field
        if self.egg is not None:
            self.egg.step(n_core_steps=1)

        return f"confirmed  confirms={self.confirms}"

    def reject(self) -> str:
        if self._last_body is None or self._last_fp is None:
            return "nothing to reject"
        if self._last_feedback is not None:
            return f"already {self._last_feedback}"

        self._last_feedback = "reject"
        self.rejects += 1
        return f"rejected  rejects={self.rejects}"

    def status(self) -> str:
        frags = len(self.field.fragments)
        return (
            f"turn={self.turn} frags={frags} confirms={self.confirms} "
            f"rejects={self.rejects} egg={self.egg is not None}"
        )