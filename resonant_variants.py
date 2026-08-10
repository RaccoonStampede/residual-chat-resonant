#!/usr/bin/env python3
"""
Resonant synthesis — tightened ranking + synthesis primitives
Cleaned and slightly hardened version.
"""

import numpy as np
import re
from typing import List, Dict, Tuple
from collections import defaultdict

STOP = set(
    "a an the of to in for on with is are was were be been being it this that these those "
    "and or but if as at by from into over after before about".split()
)

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())

def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]

def residual_signature(text: str, dim: int = 64, base: float = 11.0) -> np.ndarray:
    toks = tokenize(text)
    if not toks:
        return np.zeros(dim)
    acc = np.zeros(dim)
    for i, tok in enumerate(toks):
        h = sum(ord(c) * (i + 1) for c in tok) * 0.137 + len(tok) * 1.618
        weight = 1.4 if tok not in STOP else 0.35
        # simple deterministic drive
        phase = base + h + i * 0.09
        drive = np.array([
            np.sin(phase + j * 0.173) * 0.1 * weight
            for j in range(dim)
        ])
        acc += drive
    n = np.linalg.norm(acc) + 1e-12
    return acc / n

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

class ResonantFieldV2:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.fragments: List[str] = []
        self.sigs: List[np.ndarray] = []
        self.tok_sets: List[set] = []
        self.content_sets: List[set] = []

    def store(self, text: str):
        text = text.strip()
        if not text or len(text) < 8:
            return
        for p in re.split(r"[.!?]+", text):
            p = p.strip()
            if len(p) < 10:
                continue
            if p in self.fragments:
                continue
            self.fragments.append(p)
            self.sigs.append(residual_signature(p, self.dim))
            self.tok_sets.append(set(tokenize(p)))
            self.content_sets.append(set(content_tokens(p)))

    def store_many(self, texts: List[str]):
        for t in texts:
            self.store(t)

    def stats(self) -> Dict:
        return {"fragments": len(self.fragments)}

    def rank_keyterm_boost(self, query: str) -> List[Tuple[str, float]]:
        probe = residual_signature(query, self.dim)
        q = content_tokens(query)
        qset = set(q)
        scores = []
        for i in range(len(self.sigs)):
            r = cos(probe, self.sigs[i])
            cset = self.content_sets[i]
            hits = sum(1 for t in q if t in cset)
            coverage = hits / max(1, len(qset))
            score = 0.40 * r + 0.60 * coverage
            if hits >= 2:
                score += 0.15
            if hits >= 3:
                score += 0.10
            scores.append((self.fragments[i], float(score)))
        scores.sort(key=lambda x: -x[1])
        return scores

    def synth_lead_plus_one(self, ranked: List[Tuple[str, float]], min_score: float = 0.12) -> str:
        hits = [(f, s) for f, s in ranked if s >= min_score][:3]
        if not hits:
            return "No strong resonance."
        lead = hits[0][0].rstrip(".")
        if len(hits) == 1:
            return lead + "."
        lead_toks = set(tokenize(lead))
        for f, s in hits[1:]:
            if len(set(tokenize(f)) & lead_toks) / max(1, len(set(tokenize(f)))) < 0.75:
                return f"{lead}. {f.rstrip('.')}."
        return lead + "."