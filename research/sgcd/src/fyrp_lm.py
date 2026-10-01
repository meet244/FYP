"""Pure-Python token-level trigram LM over the Qwen3-ASR BPE vocabulary.

One asset, two consumers:
  S5  N-best rescoring  -> `seq_logprob(ids)`
  S6  shallow fusion    -> `FusionProcessor`, adds lambda * log P_LM(v | ctx) to
                           every vocab logit at each decode step.

Interpolated absolute discounting (D=0.75) down to an add-alpha unigram, so every
vocab entry has non-zero mass — required for fusion, where unseen tokens must be
penalised softly, not forbidden.
"""
import collections

import numpy as np
import torch
from transformers import LogitsProcessor

from normalize import normalize

BOS = -1  # context padding symbol (never a real id)
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


class TokenTrigramLM:
    def __init__(self, tok, texts, vocab_size, eos_id, D=0.75, alpha=0.01):
        self.tok, self.V, self.eos, self.D = tok, vocab_size, eos_id, D
        uni = np.zeros(vocab_size, dtype=np.float64)
        bi = collections.defaultdict(collections.Counter)
        tri = collections.defaultdict(collections.Counter)
        for t in texts:
            ids = [BOS, BOS] + self.encode(t) + [eos_id]
            for i in range(2, len(ids)):
                uni[ids[i]] += 1
                bi[ids[i - 1]][ids[i]] += 1
                tri[(ids[i - 2], ids[i - 1])][ids[i]] += 1
        self.p_uni = (uni + alpha) / (uni.sum() + alpha * vocab_size)
        self.log_uni = np.log(self.p_uni)
        self.bi = {k: self._pack(c) for k, c in bi.items()}
        self.tri = {k: self._pack(c) for k, c in tri.items()}
        self._cache = {}

    def encode(self, text):
        return self.tok.encode(normalize(text), add_special_tokens=False)

    @staticmethod
    def _pack(c):
        ids = np.fromiter(c.keys(), dtype=np.int64)
        cnt = np.fromiter(c.values(), dtype=np.float64)
        return ids, cnt, cnt.sum()

    def _mix(self, table, key, lower):
        """P(w|ctx) = max(c-D,0)/N + (D*types/N) * P_lower(w)."""
        if key not in table:
            return lower
        ids, cnt, n = table[key]
        p = lower * (self.D * len(ids) / n)
        p[ids] += np.maximum(cnt - self.D, 0) / n
        return p

    def dist(self, u, v):
        """Full next-token log-distribution for context (u, v). Cached."""
        key = (u, v)
        if key not in self._cache:
            p = self._mix(self.bi, v, self.p_uni.copy())
            p = self._mix(self.tri, (u, v), p)
            self._cache[key] = np.log(p)
            if len(self._cache) > 20000:
                self._cache.clear()
        return self._cache[key]

    def seq_logprob(self, text):
        ids = [BOS, BOS] + self.encode(text) + [self.eos]
        return float(sum(self.dist(ids[i - 2], ids[i - 1])[ids[i]] for i in range(2, len(ids))))

    def perplexity(self, texts):
        lp = n = 0
        for t in texts:
            lp += self.seq_logprob(t)
            n += len(self.encode(t)) + 1
        return float(np.exp(-lp / max(n, 1)))


class FusionProcessor(LogitsProcessor):
    """Shallow fusion: score = log_softmax(AM) + lam * log P_LM.

    The LM was trained on lowercased text, so each vocab id is looked up via its
    lowercase twin (`low_map`) — otherwise the LM would just fight the model's
    casing, which the scorer ignores anyway. Special tokens other than EOS get
    zero LM bonus."""

    def __init__(self, lm, prompt_len, lam, low_map, special_mask):
        self.lm, self.p0, self.lam = lm, prompt_len, lam
        self.low_map, self.special = low_map, special_mask

    def __call__(self, input_ids, scores):
        am = torch.log_softmax(scores.float(), dim=-1)
        out = []
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.p0:].tolist()
            ctx = [BOS, BOS] + [int(self.low_map[i]) for i in gen]
            lm = self.lm.dist(ctx[-2], ctx[-1])
            n = am.shape[-1]
            bonus = lm[self.low_map[:n]]
            bonus = np.where(self.special[:n], 0.0, bonus)
            bonus[self.lm.eos] = lm[self.lm.eos]
            out.append(torch.from_numpy(bonus.astype(np.float32)))
        return am + self.lam * torch.stack(out).to(am.device)


def lowercase_map(tok, n):
    """vocab id -> id of its lowercased form when that is a single token."""
    m = np.arange(n, dtype=np.int64)
    vocab = tok.get_vocab()
    conv = tok.convert_ids_to_tokens
    for tid in range(min(n, len(tok))):
        s = conv(tid)
        if s is None:
            continue
        lo = s.translate(_ASCII_LOWER)  # NOT str.lower(): byte-level chars like 'Ġ','Ã' have lowercase forms
        if lo != s and lo in vocab:
            m[tid] = vocab[lo]
    return m


def special_mask(tok, n):
    m = np.zeros(n, dtype=bool)
    for tid in tok.all_special_ids + [i for i in tok.added_tokens_decoder]:
        if tid < n:
            m[tid] = True
    m[len(tok):] = True  # padded vocab slots
    return m
