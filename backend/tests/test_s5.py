"""Check the research score, provenance contract, and resumable S5 path."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.asr import sgcd
from app.asr.backends import DecodeResult, _beam_acoustic_scores
from app.asr.catalog import ASROptions, resolve_options
from app.asr.rescore import DEFAULT_ASSET, DomainLM, asset_info, select_candidate
from app.jobs.queue import JobCancelled


class ToyTokenizer:
    def get_vocab(self):
        return {'alpha': 0, 'beta': 1, 'unknown': 2, '<eos>': 3}

    def encode(self, text, add_special_tokens=False):
        return [self.get_vocab().get(word, 2) for word in text.split()]


@pytest.fixture
def toy_lm(tmp_path):
    tokenizer = ToyTokenizer()
    uni, bi, tri = Counter(), defaultdict(Counter), defaultdict(Counter)
    for text in ['alpha beta', 'alpha beta', 'alpha alpha']:
        tokens = [-1, -1] + tokenizer.encode(text) + [3]
        for u, v, token in zip(tokens, tokens[1:], tokens[2:]):
            uni[token] += 1
            bi[(v,)][token] += 1
            tri[(u, v)][token] += 1
    data = dict(format='fyrp-token-trigram-v1', vocab_size=4, eos_id=3,
                discount=.75, alpha=.01, unigram=uni, bigram=list(bi.items()),
                trigram=list(tri.items()), provenance={},
                tokenizer_sha256=hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True,
                                                           separators=(',', ':')).encode()).hexdigest())
    path = tmp_path / 'lm.json.gz'
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    return DomainLM(tokenizer, path), path


def test_scalar_score_matches_original_vector_interpolation(toy_lm):
    lm, _ = toy_lm
    # Independently compute the original research's whole-vocabulary distribution.
    for text in ['alpha beta', 'beta alpha', 'unknown unknown', '', 'ALPHA, beta!']:
        from app.asr.normalize import normalize
        tokens = [-1, -1] + lm.tokenizer.encode(normalize(text)) + [3]
        score = 0.0
        for u, v, token in zip(tokens, tokens[1:], tokens[2:]):
            distribution = (np.array([lm.uni.get(i, 0) for i in range(4)]) + .01) / lm.denominator
            for table, context in [(lm.bi, (v,)), (lm.tri, (u, v))]:
                if context not in table:
                    continue
                counts, total = table[context]
                distribution = distribution * (.75 * len(counts) / total)
                for i, count in counts.items():
                    distribution[i] += max(count-.75, 0) / total
            score += np.log(distribution[token])
        assert lm.seq_logprob(text) == pytest.approx(score, abs=1e-12)


def test_lm_can_change_winner_and_does_not_stack_methods(toy_lm):
    lm, _ = toy_lm
    candidates = [{'text': 'unknown unknown', 'am': -1}, {'text': 'alpha beta', 'am': -2}]
    assert select_candidate(candidates, lm, weight=0)['text'] == 'unknown unknown'
    assert select_candidate(candidates, lm)['text'] == 'alpha beta'
    with pytest.raises(RuntimeError, match='no hypotheses'):
        select_candidate([], lm)


def test_changed_asset_and_tokenizer_are_rejected(toy_lm):
    lm, path = toy_lm
    with pytest.raises(RuntimeError, match='changed'):
        DomainLM(lm.tokenizer, path, expected_sha256='wrong')
    tokenizer = ToyTokenizer()
    tokenizer.get_vocab = lambda: {'incompatible': 0}
    with pytest.raises(RuntimeError, match='tokenizer'):
        DomainLM(tokenizer, path)


def test_shipped_asset_has_disjoint_sources_and_frozen_settings():
    with gzip.open(DEFAULT_ASSET, 'rt') as source:
        data = json.load(source)
    provenance = data['provenance']
    assert provenance['corpus_segments'] == 299
    assert not set(provenance['dev_lecture_ids']) & set(provenance['test_lecture_ids'])
    assert (provenance['beams'], provenance['lm_weight'], provenance['length_bonus']) == (5, .2, .5)
    assert asset_info()['lm_sha256'] == resolve_options()['lm_sha256']


def test_s5_checkpoint_resume_retains_top_beam_without_greedy_pass(monkeypatch, tmp_path):
    path = tmp_path / 'audio.wav'
    sf.write(path, np.ones(16000 * 55, dtype=np.float32) * .1, 16000)
    calls = []
    def decode(audio, config):
        calls.append(config['method'])
        return DecodeResult('rescored transcript', None, None, 'unrescored top beam')
    backend = SimpleNamespace(transcribe_rescored=decode)  # no greedy/prompt method
    monkeypatch.setattr(sgcd, 'get_backend', lambda *a, **kw: backend)
    checkpoint = str(tmp_path / 'checkpoint.json')
    def cancel(frac, message):
        if message.startswith('first pass 2/'):
            raise JobCancelled()
    with pytest.raises(JobCancelled):
        sgcd.transcribe(str(path), config=resolve_options(), checkpoint_path=checkpoint, progress=cancel)
    result = sgcd.transcribe(str(path), config=resolve_options(), checkpoint_path=checkpoint)
    assert len(calls) == len(result.spans)
    assert result.stats['resumed_spans'] == 1 and result.stats['rescored']
    assert all(s.text == 'rescored transcript' and s.baseline_text == 'unrescored top beam' for s in result.spans)
    # Reprocess older runs with their saved standard settings, not the new default.
    assert resolve_options(previous={'model_id': 'qwen-0.6b', 'method': 'baseline'})['method'] == 'baseline'
    assert resolve_options(ASROptions(model_id='whisper-turbo'))['method'] == 'baseline'


@pytest.mark.parametrize('length_penalty', [0.0, 1.0, 2.0])
def test_beam_acoustic_scores_match_transformers_token_scores(length_penalty):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    # A real, tiny decoder exercises ancestry and normalization without weights
    # downloads or an external service.
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=16,
                                      n_embd=8, n_layer=1, n_head=1,
                                      bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
    with torch.inference_mode():
        output = model.generate(torch.tensor([[1, 3, 4]]), max_new_tokens=4,
                                num_beams=5, num_return_sequences=5,
                                length_penalty=length_penalty, output_scores=True,
                                return_dict_in_generate=True)
    expected = model.compute_transition_scores(output.sequences, output.scores,
                                                output.beam_indices).sum(dim=1)
    torch.testing.assert_close(_beam_acoustic_scores(output, length_penalty=length_penalty), expected)


def test_short_beams_ignore_prompt_padding_and_unused_ancestry():
    import torch

    # Reproduce the failing dimensions: only two decode steps after a long
    # audio prompt. Different beams can finish at EOS after one or two tokens.
    ancestry = torch.zeros((5, 269), dtype=torch.int32)
    ancestry[:, :2] = torch.tensor([[0, -1], [0, 1], [0, 2], [0, -1], [0, 4]])
    output = SimpleNamespace(beam_indices=ancestry, scores=(None, None),
                             sequences_scores=torch.tensor([-1., -2., -3., -4., -5.]))
    torch.testing.assert_close(_beam_acoustic_scores(output, length_penalty=1.0),
                               torch.tensor([-1., -4., -6., -4., -10.]))


@pytest.mark.parametrize('invalid', ['empty', 'nonfinite'])
def test_invalid_beam_scores_are_rejected(invalid):
    import torch

    output = SimpleNamespace(beam_indices=torch.tensor([[-1 if invalid == 'empty' else 0]]),
                             scores=(None,), sequences_scores=torch.tensor([float('-inf') if invalid == 'nonfinite' else -1.]))
    with pytest.raises(RuntimeError, match='invalid beam scores'):
        _beam_acoustic_scores(output, length_penalty=1.0)
