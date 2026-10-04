"""Regression checks for model selection, durable recovery, and API failure states."""
from types import SimpleNamespace
from datetime import datetime, timedelta

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.asr.catalog import ASROptions, resolve_options
from app.asr.backends import DecodeResult
from app.asr import sgcd
from app.db import init_db, session_scope
from app.main import app
from app.models import Job, Lecture, Subject, TranscriptSpan
from app.jobs import queue
from app.rag.answer import Answer, _gather, _resolve_lecture
from app.rag.router import Route
from test_pipeline import client, stubbed, _await_job, _write_phone_style_recording


def test_best_researched_checkpoint_and_method_are_default():
    cfg = resolve_options()
    assert cfg['model_id'] == 'qwen-0.6b'
    assert cfg['backend'] == 'qwen' and cfg['method'] == 's5'
    assert (cfg['nbest_beams'], cfg['lm_weight'], cfg['length_bonus']) == (5, .2, .5)
    assert cfg['max_new_tokens'] == 512 and cfg['repetition_penalty'] == 1.0
    assert len(cfg['lm_sha256']) == 64
    assert cfg['language'] == 'hi' and not cfg['safeguard_enabled']


@pytest.mark.parametrize('options', [
    {'model_id': 'unknown'}, {'model_id': 'parakeet-rnnt-1.1b', 'language': 'hi'},
    {'model_id': 'parakeet-rnnt-1.1b', 'method': 'sgcd'}, {'method': 'untrained-fusion'},
    {'model_id': 'qwen-1.7b', 'method': 's5'}, {'model_id': 'whisper-turbo', 'method': 's5'},
])
def test_unsupported_choices_are_rejected(options):
    with pytest.raises(ValueError):
        ASROptions(**options)


def test_reprocess_preserves_frozen_options_and_model_switch_resets_language():
    original = resolve_options()
    original.update(span_target_s=24, max_new_tokens=128, model='/local/checkpoint')
    cfg = resolve_options(ASROptions(language='auto'), original)
    assert cfg['language'] == 'auto' and cfg['model'] == '/local/checkpoint'
    assert cfg['span_target_s'] == 24 and cfg['max_new_tokens'] == 128
    cfg = resolve_options(ASROptions(model_id='parakeet-rnnt-1.1b'), original)
    assert cfg['language'] == 'en' and cfg['method'] == 'baseline'


def test_completed_decode_spans_resume_and_changed_config_invalidates(monkeypatch, tmp_path):
    audio = tmp_path / 'recording.wav'
    sf.write(audio, np.ones(16000 * 55, dtype=np.float32) * .1, 16000)
    calls = []
    backend = SimpleNamespace(transcribe=lambda a, p: (calls.append(p) or DecodeResult('weighted inputs', None, None)))
    monkeypatch.setattr(sgcd, 'get_backend', lambda *a, **kw: backend)
    checkpoint = str(tmp_path / 'checkpoint.json')
    cfg = resolve_options(ASROptions(method='baseline'))
    def interrupted(frac, message):
        if message.startswith('first pass 2/'):
            raise queue.JobCancelled()
    with pytest.raises(queue.JobCancelled):
        sgcd.transcribe(str(audio), config=cfg, checkpoint_path=checkpoint, progress=interrupted)
    assert len(calls) == 1
    result = sgcd.transcribe(str(audio), config=cfg, checkpoint_path=checkpoint)
    assert len(calls) == len(result.spans) and result.stats['resumed_spans'] == 1
    sgcd.transcribe(str(audio), config={**cfg, 'language': 'auto'}, checkpoint_path=checkpoint)
    assert len(calls) == 2 * len(result.spans)


@pytest.mark.parametrize('method,conditioned,first_weight', [
    ('baseline', False, .90), ('s5', False, .90),
    ('sgcd', False, .90), ('sgcd', True, .45),
])
def test_progress_reserves_second_decode_only_when_it_runs(monkeypatch, tmp_path, method, conditioned, first_weight):
    audio = tmp_path / 'recording.wav'
    sf.write(audio, np.ones(16000 * 55, dtype=np.float32) * .1, 16000)
    calls, updates = [], []
    def decode(*args):
        calls.append(args)
        return DecodeResult('weighted inputs', None, None)
    monkeypatch.setattr(sgcd, 'get_backend', lambda *a, **kw: SimpleNamespace(
        transcribe=decode, transcribe_rescored=decode, supports_context=True))
    monkeypatch.setattr(sgcd.retrieve, 'get_index', lambda *a, **kw: SimpleNamespace(query=lambda text: []))
    units = [SimpleNamespace(id='unit', title='Inputs', prose='weighted inputs', keywords=[])] if conditioned else []
    result = sgcd.transcribe(str(audio), units=units, syllabus_id='syllabus' if conditioned else None,
                            config=resolve_options(ASROptions(method=method)),
                            progress=lambda fraction, message: updates.append((fraction, message)))
    count = len(result.spans)
    assert len(calls) == count * (2 if conditioned else 1)
    assert max(f for f, m in updates if m.startswith('first pass')) == pytest.approx(first_weight)
    assert [f for f, _ in updates] == sorted(f for f, _ in updates)
    assert updates[-1][0] == pytest.approx(.95)


def test_cancellation_during_inference_error_reaches_terminal_state(monkeypatch):
    init_db()
    with session_scope() as db:
        job = Job(kind='cancel_error', status='queued')
        db.add(job); db.flush(); jid = job.id
    def handler(job_id):
        with session_scope() as db:
            db.get(Job, job_id).status = 'cancelling'
        raise RuntimeError('inference interrupted')
    monkeypatch.setitem(queue._registry, 'cancel_error', handler)
    queue._run_one(jid)
    with session_scope() as db:
        assert db.get(Job, jid).status == 'cancelled'


def test_notes_retry_never_decodes_audio_and_runs_preserve_previous_transcript(client, stubbed, tmp_path):
    sid = client.post('/subjects', json={'name': 'Notes and runs'}).json()['id']
    path = tmp_path / 'voice.wav'
    _write_phone_style_recording(path, seconds=12)
    with path.open('rb') as f:
        response = client.post(f'/subjects/{sid}/lectures', files={'file': ('voice.wav', f, 'audio/wav')})
    first = _await_job(client, response.json()['id']); assert first['status'] == 'succeeded'
    lid = first['lecture_id']; calls = len(stubbed.calls)
    run1 = client.get(f'/lectures/{lid}/runs').json()[0]
    assert run1['config']['model_id'] == 'qwen-0.6b'
    notes = client.post(f'/lectures/{lid}/notes/regenerate'); assert notes.status_code == 202
    assert _await_job(client, notes.json()['id'])['status'] == 'succeeded'
    assert len(stubbed.calls) == calls
    reprocess = client.post(f'/lectures/{lid}/reprocess', json={'language': 'auto'})
    assert reprocess.status_code == 202
    assert _await_job(client, reprocess.json()['id'])['status'] == 'succeeded'
    runs = client.get(f'/lectures/{lid}/runs').json(); assert len(runs) == 2
    detail = client.get(f'/lectures/{lid}/runs/{run1["id"]}').json()
    assert detail['config']['language'] == 'hi' and detail['spans'][0]['text']
    assert client.get(f'/lectures/{lid}').json()['summary']


def test_audio_failure_is_visible_and_retryable(client, stubbed):
    sid = client.post('/subjects', json={'name': 'Bad audio'}).json()['id']
    r = client.post(f'/subjects/{sid}/lectures', files={'file': ('bad.wav', b'not audio', 'audio/wav')})
    assert r.status_code == 202
    job = _await_job(client, r.json()['id']); assert job['status'] == 'failed'
    assert client.get(f'/lectures/{job["lecture_id"]}').json()['status'] == 'failed'
    retry = client.post(f'/jobs/{job["id"]}/retry'); assert retry.status_code == 202
    assert retry.json()['asr_config'] == job['asr_config']
    assert _await_job(client, retry.json()['id'])['status'] == 'failed'


def test_material_batch_rolls_back_files_and_rows(client, stubbed):
    from app.config import settings
    sid = client.post('/subjects', json={'name': 'Atomic batch'}).json()['id']
    r = client.post(f'/subjects/{sid}/materials', files=[
        ('files', ('good.txt', b'weighted inputs', 'text/plain')),
        ('files', ('empty.txt', b'', 'text/plain')),
    ])
    assert r.status_code == 400
    assert client.get(f'/subjects/{sid}/materials').json() == []
    assert list((settings.materials_dir / sid).iterdir()) == []


def test_unavailable_image_ocr_fails_instead_of_indexing_filename(monkeypatch, tmp_path):
    from app.ingest.materials import extract_image
    from app.llm.client import LLMUnavailable
    def unavailable(*a):
        raise LLMUnavailable('configure a key')
    monkeypatch.setattr('app.ingest.materials.describe_image', unavailable)
    image = tmp_path / 'image.png'; image.write_bytes(b'fake')
    with pytest.raises(LLMUnavailable):
        extract_image(image, 'image/png')


def test_chat_releases_writer_before_answer_and_keeps_history(client, monkeypatch):
    sid = client.post('/subjects', json={'name': 'Chat concurrency'}).json()['id']
    seen = []
    def answer(db, subject, question, history):
        assert db is None
        # A separate writer succeeds while the answer is being computed.
        with session_scope() as other:
            other.add(Subject(name='Independent writer'))
        seen.append(history)
        return Answer('Grounded answer', 'lookup', [])
    monkeypatch.setattr('app.api.chat.answer_question', answer)
    r = client.post(f'/subjects/{sid}/chat', json={'question': 'weights?'})
    assert r.status_code == 200
    session = r.json()['session_id']
    r2 = client.post(f'/subjects/{sid}/chat', json={'question': 'bias?', 'session_id': session})
    assert r2.status_code == 200 and len(seen[1]) == 2
    assert len(client.get(f'/chat/sessions/{session}').json()) == 4


def test_explicit_lecture_scope_does_not_fall_back_to_whole_subject(monkeypatch):
    first = SimpleNamespace(id='first', title='Perceptrons', created_at=datetime(2025, 1, 1))
    last = SimpleNamespace(id='last', title='Backprop', created_at=datetime(2025, 1, 2))
    subject = SimpleNamespace(id='subject', lectures=[last, first])
    assert _resolve_lecture(None, subject, 'lecture 1').id == 'first'
    assert _resolve_lecture(None, subject, 'last lecture').id == 'last'
    monkeypatch.setattr('app.rag.answer.store.search', lambda *a, **kw: pytest.fail('must not search outside requested scope'))
    assert _gather(None, subject, Route('explain', 'weights', 'lecture 10', False)) == []


def test_qwen_adapter_passes_context_language_and_bounded_generation():
    import torch
    from app.asr.backends import QwenBackend
    class Inputs(dict):
        def to(self, device, dtype):
            assert device == 'cpu' and dtype == torch.float32
            return self
    inputs = Inputs(input_ids=torch.tensor([[10, 20, 30]]))
    seen = {}
    def prepare(**kwargs):
        seen['request'] = kwargs
        return inputs
    def generate(**kwargs):
        seen['generation'] = kwargs
        return torch.tensor([[10, 20, 30, 40, 50]])
    def decode(tokens, **kwargs):
        assert tokens.tolist() == [[40, 50]]
        assert kwargs['return_format'] == 'transcription_only'
        return ['  weighted inputs  ']
    backend = QwenBackend.__new__(QwenBackend)
    backend.language, backend.max_new_tokens, backend.repetition_penalty = 'hi', 128, 1.1
    backend.processor = SimpleNamespace(apply_transcription_request=prepare, decode=decode)
    backend.model = SimpleNamespace(device='cpu', dtype=torch.float32, generate=generate)
    result = backend.transcribe(np.ones(16000, dtype=np.float32), 'domain context', language=None)
    assert result.text == 'weighted inputs'
    assert seen['request']['language'] is None and seen['request']['prompt'] == 'domain context'
    assert seen['generation']['max_new_tokens'] == 128 and not seen['generation']['do_sample']
    assert seen['generation']['num_beams'] == seen['generation']['num_return_sequences'] == 1
    assert seen['generation']['use_cache'] and not seen['generation']['output_scores']


def test_audio_conversion_failure_does_not_leave_a_resumable_partial_wav(tmp_path):
    from app.ingest.audio import AudioError, to_wav16k_mono
    source = tmp_path / 'invalid.wav'; source.write_bytes(b'invalid audio')
    destination = tmp_path / 'normalised.wav'
    with pytest.raises(AudioError):
        to_wav16k_mono(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('normalised-*.wav'))


def test_existing_database_gets_additive_lecture_migration(monkeypatch, tmp_path):
    import app.db as database
    from sqlalchemy import create_engine, inspect, text
    legacy = create_engine(f'sqlite:///{tmp_path}/legacy.db')
    with legacy.begin() as connection:
        connection.execute(text('CREATE TABLE lectures (id VARCHAR(32) PRIMARY KEY, title VARCHAR(300))'))
        connection.execute(text("INSERT INTO lectures (id,title) VALUES ('existing','Existing recording')"))
    monkeypatch.setattr(database, 'engine', legacy)
    database.init_db(); database.init_db()
    assert {'source_path', 'asr_config', 'summary'} <= {c['name'] for c in inspect(legacy).get_columns('lectures')}
    with legacy.connect() as connection:
        assert connection.scalar(text("SELECT title FROM lectures WHERE id='existing'")) == 'Existing recording'
    legacy.dispose()


def test_conflicts_cancel_and_retry_use_the_completed_span(client, stubbed, monkeypatch, tmp_path):
    import threading, time
    started, release = threading.Event(), threading.Event()
    original = stubbed.transcribe
    def blocked(audio, prompt):
        started.set()
        assert release.wait(10), 'test must release inference'
        return original(audio, prompt)
    monkeypatch.setattr(stubbed, 'transcribe', blocked)
    sid = client.post('/subjects', json={'name': 'Cancellation API'}).json()['id']
    path = tmp_path / 'cancel.wav'; _write_phone_style_recording(path, seconds=12)
    try:
        with path.open('rb') as f:
            r = client.post(f'/subjects/{sid}/lectures', files={'file': ('cancel.wav', f, 'audio/wav')})
        job = r.json(); lid = job['lecture_id']
        assert started.wait(10)
        assert client.post(f'/lectures/{lid}/reprocess', json={}).status_code == 409
        assert client.delete(f'/lectures/{lid}').status_code == 409
        cancelled = client.post(f'/jobs/{job["id"]}/cancel')
        assert cancelled.status_code == 200 and cancelled.json()['status'] == 'cancelling'
    finally:
        release.set()
    for _ in range(40):
        status = client.get(f'/jobs/{job["id"]}').json()['status']
        if status == 'cancelled': break
        time.sleep(.05)
    assert status == 'cancelled'
    assert len(stubbed.calls) == 1
    retry = client.post(f'/jobs/{job["id"]}/retry'); assert retry.status_code == 202
    finished = _await_job(client, retry.json()['id']); assert finished['status'] == 'succeeded'
    assert len(stubbed.calls) == 1  # Saved first pass survives cooperative cancellation.
    assert client.get(f'/lectures/{lid}').json()['asr_stats']['resumed_spans'] == 1


def test_valid_lecture_scope_retrieves_only_recording_evidence(monkeypatch):
    from app.rag import store
    lecture = SimpleNamespace(id='recording', title='Perceptrons', created_at=datetime(2025, 1, 1))
    subject = SimpleNamespace(id='subject', lectures=[lecture])
    names = []
    def search(name, query, sid, **kwargs):
        assert kwargs['lecture_id'] == 'recording'
        names.append(name)
        return []
    monkeypatch.setattr('app.rag.answer.store.search', search)
    _gather(None, subject, Route('explain', 'weights', 'last lecture', False))
    assert names == [store.SPANS, store.NOTES]
