"""Document extraction, inline previews, and recording/document citations.

Models are stubbed; these tests exercise real uploads, workers, chunking,
source metadata and chat persistence against an isolated corpus.
"""
import io
import re
from zipfile import ZipFile

import pytest

from app.ingest.materials import extract_doc
from app.rag import store
from app.rag.answer import _render_sources
from app.schemas import Citation
from test_pipeline import client, stubbed, _await_job, _write_phone_style_recording


def _office_bytes(parts):
    output = io.BytesIO()
    with ZipFile(output, 'w') as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return output.getvalue()


def _docx():
    return _office_bytes({'word/document.xml': '''
      <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body><w:p><w:r><w:t>Round robin uses a fixed </w:t></w:r>
          <w:r><w:t>time slice.</w:t></w:r></w:p>
          <w:tbl><w:tr><w:tc><w:p><w:r><w:t>हिंदी table cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
        </w:body>
      </w:document>'''})


def _pptx():
    ns_p = 'http://schemas.openxmlformats.org/presentationml/2006/main'
    ns_r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    ns_a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    return _office_bytes({
        'ppt/presentation.xml': f'<p:presentation xmlns:p="{ns_p}" xmlns:r="{ns_r}"><p:sldIdLst><p:sldId id="256" r:id="second"/><p:sldId id="257" r:id="first"/></p:sldIdLst></p:presentation>',
        'ppt/_rels/presentation.xml.rels': '<Relationships><Relationship Id="first" Target="slides/slide1.xml"/><Relationship Id="second" Target="slides/slide2.xml"/></Relationships>',
        'ppt/slides/slide1.xml': f'<p:sld xmlns:p="{ns_p}" xmlns:a="{ns_a}"><a:p><a:r><a:t>Second displayed slide</a:t></a:r></a:p></p:sld>',
        'ppt/slides/slide2.xml': f'<p:sld xmlns:p="{ns_p}" xmlns:a="{ns_a}"><a:p><a:r><a:t>Round robin</a:t></a:r><a:r><a:t> scheduling</a:t></a:r></a:p></p:sld>',
    })


def test_word_extracts_runs_tables_and_unicode(tmp_path):
    path = tmp_path / 'syllabus.docx'
    path.write_bytes(_docx())
    text = extract_doc(path)
    assert 'fixed time slice.' in text
    assert 'हिंदी table cell' in text


def test_powerpoint_follows_presentation_order(tmp_path):
    path = tmp_path / 'slides.pptx'
    path.write_bytes(_pptx())
    text = extract_doc(path)
    assert text.startswith('[slide 1]\nRound robin scheduling')
    assert '[slide 2]\nSecond displayed slide' in text


@pytest.mark.parametrize('filename,content,expected', [
    ('syllabus.docx', _docx(), 'fixed time slice.'),
    ('slides.pptx', _pptx(), '[slide 1]'),
    ('table.csv', b'topic,description\nscheduling,time slice', 'topic,description'),
])
def test_document_upload_indexes_and_previews_without_attachment(client, stubbed, filename, content, expected):
    sid = client.post('/subjects', json={'name': 'Source previews'}).json()['id']
    response = client.post(f'/subjects/{sid}/materials', files=[('files', (filename, content))])
    assert response.status_code == 202
    queued = response.json()[0]
    job = _await_job(client, queued['id'])
    assert job['status'] == 'succeeded', job['error']
    mid = queued['material_id']
    preview = client.get(f'/materials/{mid}')
    assert preview.status_code == 200
    assert expected in preview.json()['text']
    assert preview.json()['n_chunks'] >= 1
    original = client.get(f'/materials/{mid}/file')
    assert original.content == content
    assert original.headers['content-disposition'].startswith('inline;')
    assert 'path' not in preview.json()
    assert client.get('/materials/not-found').status_code == 404


def test_pdf_citation_carries_the_passage_and_page():
    _, citations = _render_sources([{'text': '[page 3]\nRound robin uses a fixed slice.',
        'metadata': {'kind': 'pdf', 'material_id': 'slides', 'topic': 'Scheduling slides'}}])
    assert citations[0]['page'] == 3
    assert 'fixed slice' in citations[0]['excerpt']


def test_chat_uses_indexed_documents_and_transcripts(client, stubbed, monkeypatch, tmp_path):
    indexed = {}
    def upsert(name, ids, docs, metas):
        indexed.setdefault(name, []).extend(dict(id=id, text=doc, metadata=meta, score=.9)
                                         for id, doc, meta in zip(ids, docs, metas))
    monkeypatch.setattr(store, 'upsert', upsert)
    monkeypatch.setattr(store, 'search', lambda name, query, sid, **kwargs:
                        [hit for hit in indexed.get(name, []) if hit['metadata']['subject_id'] == sid])
    sid = client.post('/subjects', json={'name': 'Scheduling'}).json()['id']
    upload = client.post(f'/subjects/{sid}/materials', files=[('files', ('syllabus.docx', _docx()))])
    assert _await_job(client, upload.json()[0]['id'])['status'] == 'succeeded'
    path = tmp_path / 'lecture.wav'
    _write_phone_style_recording(path, seconds=40)
    response = client.post(f'/subjects/{sid}/lectures', files={'file': ('lecture.wav', path.read_bytes())})
    job = _await_job(client, response.json()['id'])
    assert job['status'] == 'succeeded'
    assert indexed[store.SPANS] and indexed[store.MATERIALS]
    assert indexed[store.SPANS][0]['metadata']['start_s'] == 0

    def compose(system, user, **kwargs):
        doc = re.search(r'\[(\d+)\] DOC —', user).group(1)
        transcript = re.search(r'\[(\d+)\].*— transcript @', user).group(1)
        return f'Round robin uses a fixed time slice. [{doc}] It is discussed in the recording. [{transcript}]'
    monkeypatch.setattr('app.rag.answer.complete', compose)
    answer = client.post(f'/subjects/{sid}/chat', json={'question': 'Explain round robin scheduling'})
    assert answer.status_code == 200
    body = answer.json()
    assert {cite['kind'] for cite in body['citations']} == {'doc', 'span'}
    assert all(cite['excerpt'] for cite in body['citations'])
    messages = client.get(f'/chat/sessions/{body["session_id"]}').json()
    assert [Citation.model_validate(cite).model_dump() for cite in messages[-1]['citations']] == body['citations']
