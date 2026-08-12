"""Build the two derived syllabus artefacts (§5.3) and freeze them.

(a) A retrievable chunk index: each syllabus document is split into overlapping
    passages of a fixed word length, each passage is embedded with a multilingual
    sentence-embedding model, and the vectors are stored alongside passage text and
    topic label. Multilingual embedding is required because the queries are pass-1
    transcripts containing both Devanagari and Latin script.

(b) A term lexicon: candidate technical terms extracted from the syllabus documents —
    identifier-like tokens, multi-case words, command names, library names.

The lexicon defines the metric (§8.2), so its construction procedure is fixed here and
its content hash is recorded. The stop-list below was authored from general English
function/prose vocabulary *before* any decode was inspected; it is not tuned against
observed errors. Once `syllabus/index/lexicon_manifest.json` exists, re-running this
script refuses to overwrite the lexicon unless `--refreeze` is passed, which is what
"freeze the lexicon before running any grounded condition" means operationally.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from common import ROOT, file_hash, load_config, stable_hash, write_json

# Generic prose and function-word vocabulary. Any of these appearing in a syllabus
# document is document boilerplate, not course terminology, and keeping them would
# inflate the B-WER denominator (and terminology recall) with words that have nothing
# to do with the domain.
STOP = set("""
a about above across after again against all along also am among an and any are around
around as at basic be because been before being below best between both but by can
cannot close closing come common could course covered delivered did do does doing done
down during each eight english etc every example examples few first five for formats
four from fundamentals further give given giving good had has have having he her here
hers herself him himself his how i if in into introduction is it its itself learn
learning learnt lectures live make made making me module more most my myself new nine
no nor not of off on once one only open opening or other others ought our ours
ourselves out outline over own per read reading same second seven she should show
showing shown six so some spoken such take taken taking ten terms than that the their
theirs them themselves then there these they third this those three through to too
topics tutorial two under until up us use used using versus very via vocabulary was we
welcome were what when where which while who whom why will with within without work
working would write writing written you your yours yourself yourselves technical hindi
""".split())

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.\-]{2,}")
IDENTIFIER_LIKE = re.compile(r"[_\.\-0-9]")


def categorise(surface: str) -> str:
    """Why this token counts as a technical term. Reported as lexicon composition."""
    if IDENTIFIER_LIKE.search(surface):
        return "identifier_like"      # stdio.h, Ctrl-S, printf(), version 3.3.4
    if surface != surface.lower() and surface != surface.capitalize():
        return "multi_case"           # PowerPoint, JChemPaint, GNU
    if surface[:1].isupper():
        return "capitalised"          # Impress, Thunderbird, Insert
    return "plain"                    # printf, gcc, odp


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if len(piece) < overlap and out:
            break
        out.append(" ".join(piece))
    return out


def extract_terms(text: str) -> list[tuple[str, str]]:
    """Ordered (term, category) pairs, first occurrence wins, document order kept."""
    seen, out = set(), []
    for tok in TOKEN.findall(text):
        surface = tok.strip(".-_")
        low = surface.lower()
        if len(low) < 3 or low in STOP or low in seen:
            continue
        seen.add(low)
        out.append((low, categorise(surface)))
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true",
                    help="allow overwriting an already-frozen lexicon (§5.3)")
    a = ap.parse_args()

    raw_dir = ROOT / cfg["syllabus"]["raw_dir"]
    idx = ROOT / cfg["syllabus"]["index_dir"]
    idx.mkdir(parents=True, exist_ok=True)
    size = cfg["syllabus"]["chunk_words"]
    overlap = cfg["syllabus"]["chunk_overlap"]
    model_name = cfg["syllabus"]["embed_model"]

    docs, per_topic, categories = [], {}, {}
    all_terms: set[str] = set()
    for p in sorted(raw_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        for i, c in enumerate(chunk_words(text, size, overlap)):
            docs.append({"doc_id": f"{p.stem}#{i}", "topic": p.stem, "text": c})
        pairs = extract_terms(text)
        per_topic[p.stem] = [t for t, _ in pairs]
        for t, cat in pairs:
            categories.setdefault(t, cat)
        all_terms |= {t for t, _ in pairs}

    # --- (a) chunk index -----------------------------------------------------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                       show_progress_bar=True, batch_size=16)
    np.save(idx / "emb.npy", np.asarray(emb, dtype=np.float32))
    (idx / "docs.jsonl").write_text(
        "\n".join(__import__("json").dumps(d, ensure_ascii=False) for d in docs) + "\n",
        encoding="utf-8")

    # --- (b) term lexicon ----------------------------------------------------
    lex_path = idx / "terms.txt"
    manifest_path = idx / "lexicon_manifest.json"
    if manifest_path.exists() and not a.refreeze:
        print(f"lexicon already frozen: {manifest_path.name} "
              f"({len(lex_path.read_text().split())} terms). "
              f"Chunk index rebuilt; lexicon left untouched. Use --refreeze to change.")
    else:
        lex_path.write_text("\n".join(sorted(all_terms)) + "\n", encoding="utf-8")
        write_json(idx / "terms_by_topic.json", per_topic)
        write_json(idx / "term_categories.json", categories)
        comp: dict[str, int] = {}
        for c in categories.values():
            comp[c] = comp.get(c, 0) + 1
        write_json(manifest_path, {
            "frozen": True,
            "n_terms": len(all_terms),
            "n_topics": len(per_topic),
            "terms_sha256_12": file_hash(lex_path),
            "composition": comp,
            "construction": (
                "ASCII tokens of length >= 3 from the authored syllabus documents, "
                "minus a fixed general-English prose/function-word stop-list; first "
                "occurrence per document order retained. No term was added or removed "
                "after observing model output."),
            "stoplist_size": len(STOP),
            "stoplist_sha1_12": stable_hash(sorted(STOP)),
            "source_docs": [p.name for p in sorted(raw_dir.glob("*.md"))],
            "embed_model": model_name,
            "chunk_words": size, "chunk_overlap": overlap,
        })
        print(f"lexicon FROZEN: {len(all_terms)} terms, composition={comp}")

    print(f"{len(docs)} chunks from {len(per_topic)} topics; "
          f"embeddings {np.asarray(emb).shape} -> {idx.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
