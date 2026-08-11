"""Chunk + embed the syllabus documents and extract the technical term lexicon."""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RAW = Path("syllabus/raw")
IDX = Path("syllabus/index")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Generic English words that appear in any prose document. Keeping them would inflate
# term-level recall with words that have nothing to do with the domain.
STOP = set("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could covered course delivered did
do does doing done down during each environment few for formats from further had has
have having he her here hers herself him himself his how i if in into is it its itself
lectures live me module more most my myself no nor not of off on once only or other
others ought our ours ourselves out outline over own same she should so some such than
that the their theirs them themselves then there these they this those through to too
topics under until up use used using very was we were what when where which while who
whom why will with would you your yours yourself yourselves what's it's one two three
four five six seven eight nine ten first second third are versus etc via per within
without between among across along around before after above below basic common
introduction fundamentals applications settings preferences terms vocabulary technical
english hindi spoken tutorial welcome learn learnt learning example examples work
working write writing written read reading open opening close closing new old good
best make making made take taking taken give giving given show showing shown
""".split())

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.\-]{2,}")


def chunk(text, size=120, overlap=30):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size])
            for i in range(0, max(1, len(words) - overlap), step)]


def main():
    IDX.mkdir(parents=True, exist_ok=True)
    docs = []
    for p in sorted(RAW.glob("*.md")):
        for i, c in enumerate(chunk(p.read_text(encoding="utf-8"))):
            docs.append({"doc_id": f"{p.stem}#{i}", "topic": p.stem, "text": c})

    model = SentenceTransformer(MODEL)
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(IDX / "emb.npy", emb)
    (IDX / "docs.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")

    # Term lexicon: ASCII technical tokens from the syllabus, minus generic prose words.
    # Kept in document order, not sorted: prompts truncate to the first N terms, and
    # document order puts the defining vocabulary of the topic first.
    terms, per_topic = set(), {}
    for p in sorted(RAW.glob("*.md")):
        seen, ordered = set(), []
        for tok in TOKEN.findall(p.read_text(encoding="utf-8")):
            tok = tok.lower().strip(".-_")
            if len(tok) > 2 and tok not in STOP and tok not in seen:
                seen.add(tok)
                ordered.append(tok)
        per_topic[p.stem] = ordered
        terms |= seen
    (IDX / "terms.txt").write_text("\n".join(sorted(terms)), encoding="utf-8")
    (IDX / "terms_by_topic.json").write_text(
        json.dumps(per_topic, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(docs)} chunks from {len(per_topic)} topics, {len(terms)} terms")


if __name__ == "__main__":
    main()
