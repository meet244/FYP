# Mechanism implementation notes and threats to validity

Findings about the runtime that affect how §7's mechanisms should be described in the
paper. State these explicitly; anticipating them is stronger than being asked.

## 1 M1 and M2 are less mechanically distinct than §7.2 assumes

§7.2 describes token-level biasing as acting "on the decoder's search, not on its textual
context", and mechanically distinct from M1. In faster-whisper 1.1.1 / CTranslate2 4.5.0
that distinction is weaker than it sounds, and the paper must not overclaim it.

Reading `faster_whisper/transcribe.py::get_prompt`:

```python
if previous_tokens or (hotwords and not prefix):
    prompt.append(tokenizer.sot_prev)
    if hotwords and not prefix:
        hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
        ...
        prompt.extend(hotwords_tokens)
    if previous_tokens:
        prompt.extend(previous_tokens[-(self.max_length // 2 - 1):])
```

So `hotwords` are tokenised and placed in the **same `<|startofprev|>` slot** that
`initial_prompt` (M1) occupies. CTranslate2 exposes no logit-processor hook, so there is
no shallow-fusion score bonus applied during beam search — neither mechanism reweights
hypotheses directly.

Two real differences remain, and they are what M1 vs M2 actually compares:

1. **Scope of application.** `initial_prompt` conditions only the first 30-second window;
   `hotwords` are re-supplied for every window. With `condition_on_previous_text=False`
   and sentence-length utterances this rarely matters here, but it is not identical.
2. **Content and form.** M1 injects bounded natural prose containing the terms in context;
   M2 injects a bare comma-separated hint list. §7.1's own argument — that Whisper
   imitates the style of the text it is conditioned on — predicts these behave
   differently, and the prose-vs-glossary ablation (`M1_glossary`) isolates exactly that.

**How to describe it.** M1 and M2 differ in *what is injected and how often*, both through
the decoder's prompt interface, rather than in *where in the decoder* the knowledge lands.
A genuine token-level intervention (shallow fusion over the beam, or n-best rescoring with
a term-presence bonus) would require patching CTranslate2's generation loop and is
identified as future work. The comparison remains valid and informative; the mechanism
description must simply be accurate.

The `prefix` interaction that §7.2 warns about is real and is enforced rather than
trusted: `DecodeConfig.__post_init__` raises if both `hotwords` and `prefix` are set,
because the runtime would silently drop the hints.

## 2 Temperature zero disables the runtime's repetition mitigation

§4.1 fixes `temperature=0.0` so that every run is deterministic and all differences are
attributable to the intervention. A consequence worth stating: faster-whisper's usual
repetition defence is the temperature-fallback ladder, which retries a window at a higher
temperature when the compression ratio or average log-probability crosses a threshold.
With a single temperature there is no fallback, so short or near-silent windows can
produce degenerate repetition loops (observed during the smoke test on the unrefined cuts:
`... जाएंगे जाएंगे जाएंगे जाएंगे`).

This is a property of the baseline, not of any grounding mechanism, and it applies
identically to every condition, so it does not bias the comparison — but it does inflate
the absolute insertion count, and it interacts with the utterance-duration analysis in
§8.4. Deterministic alternatives exist (`no_repeat_ngram_size`, `repetition_penalty`) and
are deliberately left at the runtime's defaults so the baseline is the documented
out-of-the-box configuration; enabling them would be a separate, reportable choice.

`analyze_errors.py` reports the top insertions, so the scale of this effect is visible in
the error analysis rather than assumed.

## 3 B-WER membership is script-aware

The corpus writes English technical terms sometimes in Latin and sometimes in Devanagari
(`slide` / `स्लाइड`, `font` / `फॉन्ट`) — often both inside one lecture. Testing lexicon
membership on the level-1 surface alone would silently exclude every Devanagari-written
term from B-WER, understating both the bias-token rate and the terminology error rate.

`Lexicon.in_bias` therefore accepts either a level-1 surface match or a level-2
(romanised, phonetically folded) match, and `Lexicon.coverage` reports the same-script
figure separately so the split is visible. This is a deviation from the simplest reading
of §8.2 and is a deliberate, documented one.

## 4 Output-level correction includes spelling canonicalisation

Because scoring compares level-1 surfaces, a hypothesis token that folds onto a lexicon
term but is spelled differently (`printff` → `printf`) is still an error. M3a therefore
rewrites such tokens to the lexicon's own surface form, restricted to Latin-script tokens:
where the model wrote a term in Devanagari, the script choice is the reference's business
and level-2 scoring already treats the two as equal. Without this restriction M3a would
transliterate Devanagari terminology into Latin and damage the headline WER.

## 5 The corpus segmentation defect is the largest threat to the absolute numbers

See §2 of `01_dataset_and_harness.md`. Baseline WER on this corpus is dominated by the
quality of the utterance windows, not by the model. Refinement improves mean
per-utterance WER on the Tier-1 sample from 0.87 to 0.55, but roughly a third of
utterances still score worse than with the distributed windows, so residual boundary
error remains a component of every absolute number reported. All comparisons are made
between conditions that read *identical* audio, so the relative results — which is what
every hypothesis is about — are unaffected.
