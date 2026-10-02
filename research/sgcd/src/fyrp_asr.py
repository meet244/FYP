"""Qwen3-ASR-0.6B wrapper for FYRP: decoding variants + three fine-tuning recipes.

Batch size 1 throughout: on MPS a padded batch of 8 measured 1.9x SLOWER than
8 sequential calls (26.6 s vs 14.2 s for 16 utts), with identical output.
"""
import math
import random
import time

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, LogitsProcessorList

REPO = "Qwen/Qwen3-ASR-0.6B-hf"
LANG = "Hindi"  # same forced-language setting as the benchmark that picked this model
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW = 200


def load(dtype=torch.float16):
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForMultimodalLM.from_pretrained(REPO, dtype=dtype).to(DEV).eval()
    return proc, model


def _inputs(proc, model, audio, prompt=None):
    return proc.apply_transcription_request(audio=audio, language=LANG, prompt=prompt).to(
        model.device, model.dtype)


@torch.no_grad()
def transcribe(proc, model, audio, prompt=None, logits_processor_fn=None):
    """Greedy decode. `logits_processor_fn(prompt_len)` -> LogitsProcessor (S6)."""
    inp = _inputs(proc, model, audio, prompt)
    p0 = inp["input_ids"].shape[1]
    kw = {}
    if logits_processor_fn is not None:
        kw["logits_processor"] = LogitsProcessorList([logits_processor_fn(p0)])
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, **kw)
    return proc.decode(out[:, p0:], return_format="transcription_only")[0].strip()


@torch.no_grad()
def nbest(proc, model, audio, k=5):
    """Beam search, k returned hypotheses with their summed AM log-prob."""
    inp = _inputs(proc, model, audio)
    p0 = inp["input_ids"].shape[1]
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, num_beams=k,
                         num_return_sequences=k, output_scores=True,
                         return_dict_in_generate=True, length_penalty=1.0)
    ts = model.compute_transition_scores(out.sequences, out.scores, out.beam_indices,
                                         normalize_logits=False)
    texts = proc.decode(out.sequences[:, p0:], return_format="transcription_only")
    res = []
    for i, t in enumerate(texts):
        s = ts[i]
        s = s[torch.isfinite(s) & (s != 0)]  # drop padding positions after EOS
        res.append(dict(text=t.strip(), am=float(s.float().sum()), n_tok=int(s.numel())))
    return res


# ------------------------------------------------------------ training ---
def make_example(proc, audio, ref):
    inp = proc.apply_transcription_request(audio=audio, language=LANG)
    tok = proc.tokenizer
    tgt = tok.encode(ref, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
    tgt = torch.tensor([tgt], dtype=inp["input_ids"].dtype)
    ids = torch.cat([inp["input_ids"], tgt], 1)
    labels = torch.cat([torch.full_like(inp["input_ids"], -100), tgt], 1)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                input_features=inp["input_features"],
                input_features_mask=inp["input_features_mask"])


class LoRALinear(torch.nn.Module):
    """y = W x + (alpha/r) * B A x ; W frozen, B initialised to 0 (starts as identity)."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.T @ self.B.T).to(x.dtype) * self.scale


def add_lora(model, r=8, alpha=16, targets=("q_proj", "v_proj")):
    """Adapters on self-attention q/v of BOTH the audio encoder and the text decoder."""
    n = 0
    for name, mod in list(model.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, torch.nn.Linear) and "self_attn" in name.split(".")[-1]:
                setattr(mod, t, LoRALinear(child, r, alpha))
                n += 1
    return n


def train(model, examples, params, lr, accum=4, epochs=1, warmup=5, seed=0,
          row_update=None, log_every=10):
    """Generic bs=1 + grad-accumulation loop. `row_update(step)` replaces the
    optimizer for the embedding-rows recipe (S7)."""
    model.train()
    rnd = random.Random(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0) if row_update is None else None
    order = [i for _ in range(epochs) for i in rnd.sample(range(len(examples)), len(examples))]
    n_upd = math.ceil(len(order) / accum)
    t0, run_loss, upd = time.time(), 0.0, 0
    for j, i in enumerate(order, 1):
        ex = {k: v.to(model.device) for k, v in examples[i].items()}
        ex["input_features"] = ex["input_features"].to(model.dtype)
        loss = model(**ex).loss / accum
        loss.backward()
        run_loss += loss.item()
        del loss, ex
        if model.device.type == "mps":
            torch.mps.empty_cache()  # varying seq lengths otherwise grow the MPS cache into swap
        if j % accum == 0 or j == len(order):
            upd += 1
            scale = min(1.0, upd / warmup)
            if row_update is not None:
                row_update(lr * scale)
            else:
                for g in opt.param_groups:
                    g["lr"] = lr * scale
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if upd % log_every == 0 or upd == n_upd:
                print(f"    upd {upd}/{n_upd}  loss={run_loss / log_every * 1:.3f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
                run_loss = 0.0
    model.eval()
    return time.time() - t0


def embedding_row_trainer(model, new_ids, betas=(0.9, 0.999), eps=1e-8):
    """Manual Adam over ONLY the new rows of the (tied) embedding matrix.

    torch.optim.Adam on the whole 151936x1024 matrix would allocate ~1.2 GB of
    moment buffers for rows we never update; here the state is n_new x 1024.
    The base weights are bf16, so the rows are updated in an fp32 master copy
    (an lr=5e-4 step is below bf16 resolution for most entries)."""
    W = model.get_input_embeddings().weight
    idx = torch.tensor(new_ids, device=W.device)
    master = W.detach()[idx].float().clone()
    m = torch.zeros_like(master)
    v = torch.zeros_like(master)
    state = {"t": 0}

    def step(lr):
        g = W.grad[idx].float()
        g = g / max(1.0, float(g.norm()))  # clip
        state["t"] += 1
        t = state["t"]
        m.mul_(betas[0]).add_(g, alpha=1 - betas[0])
        v.mul_(betas[1]).addcmul_(g, g, value=1 - betas[1])
        mh, vh = m / (1 - betas[0] ** t), v / (1 - betas[1] ** t)
        master.sub_(lr * mh / (vh.sqrt() + eps))
        with torch.no_grad():
            W[idx] = master.to(W.dtype)
        W.grad = None

    return step


def _cast_tree(x, dt):
    if torch.is_tensor(x):
        return x.to(dt) if x.is_floating_point() else x
    if isinstance(x, tuple):
        return tuple(_cast_tree(y, dt) for y in x)
    if isinstance(x, list):
        return [_cast_tree(y, dt) for y in x]
    if isinstance(x, dict):
        return {k: _cast_tree(v, dt) for k, v in x.items()}
    return x


def fp32_island(module, outer_dtype):
    """Run `module` in fp32 inside a bf16 model: fp32 master weights for the
    trainable layers, bf16 everywhere else. Returns hook handles to remove later."""
    module.float()
    h1 = module.register_forward_pre_hook(
        lambda m, a, kw: (_cast_tree(a, torch.float32), _cast_tree(kw, torch.float32)),
        with_kwargs=True)
    h2 = module.register_forward_hook(lambda m, a, out: _cast_tree(out, outer_dtype))
    return [h1, h2]
