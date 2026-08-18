"""
POS probing of frozen CPT encoders on Google Colab.

Supplement to the main CPT + finetuning study. The finetuning results are
noisy because generation stacks three noise sources on top of the encoder:
the decoder's own parameters, beam search, and surface-overlap metrics.
This strips all three off and asks one narrow question:

    is there a dose-response from CPT on the encoder?

Method (see notes/probing_methodology.md for the full write-up):
  frozen encoder -> pooled word vectors -> linear classifier -> accuracy/F1
  every checkpoint x every encoder layer x {first, last, mean} pooling.

Step 0 (the un-adapted base model) is the zero-dose reference. There is no
random-init control and no control task: those answer a different question.

Each "# %% CELL n" marker below is one Colab cell - paste them in order.
Run CELL 9 (the gate) and read its output BEFORE starting the sweep.

RAM: the sweep holds every layer's pooled features at once in fp32, ~7GB for
t5 and ~16GB for the byte models. Use a high-RAM runtime; if that is not
available, set FEATURE_DTYPE to bfloat16 or raise LAYER_STRIDE in CELL 3.
Never float16 - T5 activations overflow it. See CELL 3.
"""

# %% CELL 1 - install (same pins as the alignment notebook)
#
# datasets<4 matters: Colab ships 4.x, which removed script-based loaders.
# The HPC pyproject pins datasets>=3.0.0,<4.0.0, which is why it works there.
#
# !pip install -q "datasets>=3.0.0,<4.0.0" "transformers>=5.12.1" "huggingface_hub>=0.24" scikit-learn matplotlib
#
# THEN: Runtime > Restart session, and continue from CELL 2.

# %% CELL 2 - (only needed for the PRIVATE nguni repo; skip for t5/byt5)
# from huggingface_hub import notebook_login
# notebook_login()

# %% CELL 3 - imports and configuration
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# base model on the Hub (step 0), backup repo holding the CPT checkpoints,
# and whether the tokenizer is raw-byte (changes the word alignment path)
MODELS = {
    "t5":         ("google-t5/t5-large",
                   "ChonkeyJellyfish/cpt-xhosa-t5-large", False),
    "byt5":       ("google/byt5-large",
                   "ChonkeyJellyfish/cpt-xhosa-byt5-large", True),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "ChonkeyJellyfish/cpt-xhosa-nguni-byt5-large", True),  # PRIVATE
}
ALL_STEPS = [0] + list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))

RUN_MODELS = ["t5", "byt5"]        # add "nguni-byt5" after notebook_login()
POOLINGS = ["last", "first", "mean"]   # "last" is the headline (Dang et al.)
LAYER_STRIDE = 1                   # 1 = every layer. Raise to 2 if RAM is tight.

# Probe hyperparameters - Muller-Eberstein Appendix B.2, verbatim.
# weight_decay is 0 in the paper and stays 0 here: early stopping is the
# guard. See notes/probing_methodology.md section 5.
LR = 1e-3
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
BATCH = 64
MAX_EPOCHS = 30
PATIENCE = 3
SEED = 0

# Minimum test support for a tag to count toward the headline macro-F1.
# MasakhaPOS xho test has SCONJ(28), AUX(6), INTJ(2), SYM(1) - 37 tokens over
# 4 classes. Each is 1/15 of a 15-class macro average, so those four alone can
# swing macro-F1 by 0.27 on a couple of flipped predictions, which is an order
# of magnitude larger than the CPT effect we are trying to measure. At >= 30
# we keep 11 classes covering 99.6% of test tokens.
MIN_SUPPORT = 30

ENCODE_BATCH = 16                  # drop to 8 if the byte models OOM on GPU
DTYPE = torch.float32              # matches the HPC runs; do not change

# Stored feature precision. NEVER float16: T5-family activations grow
# monotonically through the encoder and blow past fp16's 65504 ceiling in the
# deeper layers (t5-small already reaches 3571 by layer 5 with only 6 layers).
# Overflow -> inf -> NaN after standardising -> the probe silently fails.
# This is the same reason T5 is trained in bf16 and breaks in fp16.
#   float32  = safe, ~7GB for t5 / ~16GB for the byte models (high-RAM runtime)
#   bfloat16 = safe range, half the memory, ~3 significant digits
FEATURE_DTYPE = torch.float32
OUT = Path("/content/probing")
CKPT_DIR = Path("/content/_ckpt")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# %% CELL 4 - load MasakhaPOS (isiXhosa)
from datasets import load_dataset

# masakhapos may be served as parquet or via a loading script depending on
# the datasets version, so try the plain call before the script path.
try:
    raw = load_dataset("masakhane/masakhapos", "xho")
except Exception:
    raw = load_dataset("masakhane/masakhapos", "xho", trust_remote_code=True)

# upos is a ClassLabel sequence on the script loader and may be plain
# strings if it ever gets served as parquet - handle both.
_feat = raw["train"].features["upos"]
if hasattr(_feat, "feature") and hasattr(_feat.feature, "names"):
    LABELS = list(_feat.feature.names)
    _encode_tags = lambda tags: list(tags)
else:
    LABELS = sorted({t for r in raw["train"] for t in r["upos"]})
    _lut = {n: i for i, n in enumerate(LABELS)}
    _encode_tags = lambda tags: [_lut[t] for t in tags]

N_LABELS = len(LABELS)
SPLITS = ["train", "validation", "test"]
DATA = {s: [(r["tokens"], _encode_tags(r["upos"])) for r in raw[s] if r["tokens"]]
        for s in SPLITS}

for s in SPLITS:
    n_w = sum(len(w) for w, _ in DATA[s])
    print(f"{s:11s} {len(DATA[s]):>5} sentences  {n_w:>6} words")
print(f"\n{N_LABELS} tags: {', '.join(LABELS)}")

# majority-class baseline on test - the number the probe has to beat
_test_tags = Counter(t for _, labs in DATA["test"] for t in labs)
MAJORITY = _test_tags.most_common(1)[0][1] / sum(_test_tags.values())
print(f"\nmajority baseline (test): {MAJORITY:.2%} "
      f"({LABELS[_test_tags.most_common(1)[0][0]]})")

# sanity on sequence length: byte models need max_len above this
_byte_lens = [len(" ".join(w).encode("utf-8")) for w, _ in DATA["train"]]
print(f"train sentence bytes: median {sorted(_byte_lens)[len(_byte_lens)//2]}, "
      f"max {max(_byte_lens)}")

# %% CELL 4b - how much of POS is just lexical memorisation? (instant, CPU)
#
# POS is largely a property of the word type, not the context: know the word
# is "umntu" and you know it is a NOUN. A linear probe on frozen features can
# get most of POS by memorising word identity -> tag. That identity lives in
# the EMBEDDING layer, which we measured drifting only 0.0001% over 10k CPT
# steps. If the memorisation ceiling is close to the probe's score, a flat
# dose-response curve is expected and says little about what CPT did.
_tr_lex = {}
for _ws, _ts in DATA["train"]:
    for _w, _t in zip(_ws, _ts):
        _tr_lex.setdefault(_w.lower(), Counter())[_t] += 1
LEX_TAG = {w: c.most_common(1)[0][0] for w, c in _tr_lex.items()}
TRAIN_VOCAB = set(LEX_TAG)

_seen = _unseen = _lex_ok = 0
for _ws, _ts in DATA["test"]:
    for _w, _t in zip(_ws, _ts):
        if _w.lower() in LEX_TAG:
            _seen += 1
            _lex_ok += (LEX_TAG[_w.lower()] == _t)
        else:
            _unseen += 1
_tot = _seen + _unseen
print(f"test word tokens          : {_tot}")
print(f"  seen in probe-train     : {_seen:>5} ({_seen / _tot:.1%})")
print(f"  UNSEEN (unmemorisable)  : {_unseen:>5} ({_unseen / _tot:.1%})")
print(f"\nmost-frequent-tag lookup, NO MODEL AT ALL:")
print(f"  accuracy on seen words  : {_lex_ok / _seen:.1%}")
print(f"  accuracy overall        : {_lex_ok / _tot:.1%}   <- the memorisation ceiling")
print(f"\nCompare that to the probe's score. The gap is what the encoder adds,")
print(f"and the unseen {_unseen / _tot:.0%} of tokens is where CPT could actually show up.")

# %% CELL 5 - word-to-token alignment
#
# THE highest-risk part of this notebook. Labels are per word, models emit
# tokens; a silent off-by-one here produces plausible curves from garbage.
# CELL 9 exists to catch exactly that before the sweep burns GPU hours.


def encode_sentence(tokenizer, words, is_byte, max_len):
    """-> (input_ids, [(word_index, tok_start, tok_end)]), end exclusive."""
    text = " ".join(words)

    if is_byte:
        # ByT5 vocab: 0=pad, 1=eos, 2=unk, then byte b -> id b+3, and no BOS.
        # So token index == byte index of `text`. Exact, no heuristics.
        ids = tokenizer(text, add_special_tokens=True).input_ids
        spans, cpos = [], 0
        for w in words:
            start = len(text[:cpos].encode("utf-8"))
            spans.append((start, start + len(w.encode("utf-8"))))
            cpos += len(w) + 1                      # +1 for the joining space
    else:
        # SentencePiece: take every token whose character span overlaps the
        # word. Strict inequalities matter - T5's Metaspace pretokenizer folds
        # the preceding space into a token, so "_uyahamba" spans chars 5-14
        # beside a word ending at 5; `a < ce` keeps that out of the earlier word.
        enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        spans, cpos = [], 0
        for w in words:
            cs, ce = cpos, cpos + len(w)
            idx = [i for i, (a, b) in enumerate(offsets)
                   if b > a and a < ce and b > cs]   # b > a drops specials
            spans.append((min(idx), max(idx) + 1) if idx else None)
            cpos += len(w) + 1

    ids = ids[:max_len]
    kept = [(wi, s, e) for wi, sp in enumerate(spans) if sp
            for s, e in [sp] if s < e <= len(ids)]   # drop, never mislabel
    return ids, kept


def collate(batch_ids, pad_id):
    width = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), width), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return ids, mask


# %% CELL 6 - feature extraction (one forward pass, every layer, 3 poolings)


@torch.no_grad()
def extract(encoder, tokenizer, data, is_byte, max_len, device):
    """-> (feats[layer][pooling] -> tensor, labels, words, n_dropped_words)

    `words` are the surface strings in the same order as the rows, so the
    caller can split test words into seen/unseen against the probe's training
    vocabulary - the split that separates memorisation from contextual encoding.
    """
    pad_id = tokenizer.pad_token_id
    encoded = [encode_sentence(tokenizer, w, is_byte, max_len) for w, _ in data]
    n_dropped = sum(len(w) - len(k) for (w, _), (_, k) in zip(data, encoded))

    feats, labels, words = None, [], []
    for start in range(0, len(data), ENCODE_BATCH):
        chunk = range(start, min(start + ENCODE_BATCH, len(data)))
        ids, mask = collate([encoded[i][0] for i in chunk], pad_id)
        out = encoder(input_ids=ids.to(device), attention_mask=mask.to(device),
                      output_hidden_states=True)

        rows, st, en = [], [], []
        for row, i in enumerate(chunk):
            for wi, s, e in encoded[i][1]:
                rows.append(row); st.append(s); en.append(e)
                labels.append(data[i][1][wi])
                words.append(data[i][0][wi])
        if not rows:
            continue
        rows = torch.tensor(rows, device=device)
        st = torch.tensor(st, device=device)
        en = torch.tensor(en, device=device)
        width = (en - st).unsqueeze(1).float()

        if feats is None:
            keep = list(range(0, len(out.hidden_states), LAYER_STRIDE))
            if keep[-1] != len(out.hidden_states) - 1:
                keep.append(len(out.hidden_states) - 1)   # always keep the top
            feats = {l: {p: [] for p in POOLINGS} for l in keep}

        for l in feats:
            h = out.hidden_states[l].float()
            # zero-prefixed cumsum turns "mean over [s, e)" into two gathers
            pad = torch.zeros(h.shape[0], 1, h.shape[2], device=device)
            cs = torch.cat([pad, h.cumsum(1)], dim=1)
            pooled = {"first": h[rows, st],
                      "last":  h[rows, en - 1],
                      "mean":  (cs[rows, en] - cs[rows, st]) / width}
            for p in POOLINGS:
                v = pooled[p].to(FEATURE_DTYPE)
                if not torch.isfinite(v).all():
                    raise RuntimeError(
                        f"non-finite features at layer {l}, pooling {p}. "
                        f"FEATURE_DTYPE={FEATURE_DTYPE} cannot hold this "
                        f"layer's activations (max |h| = {h.abs().max():.1f}).")
                feats[l][p].append(v.cpu())

    feats = {l: {p: torch.cat(v) for p, v in d.items()} for l, d in feats.items()}
    return feats, torch.tensor(labels, dtype=torch.long), words, n_dropped


# %% CELL 7 - the probe (Muller-Eberstein B.2 hyperparameters)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def train_probe(Xtr, ytr, Xdev, ydev, Xte, yte, device, seed=SEED,
                standardise=True, seen_mask=None):
    torch.manual_seed(seed)
    Xtr, Xdev, Xte = (v.float().to(device) for v in (Xtr, Xdev, Xte))
    ytr, ydev = ytr.to(device), ydev.to(device)

    # Per-dimension z-scoring. Not in Muller-Eberstein; added because raw
    # hidden-state scales vary hugely by layer and a single Adam LR has to
    # work for all of them. CELL 9b A/B-tests whether it actually matters.
    if standardise:
        mu = Xtr.mean(0, keepdim=True)               # train statistics only
        sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)   # divide-by-zero guard
        Xtr, Xdev, Xte = (Xtr - mu) / sd, (Xdev - mu) / sd, (Xte - mu) / sd

    probe = torch.nn.Linear(Xtr.shape[1], N_LABELS).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=LR, betas=BETAS,
                           weight_decay=WEIGHT_DECAY)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    n = Xtr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(probe(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        with torch.no_grad():
            dev_loss = F.cross_entropy(probe(Xdev), ydev).item()
        if dev_loss < best_loss - 1e-6:
            best_loss, best_epoch, bad = dev_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(
            "probe never improved on the dev split - dev_loss was almost "
            "certainly NaN. Check FEATURE_DTYPE (fp16 overflows on T5) and "
            "that the features are finite.")
    probe.load_state_dict(best_state)
    with torch.no_grad():
        pred = probe(Xte).argmax(1).cpu().numpy()
        train_acc = (probe(Xtr).argmax(1) == ytr).float().mean().item()

    gold = yte.numpy()
    ids = list(range(N_LABELS))
    _, _, per_f1, support = precision_recall_fscore_support(
        gold, pred, labels=ids, zero_division=0)
    # Headline macro-F1 averages ONLY over tags present in the test set.
    # MasakhaPOS xho ships tags with zero test support (PART, X, "_"), and
    # including them caps macro-F1 at n_present/N_LABELS no matter how good
    # the probe is. macro_f1_all_classes keeps the undiluted number.
    present = [i for i in ids if support[i] > 0]
    supported = [i for i in ids if support[i] >= MIN_SUPPORT]

    # Accuracy split by whether the test word was in the probe's training
    # vocabulary. Seen words can be answered by memorising word identity, which
    # lives in the near-frozen embedding layer; UNSEEN words cannot, so their
    # accuracy isolates contextual encoding - where CPT can actually show up.
    seen_acc = unseen_acc = None
    if seen_mask is not None:
        m = seen_mask.numpy()
        if m.any():
            seen_acc = float((pred[m] == gold[m]).mean())
        if (~m).any():
            unseen_acc = float((pred[~m] == gold[~m]).mean())

    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "accuracy_seen": seen_acc,
        "accuracy_unseen": unseen_acc,
        # HEADLINE: macro over tags with enough test support to be stable.
        "macro_f1": float(f1_score(gold, pred, labels=supported, average="macro",
                                   zero_division=0)),
        # macro over every tag that occurs at all - dominated by rare-class noise
        "macro_f1_present": float(f1_score(gold, pred, labels=present,
                                           average="macro", zero_division=0)),
        # macro over the full label set, incl. tags with zero test support
        "macro_f1_all_classes": float(f1_score(gold, pred, labels=ids,
                                               average="macro", zero_division=0)),
        "n_classes_scored": len(supported),
        "n_classes_present": len(present),
        "train_accuracy": train_acc,          # logged so overfitting is visible
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "per_class": {LABELS[i]: {"f1": float(per_f1[i]), "support": int(support[i])}
                      for i in ids},
    }


# %% CELL 8 - checkpoint fetch / release / resume (same pattern as alignment)


def fetch_checkpoint(model_name, step):
    base_id, repo_id, _ = MODELS[model_name]
    if step == 0:
        return base_id
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    snapshot_download(repo_id=repo_id, allow_patterns=[f"checkpoint-{step}/*"],
                      local_dir=str(CKPT_DIR))
    return str(CKPT_DIR / f"checkpoint-{step}")


def release():
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def done_steps(path):
    if not path.exists():
        return set()
    return {json.loads(l)["step"] for l in path.read_text().split("\n") if l.strip()}


def load_encoder(path):
    model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=DTYPE).to(DEVICE)
    return model, model.get_encoder().eval()


def max_len_for(is_byte):
    return 1024 if is_byte else 512


# %% CELL 9 - THE GATE. Run this and read it before starting the sweep.
#
# Probes the un-adapted base model at the final layer with last-token
# pooling. If accuracy does not clearly beat the majority baseline, the
# word alignment in CELL 5 is wrong and the sweep must not run.
GATE_MODEL = "t5"

_base_id, _, _is_byte = MODELS[GATE_MODEL]
_tok = AutoTokenizer.from_pretrained(_base_id)
print(f"{GATE_MODEL}: tokenizer fast={_tok.is_fast} (byte models are never fast)")
assert _is_byte or _tok.is_fast, "T5 needs the fast tokenizer for offset mapping"

_model, _enc = load_encoder(_base_id)
_ml = max_len_for(_is_byte)
_f, _y, _drop = {}, {}, {}
for _s in SPLITS:
    _f[_s], _y[_s], _w, _drop[_s] = extract(_enc, _tok, DATA[_s], _is_byte, _ml, DEVICE)
_top = max(_f["train"])

_res = train_probe(_f["train"][_top]["last"], _y["train"],
                   _f["validation"][_top]["last"], _y["validation"],
                   _f["test"][_top]["last"], _y["test"], DEVICE)

print(f"\nlayer {_top}, last-token pooling, step 0")
print(f"  majority baseline  {MAJORITY:.2%}")
print(f"  probe accuracy     {_res['accuracy']:.2%}   <- must clearly beat it")
print(f"  macro F1           {_res['macro_f1']:.2%}   "
      f"(over {_res['n_classes_scored']} tags with support >= {MIN_SUPPORT}; "
      f"{_res['macro_f1_present']:.2%} over all {_res['n_classes_present']} present)")
print(f"  train accuracy     {_res['train_accuracy']:.2%}   "
      f"(gap {_res['train_accuracy'] - _res['accuracy']:+.2%})")
print(f"  stopped at epoch   {_res['epochs_run']} (best {_res['best_epoch']}"
      f"{', HIT THE CAP - raise MAX_EPOCHS' if _res['epochs_run'] == MAX_EPOCHS else ''})")
print(f"  words dropped to truncation: "
      + ", ".join(f"{s} {_drop[s]}" for s in SPLITS) + "   <- should be ~0")
print("\n  per-class F1 (support):")
for _name, _d in sorted(_res["per_class"].items(), key=lambda kv: -kv[1]["support"]):
    print(f"    {_name:6s} {_d['f1']:.3f}  n={_d['support']}")

del _model, _enc, _f
release()

# %% CELL 9b - standardisation A/B. ~5 min (two checkpoint downloads).
#
# Settles whether the z-scoring in train_probe changes the answer. Two things
# to read: epochs_run (if raw hits MAX_EPOCHS while standardised stops early,
# raw was still improving and its accuracy is a floor, not a measurement) and
# delta (accuracy at 10k minus at 0 - the actual dose-response signal). If the
# deltas agree, the choice does not matter; pick one and footnote it.
AB_MODEL, AB_STEPS = "t5", [0, 10000]

_bid, _, _byte = MODELS[AB_MODEL]
_tk = AutoTokenizer.from_pretrained(_bid)
_ml = max_len_for(_byte)
ab, _keep = {}, {}

for _st in AB_STEPS:
    _m, _e = load_encoder(fetch_checkpoint(AB_MODEL, _st))
    _fe, _la = {}, {}
    for _s in SPLITS:
        _fe[_s], _la[_s], _, _ = extract(_e, _tk, DATA[_s], _byte, _ml, DEVICE)
    del _m, _e
    release()

    _ls = sorted(_fe["train"])
    for _l in [_ls[0], _ls[len(_ls) // 2], _ls[-1]]:
        for _std in (True, False):
            ab[(_st, _l, _std)] = train_probe(
                _fe["train"][_l]["last"], _la["train"],
                _fe["validation"][_l]["last"], _la["validation"],
                _fe["test"][_l]["last"], _la["test"],
                DEVICE, standardise=_std)
    _keep[_st] = {l: _fe["test"][l]["last"].clone() for l in
                  [_ls[0], _ls[len(_ls) // 2], _ls[-1]]}
    del _fe
    release()

# Did the features actually change between the two checkpoints? If a layer
# shows identical accuracy for both variants, check here before believing it.
print("\nfeature drift between the two checkpoints (test split, last pooling):")
for _l in sorted(_keep[AB_STEPS[0]]):
    _a = _keep[AB_STEPS[0]][_l].float()
    _b = _keep[AB_STEPS[1]][_l].float()
    print(f"  layer {_l:>3}  mean|delta| = {(_a - _b).abs().mean():.6f}   "
          f"mean|value| = {_a.abs().mean():.4f}   "
          f"relative = {(_a - _b).abs().mean() / _a.abs().mean().clamp(min=1e-9):.4%}")

# Macro-F1 is shown alongside accuracy because accuracy has little headroom
# here: NOUN+VERB+PROPN+PUNCT are 79% of test tokens and are largely readable
# off surface cues, so accuracy compresses the dose-response signal.
print(f"\n{'layer':>6} {'std?':>6} {'acc@0':>8} {'acc@10k':>9} {'dAcc':>8} "
      f"{'f1@0':>8} {'f1@10k':>8} {'dF1':>8} {'ep@0':>6} {'ep@10k':>7}")
for _l in sorted({k[1] for k in ab}):
    for _std in (True, False):
        a, b = ab[(AB_STEPS[0], _l, _std)], ab[(AB_STEPS[1], _l, _std)]
        print(f"{_l:>6} {str(_std):>6} {a['accuracy']:>8.4f} {b['accuracy']:>9.4f} "
              f"{b['accuracy'] - a['accuracy']:>+8.4f} "
              f"{a['macro_f1']:>8.4f} {b['macro_f1']:>8.4f} "
              f"{b['macro_f1'] - a['macro_f1']:>+8.4f} "
              f"{a['epochs_run']:>6} {b['epochs_run']:>7}")
print(f"\n(MAX_EPOCHS is {MAX_EPOCHS} - a row hitting it was still improving)")

# %% CELL 10 - THE SWEEP: every checkpoint x every layer x 3 poolings
#
# Rough cost per checkpoint: ~20s download, ~60s encode, then ~1.5s per probe
# (25 layers x 3 for t5, 37 x 3 for the byte models). Around 5 min/checkpoint
# for t5 and 7 min for the byte models, so ~1.7h + ~2.5h + ~2.5h for all three.
for model_name in RUN_MODELS:
    base_id, _, is_byte = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    max_len = max_len_for(is_byte)
    out_path = OUT / f"{model_name}_pos.jsonl"
    already = done_steps(out_path)
    print(f"\n=== {model_name}  ({len(already)}/{len(ALL_STEPS)} steps already done)")

    for step in ALL_STEPS:
        if step in already:
            continue
        t0 = time.time()
        model, encoder = load_encoder(fetch_checkpoint(model_name, step))

        feats, labels, words, dropped = {}, {}, {}, {}
        for split in SPLITS:
            feats[split], labels[split], words[split], dropped[split] = extract(
                encoder, tokenizer, DATA[split], is_byte, max_len, DEVICE)
        # which test words the probe could have memorised from its own training set
        _vocab = {w.lower() for w in words["train"]}
        seen_mask = torch.tensor([w.lower() in _vocab for w in words["test"]])
        del model, encoder
        release()                                  # free the model before probing
        t_enc = time.time() - t0

        layers = sorted(feats["train"])
        rows = []
        for layer in layers:
            for pool in POOLINGS:
                res = train_probe(feats["train"][layer][pool], labels["train"],
                                  feats["validation"][layer][pool], labels["validation"],
                                  feats["test"][layer][pool], labels["test"], DEVICE,
                                  seen_mask=seen_mask)
                rows.append({"model": model_name, "step": step, "layer": layer,
                             "pooling": pool, **res,
                             "n_train": len(labels["train"]),
                             "n_test": len(labels["test"]),
                             "words_dropped": dropped,
                             "majority_baseline": MAJORITY})
        with open(out_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        best = max(rows, key=lambda r: r["accuracy"])
        top = [r for r in rows if r["layer"] == layers[-1] and r["pooling"] == "last"][0]
        print(f"step {step:>5}  encode {t_enc:>4.0f}s  total {time.time()-t0:>4.0f}s  "
              f"| top/last acc {top['accuracy']:.3f} f1 {top['macro_f1']:.3f}  "
              f"| best {best['accuracy']:.3f} (L{best['layer']},{best['pooling']}) "
              f"seen {best['accuracy_seen']:.3f} UNSEEN {best['accuracy_unseen']:.3f}")

        del feats
        release()

# %% CELL 11 - plots
import matplotlib.pyplot as plt
from collections import defaultdict

PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
HEADLINE = "last"


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


for path in sorted(OUT.glob("*_pos.jsonl")):
    model = path.stem.replace("_pos", "")
    rows = read_jsonl(path)
    steps = sorted({r["step"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    head = [r for r in rows if r["pooling"] == HEADLINE]

    # 1. DOSE-RESPONSE: accuracy vs CPT step, one line per layer.
    #    Base (step 0) is drawn as a dashed horizontal reference per layer.
    colours = plt.cm.viridis([i / max(1, len(layers) - 1) for i in range(len(layers))])
    for key, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for layer, colour in zip(layers, colours):
            pts = sorted((r["step"], r[key]) for r in head if r["layer"] == layer)
            cpt = [(s, v) for s, v in pts if s > 0]
            base = [v for s, v in pts if s == 0]
            if base:
                ax.plot(0, base[0], marker="s", ms=5, color=colour)
            if cpt:
                ax.plot([s for s, _ in cpt], [v for _, v in cpt],
                        marker="o", ms=3, color=colour, lw=1.2,
                        label=f"L{layer}" if layer in (layers[0], layers[len(layers)//2], layers[-1]) else None)
        ax.axhline(MAJORITY, color="red", ls=":", lw=1.2, label="majority baseline")
        ax.set_xlabel("CPT step"); ax.set_ylabel(label)
        ax.set_title(f"{model}: {label} vs CPT step ({HEADLINE} pooling)\n"
                     f"squares at x=0 are the un-adapted base model")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(PLOTS / f"{model}_{key}_vs_step.png", dpi=150)
        plt.show()

    # 2. layer profile: accuracy vs layer, one line per checkpoint
    scolours = plt.cm.viridis([i / max(1, len(steps) - 1) for i in range(len(steps))])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for step, colour in zip(steps, scolours):
        pts = sorted((r["layer"], r["accuracy"]) for r in head if r["step"] == step)
        ax.plot([l for l, _ in pts], [v for _, v in pts], marker="o", ms=3,
                color="black" if step == 0 else colour,
                ls="--" if step == 0 else "-", lw=1.6 if step == 0 else 1.0,
                label="base" if step == 0 else (f"step {step}" if step in (steps[len(steps)//2], steps[-1]) else None))
    ax.set_xlabel("Encoder layer"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{model}: accuracy by layer ({HEADLINE} pooling)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / f"{model}_layer_profile.png", dpi=150)
    plt.show()

    # 3. pooling comparison at the layer that does best on the base model
    base_rows = [r for r in rows if r["step"] == 0]
    best_layer = max(base_rows, key=lambda r: r["accuracy"])["layer"] if base_rows else layers[-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, pool in enumerate(POOLINGS):
        pts = sorted((r["step"], r["accuracy"]) for r in rows
                     if r["pooling"] == pool and r["layer"] == best_layer)
        ax.plot([s for s, _ in pts], [v for _, v in pts], marker="o", ms=4,
                color=f"C{i}", label=pool)
    ax.set_xlabel("CPT step"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{model}: pooling comparison at layer {best_layer}")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / f"{model}_pooling.png", dpi=150)
    plt.show()

    # 4. overfitting monitor - the train/test gap we said we would report
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pts = sorted((r["step"], r["train_accuracy"] - r["accuracy"]) for r in head
                 if r["layer"] == best_layer)
    ax.plot([s for s, _ in pts], [v for _, v in pts], marker="o", ms=4, color="C3")
    ax.set_xlabel("CPT step"); ax.set_ylabel("train acc - test acc")
    ax.set_title(f"{model}: probe overfitting gap at layer {best_layer}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / f"{model}_overfit_gap.png", dpi=150)
    plt.show()

    # 5. per-class F1 vs step, for the classes with enough test support to mean anything
    sup = {k: v["support"] for k, v in head[0]["per_class"].items()}
    keep = [k for k, v in sorted(sup.items(), key=lambda kv: -kv[1])[:8] if v >= 30]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cls in enumerate(keep):
        pts = sorted((r["step"], r["per_class"][cls]["f1"]) for r in head
                     if r["layer"] == best_layer)
        ax.plot([s for s, _ in pts], [v for _, v in pts], marker="o", ms=3,
                color=f"C{i}", label=f"{cls} (n={sup[cls]})")
    ax.set_xlabel("CPT step"); ax.set_ylabel("F1")
    ax.set_title(f"{model}: per-class F1 at layer {best_layer} (support >= 30)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(PLOTS / f"{model}_per_class.png", dpi=150)
    plt.show()

# %% CELL 12 - summary table (base vs best vs final, the dose-response number)
for path in sorted(OUT.glob("*_pos.jsonl")):
    model = path.stem.replace("_pos", "")
    rows = [r for r in read_jsonl(path) if r["pooling"] == HEADLINE]
    by_layer = defaultdict(dict)
    for r in rows:
        by_layer[r["layer"]][r["step"]] = (r["accuracy"], r["macro_f1"])
    print(f"\n=== {model}  ({HEADLINE} pooling)")
    print(f"{'':>6} {'-------- accuracy --------':^28} {'-------- macro F1 --------':^28}")
    print(f"{'layer':>6} {'base':>8} {'best':>8} {'gain':>10} "
          f"{'base':>8} {'best':>8} {'gain':>10} {'@step':>7}")
    for layer in sorted(by_layer):
        d = by_layer[layer]
        if 0 not in d:
            continue
        cpt = {s: v for s, v in d.items() if s > 0}
        if not cpt:
            continue
        ba, bf = d[0]
        # rank checkpoints by macro-F1: it has ~2x the headroom of accuracy here
        bstep, (ca, cf) = max(cpt.items(), key=lambda kv: kv[1][1])
        print(f"{layer:>6} {ba:>8.4f} {ca:>8.4f} {ca - ba:>+10.4f} "
              f"{bf:>8.4f} {cf:>8.4f} {cf - bf:>+10.4f} {bstep:>7}")

# %% CELL 12b - DIAGNOSTIC: did the optimiser actually move the weights?
#
# CPT loaded the model with torch_dtype=torch.bfloat16 (continued_pretrain
# _lafand.py:234), so AdamW wrote updates into bf16 parameters with no fp32
# master copy. bf16 has an 8-bit mantissa, so the gap between representable
# values scales with magnitude. T5 embeddings sit near |w| ~ 18, where that
# gap is 0.125 - an update of ~lr = 1e-4 rounds straight back to the original
# number and is discarded. Attention/FFN weights are ~400x smaller, so their
# gaps are ~400x finer and updates do land.
#
# Comparison is done in bf16 because that is the dtype the checkpoints are
# stored in AND the dtype CPT trained in, so "identical" means the optimiser
# genuinely never moved that value.
#
# RAM: holds two fp32 copies of the model (~6GB for t5-large). Run this on a
# fresh runtime, or after the sweep - not alongside it.
import math
import numpy as np
from collections import defaultdict

DIAG_MODEL, DIAG_STEP = "t5", 10000


def _bucket(k):
    # lm_head IS shared.weight when tie_word_embeddings=True (t5-large: True),
    # so it belongs with the embeddings, not with attention/ffn.
    if any(t in k for t in ("shared", "embed_tokens", "lm_head")):
        return "embeddings"
    if "layer_norm" in k:
        return "layernorm"
    if "relative_attention_bias" in k:
        return "rel_attn_bias"
    return "attention/ffn"


def _bf16_gap(v):
    return 2.0 ** (math.floor(math.log2(abs(v))) - 7) if v else 0.0


_bid, _, _ = MODELS[DIAG_MODEL]
# named_parameters() deduplicates tied tensors; state_dict() would count the
# embedding matrix 4x (shared / encoder.embed_tokens / decoder.embed_tokens /
# lm_head) and inflate its share of the totals.
_b = dict(AutoModelForSeq2SeqLM.from_pretrained(_bid, dtype=torch.float32)
          .named_parameters())
_c = dict(AutoModelForSeq2SeqLM.from_pretrained(
    fetch_checkpoint(DIAG_MODEL, DIAG_STEP), dtype=torch.float32).named_parameters())

_agg = defaultdict(lambda: [0, 0, 0.0, 0.0])          # n, n_identical, sum w^2, max|d|
for _k, _x in _b.items():
    if _k not in _c or not _x.is_floating_point():
        continue
    _xb, _yb = _x.to(torch.bfloat16), _c[_k].to(torch.bfloat16)
    _g = _agg[_bucket(_k)]
    _g[0] += _x.numel()
    _g[1] += (_xb == _yb).sum().item()
    _g[2] += _x.double().pow(2).sum().item()
    _g[3] = max(_g[3], (_yb.float() - _xb.float()).abs().max().item())

print(f"{DIAG_MODEL}: base vs checkpoint-{DIAG_STEP}, compared in bf16\n")
print(f"{'group':16s} {'params':>13} {'% NEVER moved':>14} {'rms |w|':>9} "
      f"{'bf16 gap':>10} {'max |delta|':>12}")
for _name, (_n, _same, _sq, _mx) in sorted(_agg.items()):
    _rms = np.sqrt(_sq / _n)
    print(f"{_name:16s} {_n:>13,} {100 * _same / _n:>13.2f}% {_rms:>9.4f} "
          f"{_bf16_gap(_rms):>10.6f} {_mx:>12.2e}")

print(f"\nAn AdamW step at lr=1e-4 can only change a weight whose bf16 gap is")
print(f"below ~2e-4. Compare the 'bf16 gap' column against that threshold:")
print(f"rows far above it could not be updated at all, whatever the gradient.")

del _b, _c
release()

# %% CELL 13 - save results off the VM before it disconnects
# from google.colab import files
# shutil.make_archive("/content/probing_results", "zip", OUT)
# files.download("/content/probing_results.zip")

# or mount Drive and copy:
# from google.colab import drive
# drive.mount("/content/drive")
# shutil.copytree(OUT, "/content/drive/MyDrive/probing", dirs_exist_ok=True)
