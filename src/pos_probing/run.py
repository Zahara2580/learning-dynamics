"""Train linear POS probes on frozen encoder representations."""

# %pip -q install "transformers==4.57.1" "huggingface_hub>=0.34,<1" sentencepiece scikit-learn pandas matplotlib

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import shutil
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, T5EncoderModel

HEWITT_COMMIT = 'be70d7f42324797dcf2bb5360a94d8e64100fea8'
MULLER_COMMIT = 'b45b88df3af31fbe890043b657ce76c2fed1720b'
DATA_COMMIT = '376f4161f0425584d4bd7664122b56fa026926d3'
IMPLEMENTATION_ID = '5569fb33ea2f0154e87a20bc69ffad75deac2d12cced02271227898f23a08697'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODELS = {
    't5': ('google-t5/t5-large', 'ChonkeyJellyfish/cpt-xhosa-t5-large', False),
    'byt5': ('google/byt5-large', 'ChonkeyJellyfish/cpt-xhosa-byt5-large', True),
    'nguni-byt5': ('francois-meyer/nguni-byt5-large', 'ChonkeyJellyfish/cpt-xhosa-nguni-byt5-large', True),
    't5-bilingual': ('google-t5/t5-large', 'ChonkeyJellyfish/cpt-bilingual-t5-large', False),
    'byt5-bilingual': ('google/byt5-large', 'ChonkeyJellyfish/cpt-bilingual-byt5-large', True),
    'nguni-byt5-bilingual': (
        'francois-meyer/nguni-byt5-large',
        'ChonkeyJellyfish/cpt-bilingual-nguni-byt5-large',
        True
    )
}
STEPS = [0] + list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
POOLINGS = ['mean', 'first', 'last']
LAYERS = None
ENCODE_BATCH = 4
MAX_LEN = {False: 512, True: 1024}
SEED = 0
STANDARDISE = True
TRAINING = dict(
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    batch_size=64,
    max_epochs=30,
    patience=3,
    min_delta=1e-06
)
LABELS = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X'
]
LABEL_TO_ID = {tag: i for i, tag in enumerate(LABELS)}

OUT = Path("results/pos_probing")
RUN_MODELS = ["t5-bilingual", "byt5-bilingual", "nguni-byt5-bilingual"]
CHECKPOINT_CACHE = None


def parse_pos(text):
    """Parse sentence-separated words and UPOS labels."""
    sentences, words, tags = ([], [], [])
    for line in text.splitlines() + ['']:
        if not line.strip():
            if words:
                sentences.append((words, tags))
                words, tags = ([], [])
            continue
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2 or fields[1] not in LABEL_TO_ID:
            raise ValueError(f'Unexpected POS row: {line!r}')
        word, tag = fields
        if any((ch.isspace() for ch in word)):
            raise ValueError(f'Unexpected whitespace inside a word: {word!r}')
        words.append(word)
        tags.append(LABEL_TO_ID[tag])
    if not sentences:
        raise ValueError('Empty POS data')
    return sentences


def read_data():
    """Load the pinned MasakhaPOS splits and record their hashes."""
    data, hashes = ({}, {})
    data_dir = OUT / 'data'
    data_dir.mkdir(exist_ok=True)
    for split, filename in [('train', 'train.txt'), ('validation', 'dev.txt'), ('test', 'test.txt')]:
        url = f'https://raw.githubusercontent.com/masakhane-io/masakhane-pos/{DATA_COMMIT}/data/xho/{filename}'
        path = data_dir / f'{DATA_COMMIT}_{filename}'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                contents = response.read()
            path.write_bytes(contents)
        contents = path.read_bytes()
        hashes[split] = hashlib.sha256(contents).hexdigest()
        data[split] = parse_pos(contents.decode('utf-8'))
        print(split, len(data[split]), 'sentences,', sum((len(w) for w, _ in data[split])), 'words')
    return (data, hashes)


def encode_words(tokenizer, words, is_byte, max_len):
    """Align annotated words with byte or subword token spans."""
    text = ' '.join(words)
    if is_byte:
        raw_ids = list(text.encode('utf-8'))
        actual = tokenizer(text, add_special_tokens=False)['input_ids']
        if actual != [value + 3 for value in raw_ids]:
            raise ValueError('Tokenizer is not using the expected ByT5 UTF-8 mapping.')
        ids = tokenizer(text, add_special_tokens=True)['input_ids']
        if ids[:len(actual)] != actual or ids[len(actual):] != [tokenizer.eos_token_id]:
            raise ValueError('Unexpected ByT5 special-token layout.')
        spans, cursor = ([], 0)
        for word in words:
            end = cursor + len(word.encode('utf-8'))
            spans.append((cursor, end))
            cursor = end + 1
    else:
        if not tokenizer.is_fast:
            raise ValueError('T5 requires a fast tokenizer with character offsets.')
        enc = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False
        )
        ids, spans, cursor = (enc['input_ids'], [], 0)
        for word in words:
            end = cursor + len(word)
            idx = [
                i for i,
                ((a, b), special) in enumerate(zip(enc['offset_mapping'], enc['special_tokens_mask'])) if not special and b > a and (a < end) and (b > cursor)
            ]
            if not idx or idx != list(range(idx[0], idx[-1] + 1)):
                raise ValueError(f'Cannot align word {word!r} in {text!r}')
            spans.append((idx[0], idx[-1] + 1))
            cursor = end + 1
        if any((left[1] > right[0] for left, right in zip(spans, spans[1:]))):
            raise ValueError('A T5 token overlaps multiple annotated words; inspect this sentence.')
    if len(ids) > max_len:
        raise ValueError(f'Input has {len(ids)} tokens, limit {max_len}. Increase MAX_LEN; no truncation applied.')
    assert len(spans) == len(words) and all((0 <= a < b <= len(ids) for a, b in spans))
    return (ids, spans)


def pool_word(h, start, end, pooling):
    """Pool the token states belonging to one word."""
    if pooling == 'first':
        return h[start]
    if pooling == 'last':
        return h[end - 1]
    if pooling == 'mean':
        return h[start:end].mean(0)
    raise ValueError(pooling)


@torch.no_grad()
def extract_features(encoder, tokenizer, data, is_byte, layers=None, poolings=None, batch_size=4, max_len=1024, device=DEVICE):
    """Extract frozen word representations for each layer and pooling method."""
    encoder.eval()
    encoder.requires_grad_(False)
    poolings = list(POOLINGS if poolings is None else poolings)
    if not poolings or len(set(poolings)) != len(poolings) or (not set(poolings) <= {'mean', 'first', 'last'}):
        raise ValueError('Choose unique pooling methods: mean, first, last')
    encoded = [encode_words(tokenizer, words, is_byte, max_len) for words, _ in data]
    selected = list(range(encoder.config.num_layers + 1)) if layers is None else sorted(set(layers))
    if not selected or min(selected) < 0 or max(selected) > encoder.config.num_layers:
        raise ValueError('Invalid encoder layer indices')
    features = {layer: {pool: [] for pool in poolings} for layer in selected}
    for start in range(0, len(data), batch_size):
        chunk = encoded[start:start + batch_size]
        ids = pad_sequence(
            [torch.tensor(item[0]) for item in chunk],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        mask = torch.zeros_like(ids)
        for row, (tokens, _) in enumerate(chunk):
            mask[row, :len(tokens)] = 1
        states = encoder(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            output_hidden_states=True,
            return_dict=True
        ).hidden_states
        for layer in selected:
            for row, (_, spans) in enumerate(chunk):
                h = states[layer][row].float()
                for pool in poolings:
                    pooled = torch.stack([pool_word(h, a, b, pool) for a, b in spans]).cpu()
                    if not torch.isfinite(pooled).all():
                        raise ValueError(f'Non-finite features at layer {layer}; use float32 extraction.')
                    features[layer][pool].append(pooled)
        del states
    labels = [torch.tensor(tags, dtype=torch.long) for _, tags in data]
    return (features, labels)


class Probe(nn.Module):
    """Provide the base class for linear POS probes."""

    def print_param_count(self):
        """Print the number of probe parameters."""
        total_params = 0
        for param in self.parameters():
            total_params += numpy.prod(param.size())
        tqdm.write('Probe has {} parameters'.format(total_params))


class CrossEntropyLoss(nn.Module):
    """Sum valid word losses and divide by the number of sentences."""

    def __init__(self, args):
        """Initialise the module and its settings."""
        super(CrossEntropyLoss, self).__init__()
        tqdm.write('Constructing CrossEntropyLoss')
        self.args = args
        self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def forward(self, predictions, label_batch, length_batch):
        """Compute predictions or sentence-normalised loss for a batch."""
        if len(label_batch.size()) == 2:
            batchlen, seqlen, class_count = predictions.size()
            total_sents = torch.sum(length_batch != 0).float()
            predictions = predictions.view(batchlen * seqlen, class_count)
            label_batch = label_batch.view(batchlen * seqlen).long()
            cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch) / total_sents
        return (cross_entropy_loss, total_sents)


class OneWordLinearLabelProbe(Probe):
    """Predict POS labels with a single affine transformation."""

    def __init__(self, hidden_dim, num_labels):
        """Initialise the module and its settings."""
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_labels)

    def forward(self, batch):
        """Compute predictions or sentence-normalised loss for a batch."""
        return self.linear(batch)


def sentence_batches(xs, ys, batch_size, shuffle=False, generator=None):
    """Pad sentence features and labels into reproducible batches."""
    order = torch.randperm(len(xs), generator=generator).tolist() if shuffle else list(range(len(xs)))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield (
            pad_sequence([xs[i] for i in idx], batch_first=True),
            pad_sequence([ys[i] for i in idx], batch_first=True, padding_value=-1),
            torch.tensor([len(ys[i]) for i in idx])
        )


def fit_scaler(xs, enabled=True):
    """Calculate feature scaling statistics from training words."""
    if not enabled:
        return (torch.zeros(xs[0].shape[1]), torch.ones(xs[0].shape[1]))
    count = sum((len(x) for x in xs))
    mean = sum((x.double().sum(0) for x in xs)) / count
    variance = sum(((x.double() - mean).square().sum(0) for x in xs)) / max(count - 1, 1)
    return (mean.float(), variance.sqrt().clamp_min(1e-06).float())


class ProbeRegimen:
    """Train probes with fixed-rate Adam and validation-loss stopping."""

    def __init__(self, config, device=DEVICE):
        """Initialise the module and its settings."""
        self.config = dict(config)
        self.device = device
        self.loss = CrossEntropyLoss({})

    def set_optimizer(self, probe):
        """Configure Adam using the experiment settings."""
        c = self.config
        self.optimizer = optim.Adam(
            probe.parameters(),
            lr=c['lr'],
            betas=c['betas'],
            weight_decay=c['weight_decay']
        )

    @torch.no_grad()
    def evaluate(self, probe, xs, ys, mu, sd):
        """Compute sentence-normalised loss and word predictions."""
        probe.eval()
        total_loss, total_sentences, gold, predictions = (0.0, 0, [], [])
        for x, y, lengths in sentence_batches(xs, ys, self.config['batch_size']):
            x = ((x - mu) / sd).to(self.device)
            y = y.to(self.device)
            lengths = lengths.to(self.device)
            logits = probe(x)
            loss, count = self.loss(logits, y, lengths)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite evaluation loss')
            total_loss += loss.item() * count.item()
            total_sentences += count.item()
            keep = y != -1
            gold.extend(y[keep].cpu().tolist())
            predictions.extend(logits.argmax(-1)[keep].cpu().tolist())
        return (total_loss / total_sentences, gold, predictions)

    def train_until_convergence(self, probe, train_x, train_y, dev_x, dev_y, mu, sd, seed):
        """Train until validation loss stops improving and restore the best weights."""
        probe.to(self.device)
        self.set_optimizer(probe)
        generator = torch.Generator().manual_seed(seed)
        c = self.config
        best_loss = float('inf')
        best_state = None
        bad = 0
        history = []
        for epoch in range(1, c['max_epochs'] + 1):
            probe.train()
            total_loss, total_sentences = (0.0, 0)
            for x, y, lengths in sentence_batches(train_x, train_y, c['batch_size'], True, generator):
                x = ((x - mu) / sd).to(self.device)
                y = y.to(self.device)
                lengths = lengths.to(self.device)
                self.optimizer.zero_grad()
                loss, count = self.loss(probe(x), y, lengths)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * count.item()
                total_sentences += count.item()
            dev_loss, _, _ = self.evaluate(probe, dev_x, dev_y, mu, sd)
            history.append(dict(
                epoch=epoch,
                train_loss=total_loss / total_sentences,
                dev_loss=dev_loss,
                lr=self.optimizer.param_groups[0]['lr']
            ))
            if dev_loss < best_loss - c['min_delta']:
                best_loss = dev_loss
                best_epoch = epoch
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            else:
                bad += 1
            if bad >= c['patience']:
                break
        if best_state is None:
            raise RuntimeError('No valid probe state')
        probe.load_state_dict(best_state)
        return dict(best_epoch=best_epoch, history=history)


def score_tags(gold, pred):
    """Compute word accuracy and macro-F1 with per-class scores."""
    ids = list(range(len(LABELS)))
    _, _, per_f1, support = precision_recall_fscore_support(gold, pred, labels=ids, zero_division=0)
    present = [i for i in ids if support[i] > 0]
    return dict(
        accuracy=float(accuracy_score(gold, pred)),
        macro_f1_all_17=float(f1_score(gold, pred, labels=ids, average='macro', zero_division=0)),
        macro_f1_present=float(f1_score(gold, pred, labels=present, average='macro', zero_division=0)),
        n_words=len(gold),
        n_present_classes=len(present),
        per_class={LABELS[i]: dict(f1=float(per_f1[i]), support=int(support[i])) for i in ids}
    )


def atomic_json(path, value):
    """Write JSON through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def fit_one_layer(xs, ys, seed=SEED, config=TRAINING, device=DEVICE):
    """Train and score one probe using training-only feature scaling."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mu, sd = fit_scaler(xs['train'], STANDARDISE)
    probe = OneWordLinearLabelProbe(xs['train'][0].shape[1], len(LABELS))
    regimen = ProbeRegimen(config, device)
    training = regimen.train_until_convergence(
        probe,
        xs['train'],
        ys['train'],
        xs['validation'],
        ys['validation'],
        mu,
        sd,
        seed
    )
    result = {}
    prediction_records = {}
    for split in ['validation', 'test']:
        loss, gold, pred = regimen.evaluate(probe, xs[split], ys[split], mu, sd)
        result[split] = dict(loss=loss, **score_tags(gold, pred))
        prediction_records[split] = dict(gold=gold, predicted=pred)
    return (probe, mu, sd, dict(**result, **training), prediction_records)


def run_experiments():
    """Probe each requested checkpoint, layer and pooling method."""
    data, data_hashes = read_data()
    api = HfApi()
    from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
    repos = sorted({repo for name in RUN_MODELS for repo in (MODELS[name][:2] if any((step > 0 for step in STEPS)) else MODELS[name][:1])})
    revisions = {}
    for repo in repos:
        try:
            revisions[repo] = api.model_info(repo).sha
        except (RepositoryNotFoundError, GatedRepoError) as error:
            raise RuntimeError(f'Cannot access {repo}. Use --login with a token that can read this repository. Also check the exact repository name in MODELS. A private repository without access can appear as RepositoryNotFound. If HF_TOKEN is already configured, ensure it is valid.') from error
    versions = {p: importlib.metadata.version(p) for p in [
        'torch',
        'transformers',
        'huggingface-hub',
        'numpy',
        'scikit-learn',
        'sentencepiece'
    ]}
    manifest = dict(
        implementation=IMPLEMENTATION_ID,
        hewitt=HEWITT_COMMIT,
        muller=MULLER_COMMIT,
        dataset=DATA_COMMIT,
        data_hashes=data_hashes,
        models={n: MODELS[n] for n in RUN_MODELS},
        revisions=revisions,
        steps=STEPS,
        poolings=POOLINGS,
        layers=LAYERS,
        seed=SEED,
        standardise=STANDARDISE,
        training=TRAINING,
        max_len=MAX_LEN,
        encode_batch=ENCODE_BATCH,
        dtype='float32',
        labels=LABELS,
        versions=versions,
        device=DEVICE,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None
    )
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    run_dir = OUT / digest
    run_dir.mkdir(exist_ok=True)
    atomic_json(run_dir / 'manifest.json', manifest)
    atomic_json(
        run_dir / 'evaluation_sentences.json',
        {s: [dict(words=w, tags=[LABELS[t] for t in y]) for w, y in data[s]] for s in ['validation', 'test']}
    )
    majority = Counter((t for _, y in data['train'] for t in y)).most_common(1)[0][0]
    baseline = {}
    for split in ['validation', 'test']:
        gold = [t for _, y in data[split] for t in y]
        baseline[split] = score_tags(gold, [majority] * len(gold))
    atomic_json(run_dir / 'majority.json', dict(label=LABELS[majority], scores=baseline))
    print('Run directory:', run_dir)
    for name in RUN_MODELS:
        base, cpt, is_byte = MODELS[name]
        tokenizer = AutoTokenizer.from_pretrained(base, revision=revisions[base], use_fast=not is_byte)
        for step in STEPS:
            checkpoint_dir = run_dir / name / f'step_{step}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if (checkpoint_dir / 'complete.json').exists():
                print('Already complete:', name, step)
                continue
            repo = base if step == 0 else cpt
            kwargs = dict(revision=revisions[repo], torch_dtype=torch.float32, output_loading_info=True)
            if step:
                kwargs['subfolder'] = f'checkpoint-{step}'
                kwargs['cache_dir'] = str(CHECKPOINT_CACHE)
            encoder, info = T5EncoderModel.from_pretrained(repo, **kwargs)
            if info['missing_keys'] or info.get('mismatched_keys', []) or info.get('error_msgs', []):
                raise RuntimeError(f'Encoder weights did not load completely: {info}')
            encoder.requires_grad_(False)
            encoder.eval()
            encoder.to(DEVICE)
            atomic_json(checkpoint_dir / 'loading_info.json', info)
            features, labels = ({}, {})
            for split in ['train', 'validation', 'test']:
                features[split], labels[split] = extract_features(
                    encoder,
                    tokenizer,
                    data[split],
                    is_byte,
                    layers=LAYERS,
                    poolings=POOLINGS,
                    batch_size=ENCODE_BATCH,
                    max_len=MAX_LEN[is_byte],
                    device=DEVICE
                )
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            selected = sorted(features['train'])
            for layer in tqdm(selected, desc=f'{name}, CPT {step}'):
                for pool in POOLINGS:
                    destination = checkpoint_dir / f'pooling_{pool}' / f'layer_{layer}'
                    if (destination / 'metrics.json').exists() and (destination / 'probe.pt').exists() and (destination / 'predictions.json').exists():
                        continue
                    destination.mkdir(parents=True, exist_ok=True)
                    xs = {split: features[split][layer][pool] for split in features}
                    probe, mu, sd, result, predictions = fit_one_layer(xs, labels)
                    temp = destination / 'probe.tmp'
                    torch.save(
                        dict(
                            state_dict={k: v.cpu() for k, v in probe.state_dict().items()},
                            mean=mu,
                            std=sd,
                            hidden_dim=mu.numel(),
                            labels=LABELS,
                            pooling=pool
                        ),
                        temp
                    )
                    temp.replace(destination / 'probe.pt')
                    atomic_json(destination / 'predictions.json', predictions)
                    atomic_json(
                        destination / 'metrics.json',
                        dict(model=name, step=step, layer=layer, pooling=pool, **result)
                    )
                    del probe, xs
                for split in features:
                    del features[split][layer]
            atomic_json(checkpoint_dir / 'complete.json', dict(layers=selected, poolings=POOLINGS))
            del features, labels
            gc.collect()
            if step and CHECKPOINT_CACHE.exists():
                shutil.rmtree(CHECKPOINT_CACHE)
                print('Removed downloaded CPT checkpoint cache.')
            free_gib = shutil.disk_usage(OUT).free / 2 ** 30
            print(f'Disk free: {free_gib:.1f} GiB')
    return run_dir


def main() -> None:
    """Configure and run the POS checkpoint sweep."""
    global OUT, RUN_MODELS, STEPS, LAYERS, CHECKPOINT_CACHE
    parser = argparse.ArgumentParser(description="Linear POS probing of CPT checkpoints")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Results directory.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=RUN_MODELS)
    parser.add_argument("--steps", nargs="+", type=int, choices=STEPS, default=STEPS)
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Encoder layers; defaults to all.")
    parser.add_argument("--login", action="store_true", help="Prompt for Hugging Face access.")
    args = parser.parse_args()
    if args.login:
        from getpass import getpass
        login(token=getpass("Hugging Face token: "), add_to_git_credential=False)
    OUT, RUN_MODELS, STEPS, LAYERS = args.output_dir, args.models, args.steps, args.layers
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pos_cpt_") as cache:
        CHECKPOINT_CACHE = Path(cache)
        run_experiments()


if __name__ == "__main__":
    main()
