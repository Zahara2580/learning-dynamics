"""Load tokenized examples, group batches by length and pad model inputs."""

import linecache
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class Seq2SeqDataset(Dataset):
    """Load paired source and target token IDs from line aligned files."""

    def __init__(self, data_dir: str, type_path: str = "train", n_obs: int = None):
        self.src_file = Path(data_dir) / f"{type_path}.source"
        self.tgt_file = Path(data_dir) / f"{type_path}.target"
        self.src_lens = [len(x) for x in Path(self.src_file).open().readlines()]
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self) -> int:
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}


def sortish_sampler_indices(data: List[int], bs: int, shuffle: bool = True) -> np.ndarray:
    """Return example indices grouped by length, optionally shuffling batches."""
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=lambda i: data[i], reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = int(np.argmax([data[ck[0]] for ck in ck_idx]))
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
    rest = ck_idx[1:]
    random.shuffle(rest)
    return np.concatenate([ck_idx[0]] + rest) if rest else np.asarray(ck_idx[0])


class SortishSampler(Sampler):
    """Group similar-length examples into shuffled batches."""

    def __init__(self, data: List[int], batch_size: int, shuffle: bool = True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


class Seq2SeqCollator:
    """Truncate and pad tokenized examples for sequence-to-sequence training."""

    LABEL_PAD_ID = -100

    def __init__(self, pad_token_id: int, max_source_length: int, max_target_length: int):
        assert pad_token_id is not None, "pad_token_id must be defined"
        self.pad_token_id = pad_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        sources = [x["src_texts"] for x in batch]
        targets = [x["tgt_texts"] for x in batch]

        input_ids = [list(map(int, s.split())) for s in sources]
        max_len = max(len(i) for i in input_ids)
        if max_len > self.max_source_length:
            max_len = self.max_source_length
        attention_mask = [
            [1] * max_len if len(i) > max_len else [1] * len(i) + [0] * (max_len - len(i))
            for i in input_ids
        ]
        input_ids = [
            i[:max_len] if len(i) > max_len else i + [self.pad_token_id] * (max_len - len(i))
            for i in input_ids
        ]

        labels = [list(map(int, s.split())) for s in targets]
        max_len = max(len(i) for i in labels)
        if max_len > self.max_target_length:
            max_len = self.max_target_length
        labels = [
            i[:max_len] if len(i) > max_len else i + [self.LABEL_PAD_ID] * (max_len - len(i))
            for i in labels
        ]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": torch.LongTensor(labels),
        }
