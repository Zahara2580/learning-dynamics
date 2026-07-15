"""
Port of the data pieces of lafand-mt mt5_byt5_pre_training/util.py:
the linecache-backed Seq2SeqDataset and the task_type='generation_id'
branch of Seq2SeqDataCollator, which is the path their pretraining used
(distribute_train.sh passes --task_type generation_id).

The dataset reads {type_path}.source/.target files (space-separated
token-id strings produced offline by lafand_preprocess.py); the collator
parses the ids back, truncates to max lengths, and pads to the longest
sequence in the batch.

Faithfulness notes:
  - labels are padded with pad_token_id (0), NOT -100, exactly as the
    original does - padding therefore contributes to the loss, as it did
    for nguni-byt5's own training. Set ignore_pad_in_labels=True to pad
    with -100 instead (documented deviation, off by default).
  - truncation happens here, after masking (original behavior): a
    truncated example can lose target spans whose sentinels survive in
    the input. lafand_preprocess.py's pre-truncation keeps this rare.
  - no decoder_input_ids are built; T5 derives them from labels
    internally, as in the original.
"""

import linecache
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset


class LafandSeq2SeqDataset(Dataset):
    """Reads {type_path}.source/.target line-by-line via linecache
    (port of lafand's Seq2SeqDataset, generation_id usage)."""

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
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}


class LafandSeq2SeqCollator:
    """Port of lafand's Seq2SeqDataCollator, task_type='generation_id'
    branch: parse space-separated id strings, truncate to max lengths,
    pad to the longest sequence in the batch."""

    def __init__(
        self,
        pad_token_id: int,
        max_source_length: int,
        max_target_length: int,
        ignore_pad_in_labels: bool = False,
    ):
        assert pad_token_id is not None, "pad_token_id must be defined"
        self.pad_token_id = pad_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # -100 label padding is a deviation from the original (which pads
        # labels with pad_token_id, so padding contributes to the loss).
        self.label_pad_id = -100 if ignore_pad_in_labels else pad_token_id

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
            i[:max_len] if len(i) > max_len else i + [self.label_pad_id] * (max_len - len(i))
            for i in labels
        ]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": torch.LongTensor(labels),
        }
