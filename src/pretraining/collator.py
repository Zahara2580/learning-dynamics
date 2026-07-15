"""
DEPRECATED (2026-07-15): retired in favor of the lab-mandated lafand-mt
pipeline (offline i.i.d. masking) - see src/lafand_pretraining/. Kept
for reference and for the historical diagnostic scripts that import it.

Data collator for T5-style span corruption, used during continued
pretraining (CPT).

This is NOT an official transformers class - transformers only ships
general-purpose collators (DataCollatorForSeq2Seq, DataCollatorForLanguageModeling,
etc.), none of which implement T5's span corruption. This implementation
is adapted from the community-standard reference (originally
run_t5_mlm_flax.py from Google/HuggingFace, later ported to PyTorch),
which is the version essentially everyone who pretrains T5-family models
from scratch or via CPT ends up using or adapting.

At a high level, for each sequence in a batch:
    1. Randomly select which token positions to "corrupt" (hide), in
       contiguous spans rather than scattered individual tokens.
    2. Replace each corrupted span with a single sentinel token
       (<extra_id_0>, <extra_id_1>, ...), producing the corrupted input.
    3. Build the target sequence: just the hidden spans, each preceded
       by its matching sentinel token, which is what the model must
       learn to predict.

This collator is only used for CPT. Fine-tuning (MT, D2T) uses the
standard transformers DataCollatorForSeq2Seq instead, since fine-tuning
data already has real source/target pairs and needs no corruption.
"""

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


def compute_input_and_target_lengths(
    input_length: int, noise_density: float, mean_noise_span_length: float
) -> tuple[int, int]:
    """
    Compute the expanded input length and target length needed so that,
    after span corruption, the final input is exactly input_length long.

    Because corrupting a span of several tokens replaces them with a
    single sentinel token, the *pre-corruption* sequence needs to be
    longer than input_length to end up at exactly input_length after
    corruption. This works out the correct pre-corruption length to
    tokenize/chunk to.

    :param input_length: Desired final (post-corruption) input length.
    :param noise_density: Fraction of tokens to corrupt.
    :param mean_noise_span_length: Average length of each corrupted span.
    :return: (expanded_input_length, target_length) tuple.
    """
    def _tokens_length_to_inputs_length_targets_length(tokens_length: int) -> tuple[int, int]:
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        # Each span (noise or non-noise) is preceded by one sentinel
        # token once corrupted, except the last non-noise span.
        input_length_with_sentinels = num_nonnoise_tokens + num_noise_spans + 1
        target_length_with_sentinels = num_noise_tokens + num_noise_spans + 1
        return input_length_with_sentinels, target_length_with_sentinels

    tokens_length = input_length
    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= input_length:
        tokens_length += 1

    expanded_input_length, target_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    return tokens_length, target_length


def random_spans_noise_mask(
    length: int, noise_density: float, mean_noise_span_length: float
) -> np.ndarray:
    """
    Build a boolean mask marking which token positions to corrupt.

    Rather than corrupting scattered individual tokens, corruption is
    grouped into contiguous spans (average length mean_noise_span_length),
    alternated with untouched ("non-noise") spans, covering the full
    sequence. See _random_segmentation for how span lengths are chosen.

    :param length: Sequence length to build the mask for.
    :param noise_density: Fraction of tokens to corrupt.
    :param mean_noise_span_length: Average length of each corrupted span.
    :return: Boolean array of shape (length,), True where a token is corrupted.
    """
    orig_length = length
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    def _random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
        """Randomly partition num_items into num_segments positive-integer pieces."""
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    # Interleave non-noise and noise spans: nonnoise, noise, nonnoise,
    # noise, ... so corruption never starts right at position 0.
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]


def create_sentinel_ids(mask_indices: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Assign a unique sentinel id to each contiguous span of masked tokens.

    Only the first position of each span keeps a (sentinel) id; other
    positions in the same span are marked -1, meaning "delete this
    position" (handled by filter_input_ids).

    :param mask_indices: Boolean mask of shape (batch, length), True where corrupted.
    :param vocab_size: Sentinel base: sentinel ids count down from vocab_size - 1.
        Usually len(tokenizer), but byt5 uses 259 (see DataCollatorForT5MLM).
    :return: Integer array of shape (batch, length): sentinel id at span starts, -1 elsewhere, 0 outside spans.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(
        start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
    )
    sentinel_ids = np.where(sentinel_ids != 0, (vocab_size - sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids


def filter_input_ids(
    input_ids: np.ndarray, sentinel_ids: np.ndarray, eos_token_id: int
) -> np.ndarray:
    """
    Build the final token sequence: replace masked spans with their
    sentinel id, drop all other positions within those spans, and
    append an EOS token (accounted for by the "+ 1" in
    compute_input_and_target_lengths).

    Called twice per batch with complementary masks: once to build the
    corrupted input, once (with the inverted mask) to build the target.

    :param input_ids: Original token ids, shape (batch, length).
    :param sentinel_ids: Output of create_sentinel_ids for this mask.
    :param eos_token_id: Tokenizer's end-of-sequence token id.
    :return: Array of shape (batch, new_length), with EOS appended.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # Real tokens and sentinel ids are always >= 0; positions marked -1
    # (non-start positions within a corrupted span) are dropped here.
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
        [input_ids, np.full((batch_size, 1), eos_token_id, dtype=input_ids.dtype)],
        axis=-1,
    )

    return input_ids


def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one position right, prepending decoder_start_token_id.

    Standard "teacher forcing" setup for encoder-decoder models: at each
    decoding step, the decoder is fed the correct previous token (from
    the real labels), not its own prior prediction.

    :param input_ids: Label token ids, shape (batch, length).
    :param pad_token_id: Tokenizer's pad token id, used to replace any -100 label positions.
    :param decoder_start_token_id: Token id to prepend as the decoder's first input.
    :return: Shifted array of the same shape as input_ids.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class DataCollatorForT5MLM:
    """
    Collator that applies T5-style span corruption to a batch of
    already-tokenized, fixed-length chunks (as produced by preprocess_wura.py).

    :param tokenizer: Tokenizer matching the model being pretrained.
    :param noise_density: Fraction of tokens to corrupt (T5 default: 0.15).
    :param mean_noise_span_length: Average length of each corrupted span (T5 default: 3.0).
    :param input_length: Final (post-corruption) input length the model expects.
    :param target_length: Final target length (from compute_input_and_target_lengths).
    :param pad_token_id: Tokenizer's pad token id.
    :param decoder_start_token_id: Model's decoder start token id.
    :param sentinel_base: Sentinel ids count down from sentinel_base - 1.
        None (default) means len(tokenizer): correct for t5, whose
        <extra_id_*> tokens really are the trained sentinels, and for
        nguni-byt5, whose MAFT trained the top-of-vocab ids as sentinels.
        byt5 needs 259: the ByT5 paper reuses the final byte ids (258
        down) as sentinels, and byt5's embedding rows above 258 were
        never trained (using them gives worse-than-random loss).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
        sentinel_base: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.sentinel_base = sentinel_base if sentinel_base is not None else len(tokenizer)

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """
        Apply span corruption to a batch of pre-tokenized examples.

        :param examples: List of dicts, each with an "input_ids" key (list of token ids).
        :return: Dict of torch.Tensors: input_ids, labels, decoder_input_ids, attention_mask.
        """
        input_ids = np.array([example["input_ids"] for example in examples])
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([
            random_spans_noise_mask(expandend_input_length, self.noise_density, self.mean_noise_span_length)
            for _ in range(batch_size)
        ])
        labels_mask = ~mask_indices

        input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), self.sentinel_base)
        labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), self.sentinel_base)

        corrupted_input_ids = filter_input_ids(input_ids, input_ids_sentinel, self.tokenizer.eos_token_id)
        labels = filter_input_ids(input_ids, labels_sentinel, self.tokenizer.eos_token_id)

        if corrupted_input_ids.shape[-1] != self.input_length:
            raise ValueError(
                f"Corrupted input length {corrupted_input_ids.shape[-1]} does not match "
                f"expected input_length {self.input_length}. Check compute_input_and_target_lengths."
            )
        if labels.shape[-1] != self.target_length:
            raise ValueError(
                f"Labels length {labels.shape[-1]} does not match "
                f"expected target_length {self.target_length}. Check compute_input_and_target_lengths."
            )

        decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)

        batch = {
            "input_ids": torch.from_numpy(corrupted_input_ids).long(),
            "labels": torch.from_numpy(labels).long(),
            "decoder_input_ids": torch.from_numpy(decoder_input_ids).long(),
        }
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()

        return batch