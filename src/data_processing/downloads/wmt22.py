"""Load or download WMT22 English–isiXhosa training data."""

from datasets import load_dataset

WMT22_DATASET = "allenai/wmt22_african"
WMT22_CONFIG = "eng-xho"


def load_wmt22_train():
    """Load the full training split using the Hugging Face cache."""
    return load_dataset(WMT22_DATASET, WMT22_CONFIG, split="train")


def main() -> None:
    """Download the training split into the Hugging Face cache."""
    dataset = load_wmt22_train()
    print(f"Loaded {len(dataset)} WMT22 training pairs")


if __name__ == "__main__":
    main()
