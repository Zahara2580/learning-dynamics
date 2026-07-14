#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --job-name="check-datacollator-import"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/check_datacollator_import_%j.log
#SBATCH --error=logs/check_datacollator_import_%j.log
#
# Empirically checks whether DataCollatorForT5MLM (or any T5 span-corruption
# collator) is importable from the installed transformers package, rather
# than trusting any claim about it - tries every plausible import path and
# reports exactly what succeeds or fails. CPU-only, no GPU needed, no model
# downloads, just import attempts against the installed library.

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

uv run python3 - << 'EOF'
import transformers
print(f"Installed transformers version: {transformers.__version__}")
print(f"Installed at: {transformers.__file__}")
print()

# Every plausible import path someone might guess or have seen referenced.
attempts = [
    ("from transformers import DataCollatorForT5MLM", "DataCollatorForT5MLM"),
    ("from transformers.data.data_collator import DataCollatorForT5MLM", "DataCollatorForT5MLM"),
    ("from transformers.data import DataCollatorForT5MLM", "DataCollatorForT5MLM"),
    ("from transformers.models.t5 import DataCollatorForT5MLM", "DataCollatorForT5MLM"),
    ("from transformers.models.t5.modeling_t5 import DataCollatorForT5MLM", "DataCollatorForT5MLM"),
]

for import_stmt, name in attempts:
    try:
        exec(import_stmt)
        print(f"SUCCESS: {import_stmt}")
    except ImportError as e:
        print(f"FAILED:  {import_stmt}\n         -> {e}")
    except Exception as e:
        print(f"FAILED (other error):  {import_stmt}\n         -> {type(e).__name__}: {e}")

print()
print("=== What data collators ARE actually exported by transformers ===")
import transformers
data_collator_names = sorted(n for n in dir(transformers) if "Collator" in n or "collator" in n)
for n in data_collator_names:
    print(f"  transformers.{n}")

print()
print("=== Searching the installed package source for 'T5MLM' or 'span_corruption' or 'noise_span' ===")
import subprocess
import os
pkg_dir = os.path.dirname(transformers.__file__)
result = subprocess.run(
    ["grep", "-rl", "-e", "T5MLM", "-e", "span_corruption", "-e", "noise_span_length"],
    cwd=pkg_dir,
    capture_output=True, text=True,
)
if result.stdout.strip():
    print("Found references in these installed files:")
    print(result.stdout)
else:
    print("No references to T5MLM / span_corruption / noise_span_length found anywhere in the installed transformers package.")
EOF
