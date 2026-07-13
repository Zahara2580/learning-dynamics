#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --job-name="inspect-byt5-tied-weights"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/inspect_byt5_tied_weights_%j.log
#SBATCH --error=logs/inspect_byt5_tied_weights_%j.log
#
# One-off diagnostic (no training) to confirm the "tied weights mapping...
# but both are present in the checkpoints with different values, so we
# will NOT tie them" warning seen in the bf16/fp32 fitcheck runs. Both
# runs gave loss ~12.4/batch (eval_loss 8.34), well above ln(384)~5.95
# random-guess baseline, at step 0 before any training - ruled out bf16
# vs fp32 as the cause (both dtypes gave the same bad loss at matched
# batch size). This checks whether google/byt5-large's shared.weight
# (input embedding) and lm_head.weight (output projection) genuinely
# hold different values despite tie_word_embeddings=True in its config.

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

uv run python3 -c "
from transformers import AutoConfig, AutoModelForSeq2SeqLM
import torch

config = AutoConfig.from_pretrained('google/byt5-large')
print('tie_word_embeddings:', config.tie_word_embeddings)

model = AutoModelForSeq2SeqLM.from_pretrained('google/byt5-large', torch_dtype=torch.float32)
shared = model.get_input_embeddings().weight
lm_head = model.lm_head.weight

print('shared shape:', shared.shape, 'lm_head shape:', lm_head.shape)
print('same tensor object (data_ptr match):', shared.data_ptr() == lm_head.data_ptr())
print('allclose:', torch.allclose(shared, lm_head))
print('shared norm:', shared.norm().item(), 'lm_head norm:', lm_head.norm().item())
print('max abs diff:', (shared - lm_head).abs().max().item())
"
