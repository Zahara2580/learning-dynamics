#!/bin/bash
set -e
echo "Pulling latest code"
git pull
git log -1
echo "Loading environment variables"
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Loading Python module"
module load python/miniconda3-py3.12
echo "Setting up Hugging Face cache"
export HF_HOME=/scratch/rmdrak003/hf
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"
echo "Activating virtual environment"
source .venv/bin/activate
uv sync