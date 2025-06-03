#!/bin/bash

echo "Activating qwen environment..."
source "/etc/profile.d/conda.sh"
conda activate qwen
python AudioSet_R_qwen_main.py

echo "Deactivating qwen environment..."
conda deactivate


