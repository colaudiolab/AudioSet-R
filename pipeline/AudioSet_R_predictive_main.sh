#!/bin/bash

echo "Activating clap environment..."
source "/etc/profile.d/conda.sh"
conda activate clap
python AudioSet_R_predictive_main.py

echo "Deactivating clap environment..."
conda deactivate

