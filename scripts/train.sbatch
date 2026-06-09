#!/bin/bash
#SBATCH --job-name=pokenet_train
#SBATCH --output=slurm_logs/train-%j.out
#SBATCH --error=slurm_logs/train%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --account=dsci410_510

cd ~/pokenet-image-classifier
echo "Changed directory"

# Create and activate a virtual environment (read "Installation & Setup" under README)
source venv/bin/activate
echo "Activated virtual enviornment"

export POKEMON_DATASET_PATH=$HOME/.cache/kagglehub/datasets/noodulz/pokemon-dataset-1000/versions/2/pokemon-dataset-1000

python train_model.py --config configs/default.yaml
echo "Training complete"