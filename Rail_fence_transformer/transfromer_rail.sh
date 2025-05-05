#!/bin/bash

## specify the job and project name
#SBATCH --job-name=transformer-2
#SBATCH --account=zas31
## specify the required resources
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mem=128000
#SBATCH --time=0-06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zas31@mail.aub.edu


# Load Python
module purge
module load python/ai-4


# Run the script
python transformer_rail_fence.py
