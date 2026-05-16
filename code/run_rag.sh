#!/bin/bash
#SBATCH --job-name=erasmus-rag
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/d/hpc/projects/onj_fri/hungover_pandas/logs/rag_%j.out
#SBATCH --error=/d/hpc/projects/onj_fri/hungover_pandas/logs/rag_%j.err

# Activate conda env
source /cvmfs/sling.si/modules/el7/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate nlp

cd /d/hpc/projects/onj_fri/hungover_pandas

echo "=== Job info ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU(s)   : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

# Run predefined test questions
python code/rag.py --mode test

echo "Done."