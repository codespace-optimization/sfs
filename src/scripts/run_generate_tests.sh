#!/bin/bash
#SBATCH --partition prod
#SBATCH --time=12:00:00
#SBATCH --job-name=ldb_experiment
#SBATCH --output=slurm/ldb_experiment_-%j.out
#SBATCH --error=slurm/ldb_experiment_-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jli@nec-labs.com
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Activate conda environment if needed
# conda init
conda activate ldb

python3 programming/run_generate_tests.py 