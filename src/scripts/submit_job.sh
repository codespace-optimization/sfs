#!/bin/bash
#SBATCH --partition prod
#SBATCH --time=12:00:00
#SBATCH --job-name=llmcoding_experiment
#SBATCH --output=slurm/llmcoding_experiment_-%j.out
#SBATCH --error=slurm/llmcoding_experiment_-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jli@nec-labs.com
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# Load necessary modules or activate conda environment if needed
# module load Python/3.8.2

# Activate conda environment if needed
# conda init
conda activate llmcoding

# Run Python script with Hydra configuration files and additional arguments
# command to execute:
python -m src.main run_name=1.1-noscattering-10-gpt-3.5-turbomodel-0.0temp-10ntests-instructionpreamblemode-Falsefill_with_tests solver.params.model=gpt-3.5-turbo solver=tree dataset_name=apps200_competition-1 solver.params.strategy_library_name=mcts solver.params.number_of_tests=10 solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.branching_factor=2 solver.params.temperature=0.0 solver.params.testgen_method=simple solver.params.preamble_mode=none solver.params.do_fill_with_tests=False
