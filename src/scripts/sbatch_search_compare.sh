# DATASET_NAME="leetcode-20240121-Jul-mod"
# DATASET_NAME="codecontests"
# DATASET_NAME="apps200"
# DATASET_NAME="mbpp-py"
# DATASET_NAME="humaneval-py"
DATASET_NAME="demo3"
# DATASET_NAME="humaneval_plus"
# DATASET_NAME="mbpp_plus"

NUM_TESTS=6
TEMPERATURE=0.2 # lower temperature for reproducibility
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction
MODEL="gpt-3.5-turbo"
VERSION="0.8"
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode

# simple
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-simple-${RUN_NAME} solver=simple dataset_name=${DATASET_NAME} solver.params.temperature=${TEMPERATURE} solver.params.model=${MODEL}"

# best of N
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE}"

# line search (sequential refinement)
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-line-10-${RUN_NAME} solver.params.model=${MODEL} solver=reflexion dataset_name=${DATASET_NAME} solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=10 solver.params.temperature=${TEMPERATURE} solver.params.preamble_mode=${PREAMBLE_MODE}"

# tree search (MCTS)
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-tree-10-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE}      solver.params.preamble_mode=none solver.params.num_seeds=1 solver.params.max_iters=10"

# genetic algorithm
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-genetic-10-${RUN_NAME} solver.params.model=${MODEL} solver=genetic dataset_name=${DATASET_NAME} solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.preamble_mode=none solver.params.num_seeds=2 solver.params.max_iters=10 solver.params.num_islands=2"

# ours (SFS)
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-forestscout-10-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.preamble_mode=${PREAMBLE_MODE}"