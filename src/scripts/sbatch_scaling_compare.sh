DATASET_NAME="humaneval-py"
NUM_TESTS=6
TEMPERATURE=0.2 # lower temperature for reproducibility
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction
MODEL="gpt-3.5-turbo"
VERSION="0.8"
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode

# simple
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-simple-${RUN_NAME} solver=simple dataset_name=${DATASET_NAME} solver.params.temperature=${TEMPERATURE} solver.params.model=${MODEL}"

# random
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-40-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=40 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE}"

# reflexion
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-line-40-${RUN_NAME} solver.params.model=${MODEL} solver=reflexion dataset_name=${DATASET_NAME} solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=40 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE}"

# tree
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-tree-40-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.num_seeds=1 solver.params.max_iters=40"

# ours with 20-20 split (usually works better)
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-forestscout-20-20-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=20 solver.params.num_seeds=20 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE}"

# ours with 10-30 split
python3 src/scripts.run_submit_job.py "python -m src.main run_name=${VERSION}-forestscout-10-30-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=10 solver.params.num_seeds=30 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE}"
