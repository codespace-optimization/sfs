DATASET_NAME="humaneval-py"
NUM_TESTS=6
TEMPERATURE=0.0
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction, jabberwocky
MODEL="gpt-4o-mini"
VERSION="0.8"
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode

# tree puct
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-tree-puct-10-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.num_seeds=1 solver.params.max_iters=10 solver.params.is_puct=True"

# ours puct
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-forestscout-puct-10-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.is_puct=True"
