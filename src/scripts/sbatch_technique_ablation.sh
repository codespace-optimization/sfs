DATASET_NAME="humaneval-py"

NUM_TESTS=6
TEMPERATURE=0.0 # 0.0 temp to ensure reproducibility
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction
MODEL="gpt-3.5-turbo"
VERSION="0.4"
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode

# ours (base)
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-forestscout-10-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE}"

# - forest 
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-10-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.num_seeds=1 solver.params.max_iters=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN}"

# - scouting
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-scatteredforest-10-${RUN_NAME} solver.params.model=${MODEL} solver=scatter dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=7 solver.params.num_seeds=3 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE}"

# - scattering
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-noscattering-10-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.branching_factor=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none"
