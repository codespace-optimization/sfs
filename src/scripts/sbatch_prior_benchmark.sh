# DATASET_NAME="leetcode-20240121-Jul-mod"
# DATASET_NAME="codecontests"
# DATASET_NAME="apps200"
NUM_TESTS=6
FILTER_NUM_TESTS=6
TEMPERATURE=0.0
TEST_GEN="simple" # simple, cot
FILTER_MODE="uniform" # uniform, passing_solutions
PREAMBLE_MODE="instruction" # role, none, instruction
IS_BENCHMARK_SETUP=False # same setup as sota or not
MODEL="gpt-3.5-turbo"
VERSION="0.4"
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${IS_BENCHMARK_SETUP}benchmarksetup

DATASET_NAME="humaneval-py"

# tree
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-tree-40-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=40 solver.params.num_seeds=1 solver.params.branching_factor=5 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE}"

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP}"

DATASET_NAME="mbpp-py"

# tree
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-tree-40-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=40 solver.params.num_seeds=1 solver.params.branching_factor=5 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE}"

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP}"

# lats setup
IS_BENCHMARK_SETUP=True
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${IS_BENCHMARK_SETUP}benchmarksetup

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP}"
python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP}

DATASET_NAME="humaneval-py"

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP}"
