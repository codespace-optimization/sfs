FILTER_NUM_TESTS=3
TEMPERATURE=0.0
TEST_GEN="simple" # simple, cot
FILTER_MODE="uniform" # uniform, passing_solutions
PREAMBLE_MODE="instruction" # role, none, instruction
IS_BENCHMARK_SETUP=False # same setup as sota or not
MODEL="gpt-3.5-turbo"
VERSION="0.4"
FILL_WITH_TESTS=False


DATASET_NAME="humaneval_164_groundtruth"

# for debugging
# python -m src.main run_name=${VERSION}-test1_synthesis-mcts-5 solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP}

NUM_TESTS=3
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${FILL_WITH_TESTS}fill_with_tests

# random
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=40 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"
# python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE}

# synthesis
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# synthesis
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-2020-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=20 solver.params.num_seeds=20 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

NUM_TESTS=10
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${FILL_WITH_TESTS}fill_with_tests

# random
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=40 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"
# python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE}

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

DATASET_NAME="mbpp_groundtruth"
NUM_TESTS=1
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${FILL_WITH_TESTS}fill_with_tests

# random
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=40 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"
# python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE}

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

NUM_TESTS=10
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${FILL_WITH_TESTS}fill_with_tests

# random
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=40 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"
# python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE}

# synthesis
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-40-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=30 solver.params.num_seeds=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.filter_num_tests=${FILTER_NUM_TESTS} solver.params.filter_weighting_mode=${FILTER_MODE} solver.params.is_lats_setup=${IS_BENCHMARK_SETUP} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"