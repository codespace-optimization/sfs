NUM_TESTS=6
TEMPERATURE=0.8
TEST_GEN="simple" # simple, cot
MODEL="gpt-3.5-turbo"
VERSION="0.4"
DO_FILL_WITH_TESTS=False

# DATASET_NAME="mbpp_groundtruth"
DATASET_NAME="humaneval_164_groundtruth"

PREAMBLE_MODE="none"
RUN_NAME=${MODEL}model-${PREAMBLE_MODE}preamblemode-${NUM_TESTS}ntests-${TEMPERATURE}temp
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${DO_FILL_WITH_TESTS}"

PREAMBLE_MODE="instruction" 
RUN_NAME=${MODEL}model-${PREAMBLE_MODE}preamblemode-${NUM_TESTS}ntests-${TEMPERATURE}temp
python3 src/scripts.run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${DO_FILL_WITH_TESTS}"

PREAMBLE_MODE="role" 
RUN_NAME=${MODEL}model-${PREAMBLE_MODE}preamblemode-${NUM_TESTS}ntests-${TEMPERATURE}temp
python3 src/scripts.run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${DO_FILL_WITH_TESTS}"

PREAMBLE_MODE="jabberwocky" 
RUN_NAME=${MODEL}model-${PREAMBLE_MODE}preamblemode-${NUM_TESTS}ntests-${TEMPERATURE}temp
python3 src/scripts.run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${DO_FILL_WITH_TESTS}"

# ablation to check if higher temperature helps
PREAMBLE_MODE="none"
TEMPERATURE=1.2
RUN_NAME=${MODEL}model-${PREAMBLE_MODE}preamblemode-${NUM_TESTS}ntests-${TEMPERATURE}temp
python3 src/scripts.run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${DO_FILL_WITH_TESTS}"
