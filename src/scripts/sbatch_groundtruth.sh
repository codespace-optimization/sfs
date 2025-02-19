TEMPERATURE=0.0
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction
MODEL="gpt-3.5-turbo"
VERSION="0.8"

# Experiment configurations
CONFIGS=(
    "mbpp-py 6 True 7 3"
    "humaneval_164_groundtruth 3 False 7 3"
    "humaneval_164_groundtruth 10 False 7 3"
    "mbpp_groundtruth 1 False 7 3"
    "mbpp_groundtruth 10 False 7 3"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=' ' read -r DATASET_NAME NUM_TESTS FILL_WITH_TESTS MAX_ITERS NUM_SEEDS <<< "$CONFIG"
    
    RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${FILL_WITH_TESTS}fill_with_tests

    python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-${MAX_ITERS}_${NUM_SEEDS}-${RUN_NAME} \
        solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} \
        solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} \
        solver.params.max_iters=${MAX_ITERS} solver.params.num_seeds=${NUM_SEEDS} \
        solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 \
        solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} \
        solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"
done
