DATASET_NAME="humaneval-py"
NUM_TESTS=6
TEMPERATURE=0.0
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction

VERSION="0.8"

# List of models to run
MODELS=("gpt-3.5-turbo" "gpt-4o-mini" "gpt-4o" "claude-3-5-sonnet-20240620" "claude-3-haiku-20240307" "mistral7b-instruct" "llama8b-instruct")

for MODEL in "${MODELS[@]}"; do
    RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode
    python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-forestscout-10-${RUN_NAME} \
        solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} \
        solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} \
        solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.num_ideas_per_strategy=2 \
        solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} \
        solver.params.preamble_mode=${PREAMBLE_MODE}"
done
