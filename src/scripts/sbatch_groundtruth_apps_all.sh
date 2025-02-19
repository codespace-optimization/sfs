TEMPERATURE=0.0
TEST_GEN="simple" # simple, cot
PREAMBLE_MODE="instruction" # role, none, instruction
MODEL="gpt-3.5-turbo"
# MODEL="gpt-4o-mini"
VERSION="1.1"
FILL_WITH_TESTS=False
DATASET_NAME="apps200_competition-1"
NUM_TESTS=10
RUN_NAME=${MODEL}model-${TEMPERATURE}temp-${NUM_TESTS}ntests-${PREAMBLE_MODE}preamblemode-${FILL_WITH_TESTS}fill_with_tests

# simple
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-simple-${RUN_NAME} solver=simple dataset_name=${DATASET_NAME} solver.params.temperature=${TEMPERATURE} solver.params.model=${MODEL}"

# best of N
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-random-${RUN_NAME} solver=random dataset_name=${DATASET_NAME} solver.params.num_seeds=10 solver.params.number_of_tests=${NUM_TESTS} solver.params.model=${MODEL} solver.params.preamble_mode=none solver.params.temperature=${TEMPERATURE} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# line (reflexion)
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-line-10-${RUN_NAME} solver.params.model=${MODEL} solver=reflexion dataset_name=${DATASET_NAME} solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=10 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# tree
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-tree-10-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.num_seeds=1 solver.params.max_iters=10 solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# selfrepair
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-selfrepair-10-${RUN_NAME} solver.params.model=${MODEL} solver=selfrepair dataset_name=${DATASET_NAME} solver.params.num_feedback_per_seed=2 solver.params.num_repairs_per_feedback=2 solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.num_seeds=2 solver.params.max_iters=10 solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# PUCT tree
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-treepuct-10-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.num_seeds=1 solver.params.max_iters=10 solver.params.is_puct=True solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# genetic
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-genetic-10-${RUN_NAME} solver.params.model=${MODEL} solver=genetic dataset_name=${DATASET_NAME} solver.params.number_of_tests=${NUM_TESTS} solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.num_seeds=2 solver.params.max_iters=10 solver.params.num_islands=2 solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# sfs
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-7_3-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=7 solver.params.num_seeds=3 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# sfs-5:5
# python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-synthesis-5_5-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# - forest 
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-noforest-10-${RUN_NAME} solver.params.model=${MODEL} solver=synthesis dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.num_seeds=1 solver.params.max_iters=10 solver.params.preamble_mode=${PREAMBLE_MODE} solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# - scouting
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-noscout-10-${RUN_NAME} solver.params.model=${MODEL} solver=scatter dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=7 solver.params.num_seeds=3 solver.params.num_ideas_per_strategy=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"

# - scattering
python3 src/scripts/run_submit_job.py "python -m src.main run_name=${VERSION}-noscattering-10-${RUN_NAME} solver.params.model=${MODEL} solver=tree dataset_name=${DATASET_NAME} solver.params.strategy_library_name=mcts solver.params.number_of_tests=${NUM_TESTS} solver.params.max_iters=5 solver.params.num_seeds=5 solver.params.branching_factor=2 solver.params.temperature=${TEMPERATURE} solver.params.testgen_method=${TEST_GEN} solver.params.preamble_mode=none solver.params.do_fill_with_tests=${FILL_WITH_TESTS}"
