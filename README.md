# Scattered Forest Search

This is the official repository for the paper Scattered Forest Search. 

# Setup
1. clone this repository and cd into it.
```
cd llm-coding
```
2. In llm coding, initialize a conda environment (conda init) with the requirements.
```
conda create --name llmcoding python=3.12
conda activate llmcoding
```
Ensure the llmcoding environment is also activated in the terminal for SearchTechniques (used in next step).

3. Install the requirements int llmcoding environment using pip
```
pip install -r requirements.txt
```

4. Install pytorch based on your system configuration from https://pytorch.org/get-started/locally/

5. Add openAI key
```
export OPENAI_API_KEY='your-api-key-here'
```

# Running search
```
python -m src.main run_name=<run_name> solver=<solver> hydra.verbose=warning solver.params.<parameter>=<value> path.problems_set_path=<path>
```
Example:

```
python -m src.main run_name=sfs_0 solver=synthesis hydra.verbose=warning solver.params.strategy_library_name=mcts solver.params.num_seeds=3 solver.params.max_iters=7 paths.problem_set_path="data/original_problems/humaneval-py_hardest50.jsonl"
```
