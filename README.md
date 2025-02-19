# Scattered Forest Search: Smart Code Space Optimization and Test-time Scaling with LLMs

This is the official repository for the paper [Scattered Forest Search: Smart Code Space Optimization and Test-time Scaling with LLMs](https://codespace-optimization.github.io/).

# Setup
1. clone this repository and cd into it.
```
cd sfs
```
2. In sfs, initialize a conda environment (conda init) with the requirements.
```
conda create --name codespace-opt python=3.12
conda activate codespace-opt
```
3. Install the requirements in codespace-opt environment using pip
```
pip install -r requirements.txt
```

4. Install pytorch based on your system configuration from https://pytorch.org/get-started/locally/

5. Add openAI key
```
export OPENAI_API_KEY='your-api-key-here'
```
You can also add this to your .bashrc or .bash_profile file as follows:
```
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```
This way you don't have to set the key every time you open a new terminal.

# Running the application
The application can be run using the following command from the `sfs` directory:
```
python run.py
```
The open a browser and go to `http://127.0.0.1:5000` to access the application. See [website](https://codespace-optimization.github.io/) for more details and video tutorial.

# Running experiments
Experiments can be run from the command line using the following command:
```
python -m src.main run_name=<run_name> solver=<solver> hydra.verbose=warning solver.params.<parameter>=<value> path.problems_set_path=<path>
```
Example:
```
python -m src.main run_name=sfs_0 solver=synthesis hydra.verbose=warning solver.params.strategy_library_name=mcts solver.params.num_seeds=3 solver.params.max_iters=7 paths.problem_set_path="data/original_problems/humaneval-py_hardest50.jsonl"
```
There are also scripts set up to batch run the experiments in the paper. These are located in the `src/scripts` directory, and require slurm to be set up on your system. 

# Experiment analysis
You can analyze the results of the experiments by following the instructions in the `notebooks/data_analysis.ipynb` notebook. This notebook will load the results of the experiments and generate the plots used in the paper.

# Data preprocessing
The repo contains readily available data for the experiments. However, if you want to preprocess your own data, you can follow the instructions in the `notebooks/data_preprocessing.ipynb` notebook. This notebook will preprocess the data and save it in the correct format for the experiments.