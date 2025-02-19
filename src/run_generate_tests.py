import json
import os
from .generators.py_generate import PyGenerator
from .generators.factory import model_factory

def load_jsonl_file(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    return data

def append_to_jsonl_file(entry, filepath):
    # Check if file exists and open in append mode, or create a new file if it does not exist
    with open(filepath, 'a') as file:
        file.write(json.dumps(entry) + '\n')

def main(input_problem_set_path, output_test_path, model):
    # Load problems from the given JSONL input file
    input_problems = load_jsonl_file(input_problem_set_path)
    
    print(input_problems[0])
    # Load existing data from the output JSONL file
    output_problems = load_jsonl_file(output_test_path)

    # Convert output_problems to a set of task_ids for quick lookup
    output_task_ids = {problem['task_id'] for problem in output_problems}

    # Filter out problems that already have tests
    new_problems = [problem for problem in input_problems if problem['task_id'] not in output_task_ids]
    
    if not new_problems:
        print("All tests already generated.")
        return

    # Process each new problem
    for problem in new_problems:
        # Extract function signature and description
        func_sig = problem['prompt']
        entry_point = problem['entry_point']
        
        # Generate unit tests; assuming 5 tests are to be generated for each problem
        generated_tests, _ = PyGenerator.generate_seed_tests(func_sig=func_sig, model=model, entry_point=entry_point, num_tests=5)
        
        # Append generated tests to the problem entry under 'given_tests'
        problem['given_tests'] = generated_tests

        # Append this updated problem to the output file
        append_to_jsonl_file(problem, output_test_path)

# Assuming you have a model object ready to use
# model = model_factory("gpt-3.5-turbo-0613")
model = model_factory("gpt-4")

# Paths to the input and output files
# input_problem_set_path = 'input_data/humaneval/dataset/probs.jsonl'
# output_test_path = 'input_data/humaneval/test/test_generated_1.jsonl'
data_set_name = 'mbpp'
# data_set_name = 'humaneval'
input_problem_set_path = f'input_data/{data_set_name}/dataset/probs.jsonl'
output_test_path = f'input_data/{data_set_name}/test/test_generated_gpt4.jsonl'

# Run the script
try:
    main(input_problem_set_path, output_test_path, model)
except FileExistsError as e:
    print(e)