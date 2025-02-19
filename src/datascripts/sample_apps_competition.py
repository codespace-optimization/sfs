import json
import random

MAX_INPUT_LENGTH = 2000

def parse_assert_lines(test_string: str) -> list[str]:
    '''
    Parses out every line that starts with "assert" from the test string. Removes any leading whitespace.
    Then convert `candidate` to `solution` in the visible test cases.
    '''
    asserts = [line.strip() for line in test_string.split('\n') if line.strip().startswith("assert")]
    # replace 'candidate' with 'solution' in asserts
    asserts = [assert_line.replace('candidate', 'solution') for assert_line in asserts]
    return asserts

def sample_and_modify_jsonl(input_file_path, output_file_path, n: int, seed: int = 42):
    total_skipped = 0
    # Read the JSONL file
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    # Modify each line as needed
    modified_lines = []
    for line in lines:
        data = json.loads(line)
        if data["meta_data"]["difficulty"] == "competition":
            # Perform any modifications to the data here
            # modify data by adding `check({entry_point})` at the end of the key "test"
            data['test'] = data['test'] + f"\ncheck({data['entry_point']})"
            data['visible_tests'] = parse_assert_lines(data['test'])
            # add two visible tests to 'prompt' as docstring right after 'def solution(stdin: str) -> str:\n' (the function signature, which is at the end of the prompt)
            example_test_cases = f"    '''\n    Example inputs and outputs:\n    >>> {data['visible_tests'][0]}\n    >>> {data['visible_tests'][1]}\n    '''"
            # replace 'candidate' with 'solution' in example test cases
            example_test_cases = example_test_cases.replace('candidate', 'solution')
            data['prompt'] = data['prompt'] + example_test_cases
            
            # check if prompt size is less than MAX_INPUT_LENGTH
            if len(data['prompt']) < MAX_INPUT_LENGTH:
                modified_lines.append(json.dumps(data))
            else:
                print(f"Prompt size exceeds the maximum limit for the problem: {data['task_id']}. Skipping this problem.")
                total_skipped += 1

    # sample n lines
    random.seed(seed)
    sampled_lines = random.sample(modified_lines, n)

    # Write the modified lines to a new JSONL file
    with open(output_file_path, 'w') as output_file:
        for line in sampled_lines:
            output_file.write(line + '\n')

    print(f"Data has been modified and saved to {output_file_path}.")
    print(f"Total problems skipped: {total_skipped}")

# File path to the original JSON file
input_file_path = 'data/old_original_problems/APPS_zeroshot_for_code_generation.jsonl'
output_file_path = 'data/original_problems/apps200_competition-1.jsonl'

# Number of lines to sample
n = 200

# Sample and modify the JSONL file
sample_and_modify_jsonl(input_file_path, output_file_path, n, seed=42)