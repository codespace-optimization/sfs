import os
import json
import random
import argparse

VERSION = "0.1"

def convert_input_output(input_output: str) -> str:
    '''
    Converts the input or output string to a Python string representation.

    We do this by replacing all occurances of \n with \\n
    '''
    return input_output.replace("\n", "\\n")

def generate_dataset(directory, add_visible_tests=True):
    '''
    Generates a dataset from a directory containing the following files:
    - question.txt: a text file containing the prompt for the task
    - input_output.json: a JSON file containing the inputs and outputs for the task
    - metadata.json: a JSON file containing metadata for the task
    - solutions.json: a JSON file containing a list of solutions for the task
    '''
    question_file = os.path.join(directory, "question.txt")
    with open(question_file, 'r') as f:
        prompt = f.read().strip()
    task_id = f"apps_{os.path.basename(directory)}"
    entry_point = "solution"
    function_name = "solution"
    
    input_output_file = os.path.join(directory, "input_output.json")
    with open(input_output_file, 'r') as f:
        data = json.load(f)
    inputs = data['inputs']
    outputs = data['outputs']
    
    visible_tests = []
    tests = [f"    assert candidate('{convert_input_output(input_data)}') == '{convert_input_output(outputs[i])}'" for i, input_data in enumerate(inputs)]
    test = f"def check(candidate):\n" + "\n".join(tests) + f"\n\ndef test_check():\n    check({function_name})\n\ntest_check()\n"
    if add_visible_tests:
        visible_tests = [f"assert candidate('{convert_input_output(input_data)}') == '{convert_input_output(outputs[i])}'" for i, input_data in enumerate(inputs)]

    # read the metadata file
    metadata_file = os.path.join(directory, "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # we will need to modify the prompt a bit so that it is easier for the model to understand
    # we will do so by adding a function signature to the prompt, placing the original prompt in a comment, and adding 2 test cases to the docstring
    prompt = f"""def {function_name}(stdin: str) -> str:\n    '''\n    {prompt}\n\n    # Test Cases\n    >>> {visible_tests[0]}\n    >>> {visible_tests[1]}\n    '''\n    pass"""

    dataset = {
        "task_id": task_id,
        "prompt": prompt,
        "entry_point": entry_point,
        "test": test,
        "visible_tests": visible_tests,
    } | metadata

    # read the solutions file
    solutions_file = os.path.join(directory, "solutions.json")
    # check if solutions file exists. if not then do not add canonical solution
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
        solution = solutions[0]
        dataset['canonical_solution'] = solution
    return dataset

def process_all_subdirectories(base_directory):
    introductory_level = []
    competition_level = []
    interview_level = []
    for subdirectory in os.listdir(base_directory):
        subdirectory_path = os.path.join(base_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            try:
                dataset = generate_dataset(subdirectory_path)
                if dataset['difficulty'] == 'introductory':
                    introductory_level.append(dataset)
                elif dataset['difficulty'] == 'competition':
                    competition_level.append(dataset)
                elif dataset['difficulty'] == 'interview':
                    interview_level.append(dataset)
                else:
                    raise ValueError(f"Unknown difficulty level: {dataset['difficulty']}")
                print(f"Processed {subdirectory}")
            except Exception as e:
                print(f"Failed to process {subdirectory}: {e}")
    
    return introductory_level, competition_level, interview_level

def sample_datasets(datasets, num_samples: int, seed: int):
    random.seed(seed)
    return random.sample(datasets, num_samples)

def process_apps_raw_data(source_directory, output_directory, num_sample: int = 200, seed: int = 42):
    introductory_level, competition_level, interview_level = process_all_subdirectories(source_directory)

    introductory_file = os.path.join(output_directory, f"apps_introductory-{VERSION}.jsonl")
    with open(introductory_file, 'w') as f:
        for dataset in introductory_level:
            json.dump(dataset, f)
            f.write('\n')

    print(f"Processed {len(introductory_level)} introductory datasets and saved to {introductory_file}")

    competition_file = os.path.join(output_directory, f"apps_competition-{VERSION}.jsonl")
    with open(competition_file, 'w') as f:
        for dataset in competition_level:
            json.dump(dataset, f)
            f.write('\n')

    # additionally sample some datasets from competition level
    competition_level_sampled = sample_datasets(competition_level, num_sample, seed)
    competition_sample_file = os.path.join(output_directory, f"apps_competition_{num_sample}-{VERSION}.jsonl")
    with open(competition_sample_file, 'w') as f:
        for dataset in competition_level_sampled:
            json.dump(dataset, f)
            f.write('\n')

    print(f"Processed {len(competition_level)} competition datasets and saved to {competition_file}")

    interview_file = os.path.join(output_directory, f"apps_interview-{VERSION}.jsonl")
    with open(interview_file, 'w') as f:
        for dataset in interview_level:
            json.dump(dataset, f)
            f.write('\n')

    print(f"Processed {len(interview_level)} interview datasets and saved to {interview_file}")

    print("Finished processing all datasets")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process datasets from a source directory and save to an output directory.")
    parser.add_argument(
        "--source_directory",
        type=str,
        default=os.getcwd(),
        help="Path to the source directory containing raw data (default: current working directory)",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=os.getcwd(),
        help="Path to the output directory to save processed datasets (default: current working directory)",
    )
    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.source_directory):
        print(f"Error: Source directory '{args.source_directory}' does not exist.")
        return

    if not os.path.isdir(args.output_directory):
        print(f"Output directory '{args.output_directory}' does not exist. Creating it...")
        os.makedirs(args.output_directory)

    # Process the datasets
    process_apps_raw_data(args.source_directory, args.output_directory)

if __name__ == "__main__":
    main()