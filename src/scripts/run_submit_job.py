import sys
import subprocess  # Add this line

def submit_job(new_command_line):
    # Path to submit_job.sh
    submit_script_path = "src/scripts/submit_job.sh"

    # Read the content of submit_job.sh
    with open(submit_script_path, 'r') as f:
        script_lines = f.readlines()

    # Find the index of the line containing 'command_to_execute:'
    command_line_index = None
    for i, line in enumerate(script_lines):
        if line.strip().startswith("# command to execute:"):
            command_line_index = i
            break

    if command_line_index is None:
        print("Error: 'command_to_execute' line not found in the script")
        return

    # Replace the line after 'command_to_execute:' with the new command line
    script_lines[command_line_index + 1] = f"{new_command_line}\n"

    # Write the updated content back to submit_job.sh
    with open(submit_script_path, 'w') as f:
        f.writelines(script_lines)

    # Submit the job using sbatch
    subprocess.run(['sbatch', submit_script_path], check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python submit_job.py <new_command_line>")
        sys.exit(1)
    
    new_command_line = sys.argv[1]
    submit_job(new_command_line)
