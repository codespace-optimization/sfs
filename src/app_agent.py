from .solvers.synthesis import SynthesisSolver
from .generators.model import GPTChat
from .headers import CodingProblem, CodingProblemAndSolution

from dataclasses import dataclass

@dataclass
class LLMAgentOutput:
    output: str
    num_tests_generated: int
    num_tests_passed: int
    num_revisions: int

def process_with_llm_agent(prompt: str, model: str, api_key: str, num_tries:int = 10) -> LLMAgentOutput:
    num_seeds = num_tries // 2
    num_iters = num_tries - num_seeds

    gpt_model = GPTChat(model_name=model, api_key=api_key)
    # create a synthesis solver
    solver = SynthesisSolver(max_iters=num_iters, model=gpt_model, num_seeds=num_seeds, generator="python", executor="pyquick")

    # create coding problem
    problem = CodingProblem(task_id="main", prompt=prompt, entry_point="solution")

    # solve the problem
    solution = solver.solve(problem, results_path="")

    # return the solution
    out = solution.solution
    num_tests_generated = len(solution.test_results)
    num_tests_passed = sum(solution.test_results)
    num_revisions = solution.extra_kwargs['num_solutions']
    if out is None:
        return LLMAgentOutput(output="No solution found", num_tests_generated=num_tests_generated, num_tests_passed=num_tests_passed, num_revisions=num_revisions)
    else:
        return LLMAgentOutput(output=out, num_tests_generated=num_tests_generated, num_tests_passed=num_tests_passed, num_revisions=num_revisions)