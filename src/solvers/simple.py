from src.headers import CodingProblem, CodingProblemAndSolution
from ..generators.generator_types import Generator
from ..generators.model import ModelBase
from ..headers import Solver
from ..generators.factory import generator_factory, model_factory

from typing import Union

class SimpleSolver(Solver):
    '''
    Simply queries the LLM with the prompt and returns the first completion
    '''
    def __init__(self, generator: Generator | str, model: ModelBase | str, temperature: float = 0.0):
        if isinstance(generator, str):
            generator = generator_factory(generator)
        if isinstance(model, str):
            model = model_factory(model)
        super().__init__()
        self.generator = generator
        self.model = model
        self.temperature = temperature

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        self.model.reset_stats()
        generated_functions = self.generator.generate_func_impl(func_sig=problem.prompt, model=self.model, strategy="simple", entry_point=problem.entry_point, temperature=self.temperature)
        first_solution = generated_functions[0].code
        coding_problem_and_solution = CodingProblemAndSolution.init_from_coding_problem(problem)
        coding_problem_and_solution.solution = first_solution
        coding_problem_and_solution.extra_kwargs['total_tokens'] = self.model.total_tokens
        coding_problem_and_solution.extra_kwargs['output_tokens'] = self.model.output_tokens
        coding_problem_and_solution.extra_kwargs['total_requests'] = self.model.total_requests
        return coding_problem_and_solution