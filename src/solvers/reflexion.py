from src.headers import CodingProblem, CodingProblemAndSolution
from ..headers import Solver
from ..generators.generator_types import Generator, ModelBase
from ..generators.factory import generator_factory, model_factory
from ..executors.executor_types import Executor
from ..executors.factory import executor_factory

from .base import BaseSolver

class ReflexionSolver(BaseSolver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self,  max_iters: int, *args, feedback_mode: str = "reflect", **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.feedback_mode = feedback_mode
        self.logger.info(f"ReflexionSolver initialized with feedback_mode: {feedback_mode}")

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        self.model.reset_stats()
        
        all_solutions: list[CodingProblemAndSolution] = []
        cur_solution = CodingProblemAndSolution.init_from_coding_problem(problem)
        # cur_solution.extra_kwargs['reflections'] = []
        # cur_solution.extra_kwargs['test_feedback'] = []
        cur_solution.solution = None

        # always generate new tests for now
        cur_solution.visible_tests = tuple(self.generator.generate_internal_tests(func_sig=problem.prompt, model=self.model, max_num_tests=self.number_of_tests, entry_point=problem.entry_point, temperature=self.temperature))

        # first attempt
        while cur_solution.solution is None:
            cur_solution.solution = self.generator.generate_func_impl(func_sig=problem.prompt, model=self.model, strategy="simple", entry_point=problem.entry_point, temperature=self.temperature)[0].code

            # record token cost
            cur_solution.extra_kwargs['int_output_tokens'] = self.model.output_tokens
            cur_solution.extra_kwargs['int_total_tokens'] = self.model.total_tokens
            cur_solution.extra_kwargs['int_total_requests'] = self.model.total_requests
        
        assert isinstance(cur_solution.solution, str)
        self.executor.execute(cur_solution)
        # cur_solution.extra_kwargs['test_feedback'].append(feedback)

        all_solutions.append(cur_solution)

        self.logger.info(f"Initial solution generated")
        self.logger.info(f"is passing: {cur_solution.is_passing}")
        # if solved, exit early
        if not cur_solution.is_passing:
            self.logger.info(f"Starting reflexion")
            # use self-reflection to iteratively improve
            cur_iter = 1
            while cur_iter < self.max_iters:
                # get self-reflection
                reflection = self.generator.self_reflection(
                    cur_solution.solution, cur_solution.feedback_string, self.model, temperature=self.temperature)
                # cur_solution.extra_kwargs['reflections'] += [reflection]

                prompt = cur_solution.prompt
                # prompt += "\n[unit test results from previous impl]:\n"
                # prompt += cur_feedback

                self.logger.info(f"Reflection: {reflection}")

                if self.feedback_mode == "test-results":
                # apply self-reflection in the next attempt
                    new_func_impl = self.generator.generate_func_impl(
                        func_sig=prompt,
                        model=self.model,
                        strategy="reflexion",
                        prev_func_impl=cur_solution.solution,
                        feedback=cur_solution.feedback_string,
                        self_reflection="No self-reflection",
                        entry_point=problem.entry_point,
                        temperature=self.temperature
                    )[0].code
                elif self.feedback_mode == "reflect":
                    new_func_impl = self.generator.generate_func_impl(
                        func_sig=prompt,
                        model=self.model,
                        strategy="reflexion",
                        prev_func_impl=cur_solution.solution,
                        feedback=cur_solution.feedback_string,
                        self_reflection=reflection,
                        entry_point=problem.entry_point,
                        temperature=self.temperature
                    )[0].code
                else:
                    raise ValueError(f"Invalid feedback mode {self.feedback_mode}")
                
                new_solution = CodingProblemAndSolution.init_from_coding_problem(cur_solution)
                new_solution.solution = new_func_impl
                # new_solution.extra_kwargs['test_feedback'] = []

                assert isinstance(new_solution.solution, str)

                # record token cost
                new_solution.extra_kwargs['int_output_tokens'] = self.model.output_tokens
                new_solution.extra_kwargs['int_total_tokens'] = self.model.total_tokens
                new_solution.extra_kwargs['int_total_requests'] = self.model.total_requests
                # check if all internal unit tests pass
                self.executor.execute(new_solution)
                # new_solution.extra_kwargs['test_feedback'].append(cur_feedback)
                all_solutions.append(new_solution)
                cur_solution = new_solution

                # if solved, check if it passes the real tests, exit early
                if cur_solution.is_passing or cur_iter == self.max_iters - 1:
                    break

                cur_iter += 1

        self.record_sample_stats_on_solution(all_solutions, cur_solution)        
        
        return cur_solution