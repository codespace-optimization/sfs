from src.headers import CodingProblem, CodingProblemAndSolution
from .base import BaseSolver

from typing import Any

class SelfRepairSolver(BaseSolver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self, max_iters: int, *args, num_feedback_per_seed: int = 2, num_repairs_per_feedback: int = 2, codegen_method: str = "outcome", with_accummulated_context: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.codegen_method = codegen_method
        self.with_accummulated_context = with_accummulated_context
        self.num_feedback_per_seed = num_feedback_per_seed
        self.num_repairs_per_feedback = num_repairs_per_feedback

    def generate_strategy(self, old_strategy: CodingProblemAndSolution, self_reflection: str) -> CodingProblemAndSolution:
        cur_func_impl = old_strategy.solution
        func_sig = old_strategy.prompt
        feedback = old_strategy.feedback_string

        assert cur_func_impl is not None
        response = self.generator.generate_func_impl(func_sig=func_sig, prev_func_impl=cur_func_impl, model=self.model, strategy="reflexion", feedback=feedback, self_reflection=self_reflection, entry_point=old_strategy.entry_point, num_comps=1, temperature=self.temperature, )[0]

        new_strategy = CodingProblemAndSolution.init_from_coding_problem(old_strategy)
        new_strategy.solution = response.code
        new_strategy.generation = old_strategy.generation + 1
        new_strategy.extra_kwargs['logprob'] = response.logprob
        # record token cost
        new_strategy.extra_kwargs['int_output_tokens'] = self.model.output_tokens
        new_strategy.extra_kwargs['int_total_tokens'] = self.model.total_tokens
        new_strategy.extra_kwargs['int_total_requests'] = self.model.total_requests

        # record self reflection
        new_strategy.extra_kwargs['self_reflection'] = self_reflection
        # print out new strategy
        self.logger.debug(f"New strategy implementation: {new_strategy.solution}")
        return new_strategy
    
    def generate_self_reflection(self, old_strategy: CodingProblemAndSolution) -> str:
        cur_func_impl = old_strategy.solution
        assert cur_func_impl is not None
        func_sig = old_strategy.prompt
        feedback = old_strategy.feedback_string

        # use the debug messages to generate new ideas
        self_reflection = self.generator.generate_self_reflection(func=cur_func_impl, feedback=feedback, model=self.model, temperature=self.temperature)
        return self_reflection

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        terminate_process: bool = False
        self.model.reset_stats()
        # record the list of solutions in order they were generated
        all_solutions: list[CodingProblemAndSolution] = []
        # generate visible tests
        self.fill_with_tests(problem, self.number_of_tests)
        # generate seed functions
        seed_solutions = self.generate_seed_functions(num_seeds=self.num_seeds, problem=problem)
        # execute the seed functions to record their scores
        for seed_solution in seed_solutions:
            # self.executor.execute(seed_solution)
            all_solutions.append(seed_solution)
            if all(seed_solution.test_results):
                terminate_process = True
                break

            # also generate self.num_feedback_per_seed reflections for each seed
            for i in range(self.num_feedback_per_seed):
                self_reflection = self.generate_self_reflection(seed_solution)
                
                # generate self.num_repairs_per_feedback new solutions for each reflection
                for j in range(self.num_repairs_per_feedback):
                    # generate the new solution
                    new_solution = self.generate_strategy(seed_solution, self_reflection)
                        
                    # evaluate the new solution
                    self.executor.execute(new_solution)
                    all_solutions.append(new_solution)

                    # if the new_solution passes all the tests, break
                    if all(new_solution.test_results) or len(all_solutions) >= self.max_iters:
                        terminate_process = True
                        break

                if terminate_process:
                    break

            if terminate_process:
                break

        best_solution = max(all_solutions, key=lambda solution: sum(solution.test_results))

        self.record_sample_stats_on_solution(all_solutions, best_solution)
        return best_solution