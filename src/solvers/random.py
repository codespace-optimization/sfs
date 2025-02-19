from ..headers import CodingProblem, CodingProblemAndSolution, Solver
from .preambles import PREAMBLES_ROLE_GPT_GENERATED
from .base import BaseSolver

import numpy as np

class RandomSolver(BaseSolver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self, *args, stop_when_solved: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_when_solved = stop_when_solved

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        self.model.reset_stats()
        
        if self.do_fill_with_tests:
            # fill with tests
            self.fill_with_tests(problem, self.number_of_tests)
        else:
            problem.visible_tests = problem.visible_tests[:self.number_of_tests]

        seed_solutions = self.generate_seed_functions(self.num_seeds, problem, stop_if_solved=self.stop_when_solved)

        # assert len(seed_solutions) == self.num_seeds

        # execute the seed functions to record their scores
        # for seed_function in seed_solutions:
        #     self.executor.execute(seed_function)

        # best solution is one that has the highest score (sum(test_results)/len(strategy.visible_tests))
        if problem.visible_tests:
            best_solution = max(seed_solutions, key=lambda x: sum(x.test_results)/len(x.test_results))
        else:
            best_solution = seed_solutions[0]

        # filter for the best solution
        # best_solution = self.filter(seed_functions, weighting_mode=self.filter_weighting_mode)
        self.logger.info(f"Best solution: {best_solution.solution}")
        self.record_sample_stats_on_solution(seed_solutions, best_solution)
        return best_solution
    
        