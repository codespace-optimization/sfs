import random
from ..headers import CodingProblem, CodingProblemAndSolution, Solver, AbstractLogged
from .base import BaseSolver
from .search_utils import Graph



import matplotlib.pyplot as plt
import numpy as np

class GeneticSolver(BaseSolver):
    '''
    Implements an evolutionary method based on FunSearch
    '''

    def __init__(self, max_iters: int, *args, num_islands: int = 2, is_lats_setup: bool = False, codegen_method: str = "outcome", with_accummulated_context: bool = False, population_per_island: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.codegen_method = codegen_method
        self.with_accummulated_context = with_accummulated_context
        self.is_lats_setup = is_lats_setup
        self.num_islands = num_islands
        self.population_per_island = population_per_island

    def generate_strategy(self, old_strategy: CodingProblemAndSolution,) -> CodingProblemAndSolution:
        cur_func_impl = old_strategy.solution
        func_sig = old_strategy.prompt
        feedback = old_strategy.feedback_string
        self_reflection = "No reflections."

        assert cur_func_impl is not None
        response = self.generator.generate_func_impl(func_sig=func_sig, prev_func_impl=cur_func_impl, model=self.model, strategy="reflexion", feedback=feedback, self_reflection=self_reflection, entry_point=old_strategy.entry_point, num_comps=1, temperature=self.temperature, )[0]

        new_strategy = CodingProblemAndSolution.init_from_coding_problem(old_strategy)
        new_strategy.solution = response.code
        new_strategy.generation = old_strategy.generation + 1

        # print out new strategy
        self.logger.debug(f"New strategy implementation: {new_strategy.solution}")
        return new_strategy

    def empty_worst_half_islands(self, islands):
        '''
        Returns sorted islands with the worst half of the islands emptied 
        '''
        # sort the islands by the best score of the solutions in the island
        islands = sorted(islands, key=lambda island: max([solution.get_prop_test_passed() for solution in island]))
        # empty the worst half of the islands
        for i in range(len(islands) // 2):
            islands[i] = []
        return islands

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        self.model.reset_stats()
        self.islands = []
        # record the list of solutions in order they were generated
        all_solutions = []
        # generate visible tests
        self.fill_with_tests(problem, self.number_of_tests)
        # generate seed functions
        seed_functions = self.generate_seed_functions(num_seeds=self.num_islands, problem=problem)
        skip = False
        # execute the seed functions to record their scores
        for seed_function in seed_functions:
            # self.executor.execute(seed_function)
            all_solutions.append(seed_function)
            # place one seed function in each island
            self.islands.append([seed_function])
            if all(seed_function.test_results):
                skip = True
                break

        if not skip:
            for i in range(self.max_iters):
                found_island = False
                # find the first island with less than population_per_island solutions
                for island in self.islands:
                    if len(island) < self.population_per_island:
                        found_island = True
                        break

                if not found_island:
                    # if all islands are full, empty n/2 islands with the worst solutions
                    self.islands = self.empty_worst_half_islands(self.islands)
                    # for each empty island, sample a random island (from those not emptied) and place the best solution in the empty island
                    for i in range(len(self.islands) // 2):
                        # sample an island from latter half of the islands
                        island = random.choice(self.islands[len(self.islands) // 2:])
                        # find the best solution in the island
                        best_solution = max(island, key=lambda solution: solution.get_prop_test_passed())
                        # place the best solution in the empty island
                        self.islands[i] = [best_solution]

                # find the first island with less than population_per_island solutions
                for island in self.islands:
                    if len(island) < self.population_per_island:
                        break

                assert isinstance(island, list)

                # sample a solution from the island
                cur_solution: CodingProblemAndSolution = random.choice(island)
                assert cur_solution is not None

                # generate the new solution
                new_solution = self.generate_strategy(cur_solution)
                # record token cost
                new_solution.extra_kwargs['int_output_tokens'] = self.model.output_tokens
                new_solution.extra_kwargs['int_total_tokens'] = self.model.total_tokens
                new_solution.extra_kwargs['int_total_requests'] = self.model.total_requests
                # evaluate the new solution
                self.executor.execute(new_solution)
                all_solutions.append(new_solution)

                # add the new solution to the island
                island.append(new_solution)

                # if the new_solution passes all the tests, break
                if all(new_solution.test_results):
                    break
            
        # find the best solution in all the islands
        best_solution = max([max(island, key=lambda solution: solution.get_prop_test_passed()) for island in self.islands], key=lambda solution: solution.get_prop_test_passed())

        self.record_sample_stats_on_solution(all_solutions, best_solution)

        # if logging level is debug, generate an interactive plot of the search tree
        # if self.logger.level <= 10:
        #     fig = graph.generate_interactive_plot()
        #     # save the plot to results path
        #     fig.write_html(results_path + f"/{problem.task_id}_search_tree.html")

        # generate a static plot of the search tree and save it to results path
        # graph.generate_visualization()
        # save the plot to results path
        # plt.savefig(results_path + f"/{problem.task_id}_search_tree.png")
        return best_solution
        