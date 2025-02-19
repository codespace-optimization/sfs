from src.headers import CodingProblem, CodingProblemAndSolution
from .base import BaseSolver
from ..headers import Solver
from ..generators.generator_types import Generator, ModelBase
from ..generators.factory import generator_factory, model_factory
from ..executors.executor_types import Executor
from ..executors.factory import executor_factory
from .search_utils import Graph

from typing import Any

class TreeSolver(BaseSolver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self, max_iters: int, *args, branching_factor: int = 2, strategy_library_name: str = "mcts", codegen_method: str = "outcome", with_accummulated_context: bool = False, is_puct: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.strategy_library_name = strategy_library_name
        self.codegen_method = codegen_method
        self.with_accummulated_context = with_accummulated_context
        self.branching_factor = branching_factor
        self.is_puct = is_puct

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
        new_strategy.extra_kwargs['logprob'] = response.logprob

        # print out new strategy
        self.logger.debug(f"New strategy implementation: {new_strategy.solution}")
        return new_strategy

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        self.model.reset_stats()
        # record the list of solutions in order they were generated
        all_solutions = []
        # generate visible tests
        if self.do_fill_with_tests:
            # generate visible tests
            self.fill_with_tests(problem, self.number_of_tests)
        else:
            problem.visible_tests = problem.visible_tests[:self.number_of_tests]
        # generate seed functions
        seed_functions = self.generate_seed_functions(num_seeds=self.num_seeds, problem=problem)
        # execute the seed functions to record their scores
        for seed_function in seed_functions:
            # self.executor.execute(seed_function)
            all_solutions.append(seed_function)

        # initialize the graph
        graph = Graph.init_from_seed_solutions(seed_functions)
        graph.is_puct = self.is_puct
        # initialize general insights

        for i in range(self.max_iters):
            # simulate trajectory to get node to expand
            cur_node = graph.simulate_trajectory(stop_condition="has_unvisited_action")

            assert len(cur_node.actions) <= self.branching_factor, "Branching factor exceeded"

            # get the current solution
            cur_solution = cur_node.solution
            assert cur_solution is not None

            # if node has less than branching factor actions, add actions until it reaches the branching factor
            while len(cur_node.actions) < self.branching_factor:
                new_action = len(cur_node.actions)
                cur_node.actions.add(new_action)

            # select an unvisited action to add
            action_to_add = cur_node.get_random_unvisited_action()
            assert action_to_add is not None, "No unvisited actions left"

            # if the new_solution passes all the tests, break
            if all(cur_solution.test_results):
                break

            # generate the new solution
            new_solution = self.generate_strategy(cur_solution)
            # record token cost
            new_solution.extra_kwargs['int_output_tokens'] = self.model.output_tokens
            new_solution.extra_kwargs['int_total_tokens'] = self.model.total_tokens
            new_solution.extra_kwargs['int_total_requests'] = self.model.total_requests
            # evaluate the new solution
            self.executor.execute(new_solution)
            all_solutions.append(new_solution)
            # add the new solution to the graph
            new_node = graph.add_child_solution(parent_solution=cur_solution, action=action_to_add, solution=new_solution, score=sum(new_solution.test_results)/len(new_solution.visible_tests), logp=new_solution.extra_kwargs['logprob'])
            # backpropogate the value
            new_node.backpropogate()
            

        best_solution = graph.get_best_score_solution()
        assert best_solution is not None

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