from ..headers import CodingProblem, CodingProblemAndSolution, Solver, AbstractLogged
from .base import BaseSolver
from .search_utils import Graph

import matplotlib.pyplot as plt

class ScatterSolver(BaseSolver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self, max_iters: int, *args, strategy_library_name: str = "mcts", codegen_method: str = "outcome", with_accummulated_context: bool = False, num_ideas_per_strategy: int = 3,  **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.strategy_library_name = strategy_library_name
        self.codegen_method = codegen_method
        self.with_accummulated_context = with_accummulated_context
        self.num_ideas_per_strategy = num_ideas_per_strategy

    def generate_improvement_ideas_from_strategy(self, old_strategy: CodingProblemAndSolution, num_ideas_per_strategy: int) -> list[str]:
        # we need to first reflect, then generate ideas
        cur_func_impl = old_strategy.solution
        assert cur_func_impl is not None
        func_sig = old_strategy.prompt
        feedback = old_strategy.feedback_string
        self_reflection = "No reflections."

        # use the debug messages to generate new ideas
        new_ideas = self.generator.strategist_generate_ideas(func_sig=func_sig, prev_func_impl=cur_func_impl, num_ideas=num_ideas_per_strategy, model=self.model, strategy="reflexion", feedback=feedback, self_reflection=self_reflection, temperature=self.temperature)

        # print out new_ideas
        self.logger.debug(f"New ideas: {new_ideas}")
        return new_ideas

    def generate_strategy(self, old_strategy: CodingProblemAndSolution, improvement_idea: str,) -> CodingProblemAndSolution:
        cur_func_impl = old_strategy.solution
        assert cur_func_impl is not None
        func_sig = old_strategy.prompt
        feedback = old_strategy.feedback_string
        self_reflection = "No reflections."

        strategy_return = self.generator.strategist_generate_strategy(func_sig=func_sig, prev_func_impl=cur_func_impl, model=self.model, strategy=self.codegen_method, feedback=feedback, self_reflection=self_reflection, idea_to_implement=improvement_idea, num_comps=1, temperature=self.temperature)

        new_strategy = CodingProblemAndSolution.init_from_coding_problem(old_strategy)
        new_strategy.solution = strategy_return.code
        new_strategy.generation = old_strategy.generation + 1
        new_strategy.extra_kwargs['logprob'] = strategy_return.logprob

        # record token cost
        new_strategy.extra_kwargs['int_output_tokens'] = self.model.output_tokens
        new_strategy.extra_kwargs['int_total_tokens'] = self.model.total_tokens
        new_strategy.extra_kwargs['int_total_requests'] = self.model.total_requests

        # print out new strategy
        self.logger.debug(f"New strategy implementation: {new_strategy.solution}")
        return new_strategy

    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        self.model.reset_stats()
        # record the list of solutions in order they were generated
        all_solutions = []
        # generate visible tests
        self.fill_with_tests(problem, self.number_of_tests)
        # generate seed functions
        seed_functions = self.generate_seed_functions(num_seeds=self.num_seeds, problem=problem)
        # execute the seed functions to record their scores
        for seed_function in seed_functions:
            # self.executor.execute(seed_function)
            all_solutions.append(seed_function)

        # initialize the graph
        graph = Graph.init_from_seed_solutions(seed_functions)
        # initialize general insights

        for i in range(self.max_iters):
            # simulate trajectory to get node to expand
            cur_node = graph.simulate_trajectory(stop_condition="has_unvisited_actions_or_no_better_child")
            # get the current solution
            cur_solution = cur_node.solution
            assert cur_solution is not None

            # if the new_solution passes all the tests, break
            if all(cur_solution.test_results):
                break

            counter = 0
            while not cur_node.has_unvisited_actions():
                if counter > 5:
                    raise ValueError("Could not generate new actions")
                # generate new improvement directions
                directions = self.generate_improvement_ideas_from_strategy(cur_solution, self.num_ideas_per_strategy)
                # add the directions to the node
                cur_node.actions.update(directions)
                counter += 1
                

            # print(cur_node.actions, cur_node.action_to_node.keys())
            # take one of the unvisited actions
            action = next(iter(cur_node.actions - cur_node.action_to_node.keys()))
            # generate the new solution
            new_solution = self.generate_strategy(cur_solution, action)
            # evaluate the new solution
            self.executor.execute(new_solution)
            all_solutions.append(new_solution)
            # add the new solution to the graph
            new_node = graph.add_child_solution(cur_solution, action, new_solution, sum(new_solution.test_results)/len(new_solution.visible_tests))
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
    
        