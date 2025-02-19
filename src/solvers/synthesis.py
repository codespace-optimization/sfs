from ..headers import CodingProblem, CodingProblemAndSolution, Solver, AbstractLogged
from .base import BaseSolver
from .search_utils import Graph



import matplotlib.pyplot as plt
import numpy as np

class SynthesisSolver(BaseSolver):
    '''
    Uses reflexion (feedback from executing tests) to iteratively improve the solution
    '''

    def __init__(self, max_iters: int, *args, is_lats_setup: bool = False, strategy_library_name: str = "mcts", codegen_method: str = "outcome", with_accummulated_context: bool = False, num_ideas_per_strategy: int = 3,  use_llm_judge_tie_breaker: bool = False, is_puct: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.strategy_library_name = strategy_library_name
        self.codegen_method = codegen_method
        self.with_accummulated_context = with_accummulated_context
        self.num_ideas_per_strategy = num_ideas_per_strategy
        self.use_llm_judge_tie_breaker = use_llm_judge_tie_breaker
        self.is_lats_setup = is_lats_setup
        self.is_puct = is_puct

    def generate_updated_summary(self, old_summary: str, old_strategy: CodingProblemAndSolution, idea: str, new_strategy: CodingProblemAndSolution,) -> str: 
        '''
        Generates an updated summary based on the old summary and new idea
        '''
        new_summary = self.generator.generate_updated_summary(summary=old_summary, prompt=old_strategy.prompt, prev_impl=old_strategy.solution, prev_feedback=old_strategy.feedback_string, improvement_idea=idea, new_impl=new_strategy.solution, model=self.model, new_feedback=new_strategy.feedback_string, temperature=self.temperature, num_comps=1, is_few_shot=True, score_difference=(sum(new_strategy.test_results)-sum(old_strategy.test_results))/len(old_strategy.test_results)) # type: ignore
        return new_summary

    def generate_improvement_ideas_from_strategy_with_summary(self, old_strategy: CodingProblemAndSolution, num_ideas_per_strategy: int, summary: str) -> list[str]:
        # we need to first reflect, then generate ideas
        cur_func_impl = old_strategy.solution
        assert cur_func_impl is not None
        func_sig = old_strategy.prompt
        feedback = old_strategy.feedback_string

        # use the debug messages to generate new ideas
        new_ideas = self.generator.generate_ideas_with_summary(func_sig=func_sig, prev_func_impl=cur_func_impl, num_ideas=num_ideas_per_strategy, model=self.model, feedback=feedback, temperature=self.temperature, summary=summary)

        # print out new_ideas
        self.logger.info(f"New ideas: {new_ideas}")
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
            # append the seed function to the list of all solutions
            all_solutions.append(seed_function)

            if self.is_lats_setup: # stop early if solution found
                is_solved = self.executor.evaluate(seed_function)
                if is_solved:
                    return seed_function  

        # initialize the graph
        graph = Graph.init_from_seed_solutions(seed_functions)
        graph.is_puct = self.is_puct
        # initialize general insights
        general_insights: str = 'None'

        for i in range(self.max_iters):
            # simulate trajectory to get node to expand
            cur_node = graph.simulate_trajectory(stop_condition="has_unvisited_actions_or_no_better_child")
            # get the current solution
            cur_solution = cur_node.solution
            assert cur_solution is not None
            # if current solution (including seed solutions) is solved, break
            if all(cur_solution.test_results) and not self.is_lats_setup:
                break
            
            if i > 1:
                assert general_insights != 'None'
            counter = 0
            while not cur_node.has_unvisited_actions():
                if counter > 5:
                    self.logger.warning(f"Could not generate new actions with feedback: \n {cur_solution.feedback_string} \n\n Proposed solution: \n {cur_solution.solution}")
                    raise ValueError(f"Could not generate new actions with feedback: \n {cur_solution.feedback_string} \n\n Proposed solution: \n {cur_solution.solution}")
                
                # self.logger.debug(f"Generating new improvement ideas with insights: {general_insights} on iteration {i}")
                # generate new improvement directions
                directions = self.generate_improvement_ideas_from_strategy_with_summary(cur_solution, self.num_ideas_per_strategy, general_insights)
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
            # record token cost
            new_solution.extra_kwargs['int_output_tokens'] = self.model.output_tokens
            new_solution.extra_kwargs['int_total_tokens'] = self.model.total_tokens
            new_solution.extra_kwargs['int_total_requests'] = self.model.total_requests
            # append the new solution to the list of all solutions
            all_solutions.append(new_solution)

            if self.is_lats_setup: # stop early if solution found
                is_solved = self.executor.evaluate(new_solution)
                if is_solved:
                    return new_solution
            # add the new solution to the graph
            new_node = graph.add_child_solution(cur_solution, action, new_solution, sum(new_solution.test_results)/len(new_solution.visible_tests), logp=new_solution.extra_kwargs['logprob'])
            # backpropogate the value
            new_node.backpropogate()
            # update general insights
            general_insights = self.generate_updated_summary(general_insights, cur_solution, action, new_solution)

        # if not self.use_llm_judge_tie_breaker:
        best_solution = graph.get_best_score_solution()
        assert best_solution is not None

        self.record_sample_stats_on_solution(all_solutions=all_solutions, best_solution=best_solution)

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
    
        