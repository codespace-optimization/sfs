from dataclasses import dataclass
from .utils import AbstractLogged
from .headers import CodingProblem, CodingProblemAndSolution, CodingVisibleTests, Solver
from .executors.executor_types import Executor

import json
import os
from typing import Optional, Iterator
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.evaluate import evaluate

@dataclass
class PlusEvaluateFlags:
    dataset: str  # No default value since it's required
    samples: str  # No default value since it's required
    base_only: bool = False
    parallel: Optional[int] = None  # Default is None to allow auto-calculation based on CPU count
    i_just_wanna_run: bool = False
    test_details: bool = False
    min_time_limit: float = 1.0
    gt_time_limit_factor: float = 4.0
    mini: bool = False
    noextreme: bool = False

class CodingExperimentManager(AbstractLogged):
    '''
    Coding problem environment that sets up our experiment
    '''

    def __init__(self, solver: Solver, executor: Executor, solution_set_path: str, report_card_path: str, results_path: str, problem_set_path: Optional[str] = None, problem_set_name: Optional[str] = None, test_set_path: Optional[str] = None, is_plus_format: bool = False) -> None:
        '''
        Args:
            solver: Solver that will be used to solve the problems
            executor: Executor that will be used to evaluate the solutions
            problem_set_path: Path to the problem set
            solution_set_path: Path to the solution set
            test_set_path: Path to the test set (optional)
        '''
        # assert that either problem_set_path or problem_set_name is provided
        assert problem_set_path or problem_set_name, "Either problem_set_path or problem_set_name must be provided"
        super().__init__()
        self.solver = solver
        self.executor = executor
        self.problem_set_path = problem_set_path
        self.problem_set_name = problem_set_name
        self.solution_set_path = solution_set_path
        self.test_set_path = test_set_path
        self.report_card_path = report_card_path
        self.results_path = results_path
        self.tests = self.load_visible_tests(test_set_path) if test_set_path else {}
        self.solved_task_ids = self.load_solved_task_ids(solution_set_path)
        self.is_plus_format = is_plus_format

    @staticmethod
    def load_visible_tests(test_set_path: str) -> dict[str, list]:
        list_of_test_dicts = CodingExperimentManager.load_jsonl_file(test_set_path)
        tests = {test_dict['task_id']: test_dict['visible_tests'] for test_dict in list_of_test_dicts}
        return tests

    @staticmethod
    def load_jsonl_file(filepath: str) -> list:
        data = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def append_to_jsonl_file(entry: dict, filepath: str) -> None:
        # recursively remove anything in `entry` that is not json serializable
        with open(filepath, 'a') as file:
            file.write(json.dumps(entry) + '\n')

    def load_solved_task_ids(self, solution_set_path: str) -> set:
        task_ids = set()
        if os.path.exists(solution_set_path):
            with open(solution_set_path, 'r') as file:
                for line in file:
                    solution_dict = json.loads(line)
                    task_ids.add(solution_dict['task_id'])
        return task_ids

    def solve_unsolved_problems(self, pass_at_k: int = 1) -> None:
        '''
        Solve any problems that have not yet been solved
        '''
        for problem in self.stream_problems():
            if problem.task_id not in self.solved_task_ids:
                if problem.task_id in self.tests:
                    problem.visible_tests =problem.visible_tests + tuple(self.tests[problem.task_id])
                
                solution = self.solver.solve(problem, self.results_path)
                if self.problem_set_name != 'humaneval_plus' and self.problem_set_name != 'mbpp_plus':
                    self.evaluate_solution(solution)
                self.solved_task_ids.add(solution.task_id)
                self.append_to_jsonl_file(solution.get_dict(), self.solution_set_path)
                self.logger.info("Solved problem %s", problem.task_id)
        # evaluate the solutions
        self.evaluate_solutions(self.solution_set_path)
        # log the solution stats
        # self.log_solution_stats()

    def evaluate_solutions(self, path: str) -> None:
        '''
        Evaluate the solutions in the solution set
        '''
        if self.problem_set_name == 'humaneval_plus':
            evaluate_flags = PlusEvaluateFlags(dataset='humaneval', samples=path)
            evaluate(evaluate_flags)
        elif self.problem_set_name == 'mbpp_plus':
            evaluate_flags = PlusEvaluateFlags(dataset='mbpp', samples=path)
            evaluate(evaluate_flags)
        else:
            pass
            # with open(path, 'r') as file:
            #     for line in file:
            #         solution_dict = json.loads(line)
            #         solution = CodingProblemAndSolution(**solution_dict)
            #         self.evaluate_solution(solution)
            #         self.append_to_jsonl_file(solution.get_dict(), self.solution_set_path)
                    # self.logger.info("Evaluated problem %s", solution.task_id)

    def evaluate_solution(self, solution: CodingProblemAndSolution) -> bool:
        '''
        Evaluate a solution
        '''
        return self.executor.evaluate(solution)

    def stream_problems(self) -> Iterator[CodingProblem]:
        '''
        Stream problems from the problem set. This way we don't have to load all problems into memory at once
        '''
        if self.problem_set_path:
            with open(self.problem_set_path, 'r') as file:
                for line in file:
                    problem_dict = json.loads(line)
                    yield CodingProblem(**problem_dict)
        else:
            if self.problem_set_name == 'humaneval_plus':
                for task_id, problem_dict in get_human_eval_plus().items():
                    yield CodingProblem(**problem_dict)
            elif self.problem_set_name == 'mbpp_plus':
                for task_id, problem_dict in get_mbpp_plus().items():
                    yield CodingProblem(**problem_dict)
            else:
                raise ValueError("Invalid problem_set_name")

    def append_to_report_card(self, entry: str) -> None:
        '''
        Append an entry to the report card
        '''
        with open(self.report_card_path, 'a') as file:
            file.write(entry + '\n')

    def log_solution_stats(self) -> None:
        '''
        We log stats such as the proportion of problems solved and the confusion matrix
        '''
        solved_problems = 0
        total_problems = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        first_generation = 0
        pass_on_first_try = 0
        solved_not_on_first_try = 0
        solved_first_generation = 0
        solved_secondplus_generation = 0
        sum_similarity = 0

        with open(self.solution_set_path, 'r') as file:
            for line in file:
                solution_dict = json.loads(line)
                total_problems += 1
                if solution_dict['is_solved']:
                    solved_problems += 1
                    if 'pass_on_first_try' in solution_dict:
                        if not solution_dict['pass_on_first_try']:
                            solved_not_on_first_try += 1

                if solution_dict['is_solved'] and solution_dict['is_passing']:
                    true_positive += 1
                elif not solution_dict['is_solved'] and not solution_dict['is_passing']:
                    true_negative += 1
                elif solution_dict['is_solved'] and not solution_dict['is_passing']:
                    false_negative += 1
                elif not solution_dict['is_solved'] and solution_dict['is_passing']:
                    false_positive += 1

                if solution_dict['generation'] == 0:
                    first_generation += 1
                    if solution_dict['is_solved']:
                        solved_first_generation += 1
                else:
                    if solution_dict['is_solved']:
                        solved_secondplus_generation += 1

                if 'pass_on_first_try' in solution_dict:
                    if solution_dict['pass_on_first_try']:
                        pass_on_first_try += 1

                if 'avg_similarity' in solution_dict['extra_kwargs']:
                    sum_similarity += solution_dict['extra_kwargs']['avg_similarity']
                    

        proportion_solved = solved_problems / total_problems if total_problems > 0 else 0
        self.append_to_report_card(f"Proportion of problems solved: {proportion_solved}")
        self.append_to_report_card("Confusion matrix:")
        self.append_to_report_card(f"True positive: {true_positive}")
        self.append_to_report_card(f"True negative: {true_negative}")
        self.append_to_report_card(f"False positive: {false_positive}")
        self.append_to_report_card(f"False negative: {false_negative}")

        
        self.append_to_report_card(f"Proportion of solutions that do not pass validation tests: {(true_negative + false_negative) / total_problems}")
        self.append_to_report_card(f"Proportion of passing on first try: {pass_on_first_try / total_problems}")
        # self.append_to_report_card(f"Proportion of solved given passing on first try: {pass_on_first_try / (true_positive + false_positive)}")
        self.append_to_report_card(f"Proportion of passing not on first try: {1.0 - pass_on_first_try / total_problems - (true_negative + false_negative) / total_problems}")
        self.append_to_report_card(f"Proportion of solved not on first try: {solved_not_on_first_try/total_problems}")
        self.append_to_report_card(f"Proportion of using 1st generation solution as submitted solution: {first_generation / total_problems}")
        self.append_to_report_card(f"Proportion of solved using 1st generation solution as submitted solution: {solved_first_generation / total_problems}")
        self.append_to_report_card(f"Proportion of solved using 2nd+ generation solution as submitted solution: {solved_secondplus_generation / total_problems}")
        self.append_to_report_card(f"Average similarity of solutions: {sum_similarity / total_problems}")
