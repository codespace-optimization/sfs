from typing import NamedTuple, List, Tuple
from abc import ABC, abstractmethod
from ..utils import AbstractLogged
from ..headers import CodingProblemAndSolution

class ExecuteResult(NamedTuple):
    is_passing: bool
    feedback: str
    state: Tuple[bool]

class Executor(AbstractLogged):
    def __init__(self, **kwargs):
        super().__init__()
    
    def execute(self, problem_and_solution: CodingProblemAndSolution, timeout: int = 5, will_modify: bool = True) -> ExecuteResult:
        is_passing, feedback, state = self._execute(problem_and_solution, timeout)
        if will_modify:
            problem_and_solution.is_passing = is_passing
            problem_and_solution.feedback_string = feedback
            problem_and_solution.test_results = state
        return ExecuteResult(is_passing, feedback, state)

    @abstractmethod
    def _execute(self, problem_and_solution: CodingProblemAndSolution, timeout: int = 5) -> ExecuteResult:
        ...

    def evaluate(self, problem_and_solution: CodingProblemAndSolution, timeout: int = 5,  will_modify: bool = True) -> bool:
        is_solved = self._evaluate(problem_and_solution, timeout)
        if will_modify:
            problem_and_solution.is_solved = is_solved
        return is_solved

    @abstractmethod
    def _evaluate(self, problem_and_solution: CodingProblemAndSolution, timeout: int = 5) -> bool:
        ...