from ..headers import *
from ..utils import AbstractLogged

from typing import Optional
from dataclasses import dataclass, field

# Assuming other classes (CodingProblem, CodingProblemAndSolution) are defined in another module

class DummySolver(Solver):
    '''
    Used for testing purposes. Always returns a fixed, proposed solution that does not necessarily solve the problem.
    '''
    def __init__(self, strategy='default', verbosity=1):
        super().__init__()
        self.strategy = strategy
        self.verbosity = verbosity

    def solve(self, problem: CodingProblem) -> CodingProblemAndSolution:
        '''
        Solve the given coding problem using a dummy approach.
        Always returns a fixed, proposed solution that does not necessarily solve the problem.
        '''
        if self.verbosity > 0:
            self.log(f"Solving problem {problem.task_id} using strategy {self.strategy}")

        if not isinstance(problem, CodingProblem):
            raise TypeError("Expected a CodingProblem instance")
        print(problem.prompt)

        # Simulate a solution process
        proposed_solution = "def solution():\n    return 'This is a fixed solution'"  # Dummy solution
        is_solved = False  # For demonstration, assume solution does not pass hidden tests
        is_passing = False  # Assume it also does not pass visible tests

        return CodingProblemAndSolution(
            task_id=problem.task_id,
            prompt=problem.prompt,
            entry_point=problem.entry_point,
            test=problem.test,
            canonical_solution=problem.canonical_solution,
            # language=problem.language,
            visible_tests=problem.visible_tests,
            proposed_solution=proposed_solution,
            is_solved=is_solved,
            is_passing=is_passing
        )

