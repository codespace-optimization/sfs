from .utils import AbstractLogged

from dataclasses import dataclass, field, fields
from abc import abstractmethod
from typing import Optional, Any

@dataclass
class Problem():
    '''
    Coding problem
    '''
    task_id: str # Task ID
    
    def __eq__(self, other):
        if not isinstance(other, CodingProblem):
            return NotImplemented
        return self.task_id == other.task_id

    @staticmethod
    def filter_out_unrelated_kwargs(clss, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Filters out keys from the kwargs dictionary that are not fields of the given class.
        
        Args:
            cls (Type): The class to filter kwargs for.
            kwargs (Dict[str, Any]): The dictionary of keyword arguments.
            
        Returns:
            Dict[str, Any]: Filtered keyword arguments.
        """
        valid_keys = {field.name for field in fields(clss)}
        # print("valid keys", valid_keys)
        return {key: value for key, value in kwargs.items() if key in valid_keys}
    
    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        # Filter kwargs based on the class fields and then set attributes
        filtered_kwargs = Problem.filter_out_unrelated_kwargs(self.__class__, kwargs)
        for key, value in filtered_kwargs.items():
            setattr(self, key, value)
    
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple

@dataclass
class CodingProblem(Problem):
    '''
    Coding problem
    '''
    prompt: str
    entry_point: str
    visible_tests: tuple[str, ...] = field(default_factory=tuple)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_unwritten_kwargs: dict[str, Any] = field(default_factory=dict)

    # the following fields are required for non-plus formatted problems
    test: str = ""

    # the following fields are required for plus formatted problems
    canonical_solution: str = ""
    base_input: list = field(default_factory=list)
    plus_input: list = field(default_factory=list)
    # contract: str = ""
    # assertion: str = ""

    def __init__(self, task_id, prompt, entry_point, test: str = "", 
                 visible_tests: Optional[Tuple[str, ...]] = None, 
                 canonical_solution: str = "", 
                 base_input: Optional[List] = None, 
                 plus_input: Optional[List] = None, 
                 contract: str = "", 
                 assertion: str = "", 
                 **kwargs):
        super().__init__(task_id, **kwargs)
        self.prompt = prompt
        self.entry_point = entry_point
        self.test = test
        self.visible_tests = visible_tests if visible_tests is not None else tuple()
        self.canonical_solution = canonical_solution
        self.base_input = base_input if base_input is not None else []
        self.plus_input = plus_input if plus_input is not None else []
        self.contract = contract
        self.assertion = assertion

        # store extra, unrelated kwargs in extra_kwargs
        self.extra_kwargs = kwargs
        self.extra_unwritten_kwargs = dict()


@dataclass
class CodingVisibleTests(Problem):
    '''
    Additional fields for a coding problem with visible tests
    '''
    visible_tests: list[str]

@dataclass
class CodingProblemAndSolution(CodingProblem):
    '''
    Additional fields for a coding problem with a proposed solution
    '''
    solution: Optional[str] = None # Proposed solution
    is_solved: bool = False # Whether the proposed solution passes the hidden tests
    is_passing: bool = False # whether the proposed solution passes the visible tests
    feedback_string: str = "" # Feedback string
    self_reflection: str = "" # Self-reflection string
    test_results: tuple[bool, ...] = field(default_factory=tuple) # Results of the visible tests
    generation: int = 0 # Generation number
    pass_on_first_try: bool = False # Whether the proposed solution passes the visible tests on the first try

    @staticmethod
    def init_from_coding_problem(problem: CodingProblem) -> 'CodingProblemAndSolution':
        '''
        Initialize a CodingProblemAndSolution from a CodingProblem
        '''
        return CodingProblemAndSolution(
            task_id=problem.task_id,
            prompt=problem.prompt,
            entry_point=problem.entry_point,
            test=problem.test,
            visible_tests=problem.visible_tests,
            canonical_solution=problem.canonical_solution,
            base_input=problem.base_input,
            plus_input=problem.plus_input,
        )
    
    def __hash__(self) -> int:
        '''
        Hash the prompt and proposed solution
        '''
        # print(self.prompt, self.solution)
        assert isinstance(self.prompt, str), f"Prompt is not a string: {self.prompt}"
        assert isinstance(self.solution, str), f"Solution is not a string: {self.solution}"
        return hash((self.prompt, self.solution))

    def get_dict(self) -> dict:
        '''
        Returns everything but `extra_unwritten_kwargs`, which is not serializable
        And [`test`, `canonical_solution`, `base_input`, `plus_input`, `contract`, `assertion`] since these are large
        '''
        return {
            'task_id': self.task_id,
            'prompt': self.prompt,
            'entry_point': self.entry_point,
            'visible_tests': self.visible_tests,
            'solution': self.solution,
            'is_solved': self.is_solved,
            'is_passing': self.is_passing,
            'feedback_string': self.feedback_string,
            'self_reflection': self.self_reflection,
            'test_results': self.test_results,
            'generation': self.generation,
            'pass_on_first_try': self.pass_on_first_try,
            'extra_kwargs': self.extra_kwargs
        }

    def get_prop_test_passed(self) -> float:
        '''
        Returns the proportion of visible tests passed
        '''
        return sum(self.test_results)/len(self.visible_tests)

class Solver(AbstractLogged):
    '''
    Abstract class for generating solutions to coding problems
    '''

    @abstractmethod
    def solve(self, problem: CodingProblem, results_path: str) -> CodingProblemAndSolution:
        '''
        Generate a solution to a coding problem
        '''
        pass
    

class Analyzer(AbstractLogged):
    '''
    Abstract class for analyzing the solutions to coding problems and providing feedback
    '''

    @abstractmethod
    def analyze(self, problem: CodingProblemAndSolution) -> float:
        '''
        Analyze the solution to a coding problem and provide feedback.

        Usually the feedback is stored under problem.extra_kwargs['feedback_string']

        Args:
            problem (CodingProblemAndSolution): The coding problem and solution to analyze

        Returns:
            float: The score of the solution
        '''
        pass