from ..headers import *
from ..executors.executor_types import Executor
from ..generators.generator_types import Generator, ModelBase

class UnitTestAnalyzer(Analyzer):
    '''
    Analyzes the unit tests
    '''
    
    def __init__(self, executor: Executor, generator: Generator, model: ModelBase, number_of_tests: int,):
        super().__init__()
        self.executor = executor
        self.generator = generator
        self.model = model
        self.number_of_tests = number_of_tests


    def analyze(self, problem: CodingProblemAndSolution) -> float:
        '''
        Analyze the unit tests
        '''
        # generate internal tests if they are less than the number of tests
        # if len(problem.visible_tests) < self.number_of_tests:
            # problem.visible_tests = self.generator.internal_tests(problem.prompt, self.model, self.number_of_tests)

        # always generate new tests for now
        problem.visible_tests = self.generator.internal_tests(problem.prompt, self.model, self.number_of_tests)
    
        
