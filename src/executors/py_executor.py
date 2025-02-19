import ast
import signal
import astunparse
from typing import List

from numpy import isin

from .executor_utils import function_with_timeout, eval_code, function_with_timeout_quick
from .executor_types import ExecuteResult, Executor
from ..headers import CodingProblemAndSolution

class PyExecutor(Executor):

    def __init__(self, is_quick: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.is_quick = is_quick

    def _execute(self, problem_and_solution: CodingProblemAndSolution, timeout: int = 5) -> ExecuteResult:
        func = problem_and_solution.solution
        tests = problem_and_solution.visible_tests

        # Combine function code and all assert statements
        imports = 'from typing import *'
        combined_code = f'{imports}\n{func}\n' + "\n".join(tests)

        try:
            self.logger.debug(f"Executing combined python code with timeout:\n\n{combined_code}")
            if self.is_quick:
                function_with_timeout_quick(exec, (combined_code, globals()), timeout)
            else:
                function_with_timeout(exec, (combined_code, globals()), timeout)

            # If no exception occurs, all tests passed
            state = [True] * len(tests)
            feedback = "All tests passed successfully."
            return ExecuteResult(True, feedback, tuple(state))
        except Exception as e:
            self.logger.debug(f"Exception occurred during execution: {e}")

            # Re-execute each test to determine which failed
            success_tests = []
            failed_tests = []
            state = []

            for test in tests:
                test_code = f'{imports}\n{func}\n{test}'
                try:
                    self.logger.debug(f"Executing individual test code with timeout:\n\n{test_code}")
                    if self.is_quick:
                        function_with_timeout_quick(exec, (test_code, globals()), timeout)
                    else:
                        function_with_timeout(exec, (test_code, globals()), timeout)
                    success_tests.append(test)
                    state.append(True)
                except Exception as test_exception:
                    self.logger.debug(f"Test failed: {test_exception}")
                    output = get_output(func, test, timeout=timeout, is_quick=self.is_quick)
                    output_str = str(output)[:100] + ("..." if len(str(output)) > 100 else "")
                    failed_tests.append(f"{test} # output: {output_str}")
                    state.append(False)

            # Generate feedback for failed and successful tests
            feedback = "Tests failed:"
            feedback += "\n" + "\n".join(failed_tests[:5])  # Show up to 5 failed tests for brevity
            feedback += "\n\nTests passed:"
            feedback += "\n" + "\n".join(success_tests[:5])  # Show up to 5 successful tests for brevity

            return ExecuteResult(False, feedback, tuple(state))


    def _evaluate(self, problem_and_solution: CodingProblemAndSolution, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        test = problem_and_solution.test
        func = problem_and_solution.solution
        name = problem_and_solution.entry_point
        
        code = f"""{func}

{test}
    """
        try:
            self.logger.debug(f"Evaluating python code with timeout: \n\n{code}")
            if self.is_quick:
                function_with_timeout_quick(exec, (code, globals()), timeout)
            else:
                function_with_timeout(exec, (code, globals()), timeout)
            # eval_code(line=code, exec_globals=globals(), timeout=5)
            self.logger.debug(f"Correct solution. Code executed successfully")
            return True
        except Exception as e:
            self.logger.debug(f"Incorrect solution. Exception occured: {e}")
            return False

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, is_quick: bool = False, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        if is_quick:
            output = function_with_timeout_quick(eval, (func_call, globals()), timeout)
        else:
            output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    


# if __name__ == "__main__":
#     pass
#     # Test the function
#     func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
#     tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
#     print(PyExecutor().execute(func, tests, timeout=1))