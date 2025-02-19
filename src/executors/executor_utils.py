import os
import signal
import json
import multiprocessing
import concurrent.futures


def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)



def timeout_handler(_, __):
    raise TimeoutError()

import os, json
def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)

from threading import Thread
class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret
    
def function_with_timeout(func, args, timeout):
    """
    Executes a function with a timeout using the signal-based approach.

    Args:
        func: The function to execute.
        args: A tuple of arguments to pass to the function.
        timeout: Maximum time in seconds allowed for execution.

    Returns:
        The result of the function execution.

    Raises:
        TimeoutError: If execution exceeds the timeout.
        Exception: If the function raises an exception.
    """
    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the timeout
    signal.alarm(timeout)

    try:
        # Call the function and return the result
        result = func(*args)
        # Cancel the alarm if function completes on time
        signal.alarm(0)
        return result
    except TimeoutError as e:
        raise e
    except Exception as e:
        # Cancel the alarm in case of other exceptions
        signal.alarm(0)
        raise e


def function_with_timeout_quick(func, args, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Function execution exceeded the timeout.")

if __name__ == "__main__":
    func_string = """def solution(stdin: str) -> str:
    cache_size, num_objects, num_accesses, *accesses = map(int, stdin.split())
    cache = []
    cache_dict = {}
    num_reads = 0

    for obj in accesses:
        if obj in cache_dict:
            cache.remove(obj)
            cache.append(obj)
        else:
            if len(cache) == cache_size:
                del cache_dict[cache.pop(0)]
            cache.append(obj)
            cache_dict[obj] = True
            num_reads += 1

    return str(num_reads)


def check(candidate):
    assert candidate('1 2 3\\n0\\n0\\n1') == '2'
    assert candidate('3 4 8\\n0\\n1\\n2\\n3\\n3\\n2\\n1\\n0') == '5'

check(solution)"""

    try:
        # Execute the function with a timeout of 1 second and memory limit of 256 MB
        result = function_with_timeout(exec, (func_string, globals()), timeout=5, memory_limit_mb=2048)
        print("Function executed successfully.")
    except TimeoutError:
        print("Function execution exceeded the timeout limit.")


# ====== check test io ======

import contextlib, io, signal
@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield stream

class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

def eval_code(line, timeout=3., return_exec_globals=False, exec_globals=None):
    # try:
    exec_globals = {} if exec_globals is None else exec_globals
    with swallow_io() as s:
        with time_limit(timeout):
            result = exec(line, exec_globals)
    if return_exec_globals:
        return exec_globals
    else:
        return result
    # except TimeoutException:
    #     return 'timed out'
    # except BaseException as e:
    #     return f"failed: {e}\nPrinted outputs: {s.getvalue()}"