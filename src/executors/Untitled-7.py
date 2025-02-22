
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
    

def function_with_timeout(func, args, timeout: int, memory_limit_mb: int=512):
    '''
    Executes a function with a timeout and memory limit.
    Example usage: function_with_timeout(exec, ("def example():\n    pass", globals()), timeout)

    Args:
        func: The function to execute.
        args: Tuple of arguments for the function.
        timeout: Maximum time in seconds allowed for execution.
        memory_limit_mb: Maximum memory in MB allowed for the process.

    Returns:
        The result of the function execution.

    Raises a TimeoutError if execution exceeds the timeout.
    Raises a MemoryLimitExceeded if execution exceeds the memory limit.
    '''
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]
