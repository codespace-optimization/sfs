import logging
from abc import ABC

class AbstractLogged(ABC):
    _instance_counter = 0

    def __init__(self):
        # Increment the instance counter and set it as part of the logger name
        type(self)._instance_counter += 1
        self.instance_id = type(self)._instance_counter
        
        # Create a logger with a name based on the class name and instance counter
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.instance_id}")

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)