from typing import List, Optional, Union
from abc import abstractmethod, ABC

from dataclasses import dataclass
from ..utils import AbstractLogged
from ..headers import CodingProblemAndSolution

from .model import ModelBase


@dataclass
class StrategyReturn:
    code: str
    logprob: float

class Generator(AbstractLogged):
    @abstractmethod
    def self_reflection(self, func: str, feedback: str, model: ModelBase, temperature: float) -> str:
        ...

    @abstractmethod
    def generate_func_impl(self, func_sig: str, strategy: str, model: ModelBase,entry_point: str, temperature: float, num_comps: int = 1, preamble: str = '', is_few_shot: bool = True, prev_func_impl: Optional[str] = None, feedback: Optional[str] = None, self_reflection: Optional[str] = None, 
    ) -> list[StrategyReturn]:
        ...

    @abstractmethod
    def generate_internal_tests(self, func_sig: str, max_num_tests: int, model: ModelBase, entry_point: str, temperature: float, method: str = 'simple', ) -> list[str]:
        '''
        Generates validation tests for the function implementation
        '''
        ...

    @abstractmethod
    def strategist_generate_strategy(self, func_sig: str, model: ModelBase, strategy: str, prev_func_impl: str, feedback: str, self_reflection: str, idea_to_implement: str, temperature: float, num_comps: int = 1,) -> StrategyReturn:
        ...

    # @abstractmethod
    # def strategist_generate_strategy_accumulated_context(self, func_sig: str, model: ModelBase, strategy: str, prev_solutions: list[CodingProblemAndSolution], idea_to_implement: str, temperature: float, num_comps: int = 1, ) -> str:
    #     ...

    @abstractmethod
    def strategist_generate_ideas(self, func_sig: str, prev_func_impl: str, num_ideas: int, model: ModelBase, strategy: str, feedback: str, self_reflection: str, temperature: float, num_comps: int = 1) -> list[str]:
        ...

    @abstractmethod
    def generate_ideas_with_summary(self, func_sig: str, prev_func_impl: str, num_ideas: int, model: ModelBase,  feedback: str, temperature: float, summary: str, num_comps: int = 1, is_few_shot: bool = True,) -> list[str]:
        ...

    @abstractmethod
    def generate_updated_summary(self, prompt: str, prev_impl: str, prev_feedback: str, improvement_idea: str, new_impl: str, model: ModelBase, new_feedback: str, temperature: float, summary: str,  score_difference: float, num_comps: int = 1, is_few_shot: bool = True,) -> str:
        ...
