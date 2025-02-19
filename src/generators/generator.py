from math import log
from ..utils import AbstractLogged
from typing import Optional, List, Callable
from .model import ModelBase, message_to_str, Message
from ..headers import CodingProblem, CodingProblemAndSolution
from .generator_types import Generator, StrategyReturn
from ..string_utils import StringUtils


import random
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    code_block_instruction: str

    parse_code_block: Callable[[str], str]
    add_code_block: Callable[[str], str]
    parse_tests: Callable[[str], List[str]]
    is_syntax_valid: Callable[[str], bool]
    parse_ideas: Callable[[str, int], list[str]]
    parse_insights: Callable[[str], str]

    reflection_chat_instruction: str
    reflection_few_shot: list[Message]
    simple_chat_instruction: str
    simple_generate_few_shot: list[Message]
    self_reflection_chat_instruction: str
    self_reflection_few_shot: list[Message]
    test_generation_chat_instruction: str
    test_generation_few_shot: list[Message]
    test_generation_chat_instruction_cot: str
    test_generation_few_shot_cot: list[Message]
    test_refine_chat_instruction: str
    test_refine_few_shot: list[Message]
    idea_generation_instruction: str
    idea_generation_few_shot: list[Message]
    strategy_generation_instruction: str
    strategy_generation_few_shot: list[Message]
    summary_update_instruction: str
    summary_update_few_shot: list[Message]
    scout_idea_generation_instruction: str
    scout_idea_generation_few_shot: list[Message]
    

class GenericChatGenerator(Generator):

    def __init__(self, config: GeneratorConfig):
        '''
        Initialize the generator with the model and instruction prompts
        ''' 
        super().__init__()
        self.config = config


    def generate_self_reflection(self, func: str, feedback: str, model: ModelBase, temperature: float) -> str:
        '''
        Generate a self-reflection message
        '''
        if model.is_chat:
            if self.config.self_reflection_few_shot is not None:
                messages = [
                    Message(
                        role="system",
                        content=self.config.self_reflection_chat_instruction,
                    ),
                ] + self.config.self_reflection_few_shot + [
                    Message(
                        role="user",
                        content=f'[function impl]:\n{self.config.add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                    )
                ]
                reflection = model.generate(input_data=messages, temperature=temperature)[0]
                print(f'Self reflection output: {reflection}')
            else:
                messages = [
                    Message(
                        role="system",
                        content=self.config.self_reflection_chat_instruction,
                    ),
                    Message(
                        role="user",
                        content=f'[function impl]:\n{self.config.add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                    )
                ]
                reflection = model.generate(input_data=messages, temperature=temperature)[0]
        else:
            raise ValueError("Model is not chat based")
        return reflection.content 

    def strategist_generate_ideas(self, func_sig: str, prev_func_impl: str, num_ideas: int, model: ModelBase, strategy: str, feedback: str, self_reflection: str, temperature: float, num_comps: int = 1) -> list[str]:
        if strategy != "reflexion" and strategy != "simple":
            raise ValueError(
                f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
        if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
            raise ValueError(
                f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")
        
        assert model.is_chat, "Model is not chat based"
        if strategy == "reflexion":
            prompt = f"{self.config.idea_generation_instruction}"
            # prompt = f"{self.config.idea_generation_instruction}\n{self.config.code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            # self.config.print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
            ] + self.config.idea_generation_few_shot + [
                Message(
                    role="user",
                    content=f"""[prompt]:\n{func_sig}\n\n[previous impl]:{self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n What are some reflections you can draw and what are {num_ideas} ways in which you can fix the previous impl so that it passes all the tests? Be as specific and concrete as possible, mentioning specific code blocks or helper functions that you will modify. Remember to number your ideas as Idea 1:, Idea 2:, ... etc. \n\n[reflection and {num_ideas} improvement ideas]:"""
                ),
            ]
            model_responses = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
            

            return self.config.parse_ideas(model_responses[0].content, num_ideas)
        else:
            raise ValueError("Model is not chat based")
        
    def generate_ideas_with_summary(self, func_sig: str, prev_func_impl: str, num_ideas: int, model: ModelBase,  feedback: str, temperature: float, summary: str, num_comps: int = 1, is_few_shot: bool = True,) -> list[str]:
        assert model.is_chat, "Model is not chat based"

        sys_message = Message(role="system", content=self.config.scout_idea_generation_instruction)
        user_message = Message(
                role="user",
                content=f"[prompt]:\n{func_sig}\n\n[previous impl]:{self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[general insights]:\n{summary}\n\nWhat are some reflections you can draw based on the results and insights, and what are {num_ideas} different ways in which you can fix the previous implementation so that it passes all the tests? Be as specific and concrete as possible, mentioning specific code blocks or helper functions that you will modify. Remember to number your ideas as Idea 1:, Idea 2:, etc. \n\n[reflection and {num_ideas} improvement ideas]:"
            )
        if is_few_shot:
            messages = [sys_message] + self.config.scout_idea_generation_few_shot + [user_message]
        else:
            messages = [sys_message, user_message]

        model_responses = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
        return self.config.parse_ideas(model_responses[0].content, num_ideas)

    def strategist_generate_strategy(self, func_sig: str, model: ModelBase, strategy: str, prev_func_impl: str, feedback: str, self_reflection: str, idea_to_implement: str, temperature: float, num_comps: int = 1, ) -> StrategyReturn:
        assert model.is_chat, "Model is not chat based"
        if strategy == "reflexion":
            prompt = f"{self.config.strategy_generation_instruction}\n{self.config.code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
            ] + self.config.strategy_generation_few_shot + [
                Message(
                    role="user",
                    content=f"[prompt]:\n{func_sig}\n\n[previous impl]: {self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback} \n\n[reflection]:\n{self_reflection} \n\n[improvement idea]:\n{idea_to_implement} \nIncorporate the idea above into the previous impl as best as you can. \n\n[improved impl]:",
                ),
            ]
            model_responses = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
        elif strategy == "outcome":
            # raise ValueError("Outcome strategy not recommended")
            prompt = f"{self.config.strategy_generation_instruction}\n{self.config.code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
            ] + self.config.strategy_generation_few_shot + [
                Message(
                    role="user",
                    content=f"[prompt]:\n{func_sig}\n\n[previous impl]: {self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback} \n\n[improvement idea]:\n{idea_to_implement} \nIncorporate the idea above into the previous impl as best as you can. \n\n[improved impl]:",
                ),
            ]
            model_responses = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
        elif strategy == "minimal":
            raise ValueError("Minimal strategy not recommended")
            prompt = f"{self.config.strategy_generation_instruction}\n{self.config.code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
            ] + self.config.strategy_generation_few_shot + [
                Message(
                    role="user",
                    content=f"[prompt]:\n{func_sig}\n\n[previous impl]: {self.config.add_code_block(prev_func_impl)} \n\n[improvement idea]:\n{idea_to_implement} \nIncorporate the idea above into the previous impl as best as you can. \n\n[improved impl]:",
                ),
            ]
            model_responses = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
        else:
            raise ValueError(f"Invalid strategy: `{strategy}`")

        return StrategyReturn(code=self.config.parse_code_block(model_responses[0].content), logprob=model_responses[0].logprob)

    def generate_internal_tests(self, func_sig: str, max_num_tests: int, model: ModelBase, entry_point: str, temperature: float, method: str = 'simple', ) -> list[str]:
        '''
        Generate internal (visible, validation) tests for the given coding problem
        '''
        assert model.is_chat, "Model is not chat based"
        if method == 'react':
            raise ValueError("React method not recommended")
            messages = [
                Message(
                    role="system",
                    content=self.config.test_generation_chat_instruction,
                ),
            ] + self.config.test_generation_few_shot + [
                Message(
                    role="user",
                    content=f"[func signature]:\n{func_sig}\n\n[think]:"
                )
            ]
            outputs = model.generate(input_data=messages, max_tokens=1024, temperature=temperature)
            # print(f'React test generation outputs: {outputs}')
        elif method == 'simple':
            messages = [
                Message(
                    role="system",
                    content=self.config.test_generation_chat_instruction,
                ),
            ] + self.config.test_generation_few_shot + [
                Message(
                    role="user",
                    content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[{max_num_tests} unit tests]:",
                )
            ]
            outputs = model.generate(input_data=messages, max_tokens=1024, temperature=temperature)
        elif method == 'cot':
            raise ValueError("COT method not recommended")
            messages = [
                Message(
                    role="system",
                    content=self.config.test_generation_chat_instruction_cot,
                ),
            ] + self.config.test_generation_few_shot_cot + [
                Message(
                    role="user",
                    content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[{max_num_tests} unit tests]:",
                )
            ]
            outputs = model.generate(input_data=messages, max_tokens=1024, temperature=temperature)
        elif method == 'refine':
            raise ValueError("Refine method not recommended")
            messages = [
                Message(
                    role="system",
                    content=self.config.test_generation_chat_instruction,
                ),
            ] + self.config.test_generation_few_shot + [
                Message(
                    role="user",
                    content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[{max_num_tests} unit tests]:",
                )
            ]
            outputs = model.generate_chat(messages=messages, max_tokens=1024, temperature=temperature)

            # parse out all the tests and refine them using the LLM
            all_tests = self.config.parse_tests(outputs[0])  # NOTE: outputs is a list of strings
            valid_tests = [test for test in all_tests if self.config.is_syntax_valid(test)]
            valid_tests = valid_tests[:max_num_tests]
            
            # now refine the tests in another refine step for each test
            for i, test in enumerate(valid_tests):
                messages = [
                    Message(
                        role="system",
                        content=self.config.test_refine_chat_instruction,
                    ),
                ] + self.config.test_refine_few_shot + [
                    Message(
                        role="user",
                        content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[proposed test]\n{test}\n\n[analysis and improved test]:",
                    )
                ]
                outputs = model.generate_chat(messages=messages, max_tokens=1024, temperature=temperature)
                parsed_list = self.config.parse_tests(outputs[0]) # if test is correct, then length is 0
                if len(parsed_list) > 0:
                    valid_tests[i] = parsed_list[0]

            return valid_tests
        else:
            raise ValueError(f"Invalid method: `{method}`")
    
        all_tests = self.config.parse_tests(outputs[0].content)  # NOTE: outputs is a list of strings
        valid_tests = [test for test in all_tests if self.config.is_syntax_valid(test)]

        # return top max_num_tests tests
        return valid_tests[:max_num_tests]

        # return self.config.sample_n_random(valid_tests, max_num_tests)

    def generate_func_impl(self, func_sig: str, strategy: str, model: ModelBase,entry_point: str, temperature: float, num_comps: int = 1, preamble: str = '', is_few_shot: bool = True, prev_func_impl: Optional[str] = None, feedback: Optional[str] = None, self_reflection: Optional[str] = None, ) -> list[StrategyReturn]:
        '''
        Generate function implementation based on the given strategy
        '''
        if strategy != "reflexion" and strategy != "simple":
            raise ValueError(
                f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
        if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
            raise ValueError(
                f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

        if model.is_chat:
            if strategy == "reflexion":
                
                system_prompt = f"{self.config.reflection_chat_instruction}\n{self.config.code_block_instruction}\n{preamble}"
                # func_bodies is a really bad name, as it can also be just 1 string
                # self.config.print_messages(prompt, message)
                messages = [
                    Message(
                        role="system",
                        content=system_prompt,
                    ),
                ] + self.config.reflection_few_shot + [
                    Message(
                        role="user",
                        content=f"[prompt]:\n{func_sig}\n\n[previous impl]:\n{self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:",
                    ),
                ]
                func_bodies = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
            else:
                system_prompt = f"{self.config.simple_chat_instruction}\n{self.config.code_block_instruction}\n{preamble}"

                if is_few_shot:
                    messages = [
                        Message(
                            role="system",
                            content=system_prompt
                        ),
                    ] + self.config.simple_generate_few_shot + [
                        Message(
                            role="user",
                            content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[object impl]:",
                        ),
                    ]
                else:
                    messages = [
                        Message(
                            role="system",
                            content=system_prompt,
                        ),
                        Message(
                            role="user",
                            content=func_sig,
                        ),
                    ]
                func_bodies = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
        else:
            raise ValueError("Model is not chat based")

        strategy_returns = [StrategyReturn(code=self.config.parse_code_block(func_body.content), logprob=func_body.logprob) for func_body in func_bodies]
        # self.print_generated_func_body("\n\n".join(func_bodies))
        return strategy_returns

    def generate_updated_summary(self, prompt: str, prev_impl: str, prev_feedback: str, improvement_idea: str, new_impl: str, model: ModelBase, new_feedback: str, temperature: float, summary: str, score_difference: float, num_comps: int = 1, is_few_shot: bool = True) -> str:
        assert model.is_chat

        sys_message = Message(role="system", content=self.config.summary_update_instruction)
        user_message = Message(role="user", content=f"[prompt]:\n{prompt}\n\n[previous implementation]:\n{self.config.add_code_block(prev_impl)}\n\n[previous implementation feedback]:\n{prev_feedback}\n\n[improvement idea]:\n{improvement_idea}\n\n[new implementation]:\n{self.config.add_code_block(new_impl)}\n\n[new implementation feedback]:\n{new_feedback}\n\n[score improvement]:\n{score_difference}\n\n[previous insights]:\n{summary}\n\n# Reflect on how the improvement idea changed the performance of the code implementation and update your insights accordingly.\n[updated insights]:")

        if is_few_shot:
            messages = [sys_message] + self.config.summary_update_few_shot + [user_message]
        else:
            messages = [sys_message, user_message]

        model_response = model.generate(input_data=messages, num_comps=num_comps, temperature=temperature)
        return self.config.parse_insights(model_response[0].content)

    @staticmethod
    def sample_n_random(items: List[str], n: int) -> List[str]:
        """Sample min(n, len(items)) random items from a list"""
        assert n >= 0
        if n >= len(items):
            return items
        return random.sample(items, n)

    @staticmethod
    def print_messages(system_message_text: str, user_message_text: str) -> None:
        pass
    #     print(f"""----------------------- SYSTEM MESSAGE -----------------------)
    # {system_message_text}
    # ----------------------------------------------
    # ----------------------- USER MESSAGE -----------------------
    # {user_message_text}
    # ----------------------------------------------
    # """, flush=True)

    @staticmethod
    def print_generated_func_body(func_body_str: str) -> None:
        pass
    #     print(f"""--------------------- GENERATED FUNC BODY ---------------------
    # {func_body_str}
    # ------------------------------------------""")
