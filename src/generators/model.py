from sre_constants import OP_IGNORE
from typing import List, Union, Optional, Literal
from dataclasses import dataclass
import dataclasses
from ..utils import AbstractLogged
from abc import abstractmethod
import tiktoken
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights


class TimeoutException(Exception):
    """Custom exception for handling timeouts."""
    pass

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai

MessageRole = Literal["system", "user", "assistant"]


@dataclass
class Message():
    role: MessageRole
    content: str

    def display_string(self) -> str:
        if self.role == "system":
            out = f"""----------------------- SYSTEM MESSAGE -----------------------)
{self.content}
----------------------------------------------
"""
        elif self.role == "user":
            out = f"""----------------------- USER MESSAGE -----------------------)
{self.content}
----------------------------------------------
"""
        elif self.role == "assistant":
            out = f"""----------------------- ASSISTANT MESSAGE -----------------------)
{self.content}
----------------------------------------------
"""
        return out
    


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

@dataclass
class LLMResponse:
    content: str
    logprob: float

class ModelBase(AbstractLogged):
    def __init__(self, name: str,):
        self.name, self.total_requests, self.total_tokens, self.output_tokens = name, 0, 0, 0
        self.is_chat = False
        super().__init__()

    def set_temperature(self, temperature: float): self.default_temperature = temperature

    def reset_stats(self): self.total_requests, self.total_tokens, self.output_tokens = 0, 0, 0

    def add_request(self, input_tokens: int, output_tokens: int):
        self.total_requests += 1; self.total_tokens += input_tokens + output_tokens; self.output_tokens += output_tokens

    def __repr__(self) -> str: return f'{self.name}'

    def generate(self, input_data: list[Message] | str, max_tokens: int = 4096, temperature: float = 0.0, num_comps: int = 1, stop_strs: Optional[List[str]] = None) -> List[LLMResponse]:
        is_chat = isinstance(input_data, list)
        input_display_string = "\n".join([m.display_string() for m in input_data]) if is_chat else input_data
        self.logger.info(f"Generating {'chat' if is_chat else 'text'} with input:\n{input_display_string}")
        print(f"Generating {'chat' if is_chat else 'text'} with input:\n{input_display_string}")
        out = self._generate_request(input_data, is_chat, max_tokens, temperature, num_comps, stop_strs)
        self.logger.info(f"Generated output:\n{[resp.content for resp in out]}")
        print(f"Generated first output:\n{out[0].content}")
        return out

    @abstractmethod
    def _generate_request(self, input_data, is_chat: bool, max_tokens: int, temperature: float, num_comps: int, stop_strs: Optional[List[str]] = None) -> List[LLMResponse]: pass


class GPTChat(ModelBase):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name,)
        self.client = openai.OpenAI(api_key=api_key) # infers api_key from environment variable
        self.is_chat = True
        try: self.encoding = tiktoken.encoding_for_model(model_name)
        except: self.encoding = tiktoken.get_encoding("o200k_base")

    def _generate_request(self, input_data, is_chat: bool, max_tokens: int, temperature: float, num_comps: int, stop_strs: Optional[List[str]] = None) -> List[LLMResponse]:
        if is_chat:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[dataclasses.asdict(m) for m in input_data],
                max_tokens=max_tokens, temperature=temperature, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, n=num_comps, logprobs=True,
            )
            input_tokens = sum(len(self.encoding.encode(m.content)) for m in input_data)
            output_tokens = sum(len(self.encoding.encode(choice.message.content)) for choice in response.choices)
            out = [
                LLMResponse(
                    content=choice.message.content,
                    logprob=sum(logprob.logprob for logprob in choice.logprobs.content)  # Sum logprobs
                )
                for choice in response.choices
            ]
            print('logprobs', [res.logprob for res in out])
        else:
            response = self.client.completions.create(
                model=self.name,
                prompt=input_data,
                temperature=temperature, max_tokens=max_tokens, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=stop_strs, n=num_comps, logprobs=True,
            )
            input_tokens = len(self.encoding.encode(input_data))
            output_tokens = sum(len(self.encoding.encode(choice.text)) for choice in response.choices)
            out = [
                LLMResponse(
                    content=choice.text,
                    logprob=sum(logprob.logprob for logprob in choice.logprobs.content)  # Sum logprobs
                )
                for choice in response.choices
            ]

        self.add_request(input_tokens, output_tokens)
        self.logger.info(f"Total tokens: {self.total_tokens}, Output tokens: {self.output_tokens}, Total requests: {self.total_requests}")
        return out

class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")

class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class HFModelBase(ModelBase):
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the Hugging Face pipeline-based model.

        :param model_name: Name of the Hugging Face model to load.
        :param device: Device to load the model on ('cpu', 'cuda', or 'cuda:<device_id>'). Defaults to the best available device.
        """
        super().__init__(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading model on device: {self.device}")
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto" if "cuda" in self.device else None,
            model_kwargs={"torch_dtype": torch.bfloat16 if "cuda" in self.device else torch.float32},
        )

    def _generate_request(
        self,
        input_data: List[Message],
        is_chat: bool,
        max_tokens: int,
        temperature: float,
        num_comps: int,
        stop_strs: Optional[List[str]] = None
    ) -> List[LLMResponse]:
        if not isinstance(input_data, list) or not all(isinstance(msg, Message) for msg in input_data):
            raise ValueError("input_data must be a List of Message objects.")

        # Convert Message objects to dictionaries
        messages = [dataclasses.asdict(m) for m in input_data]

        do_sample = temperature > 0.0

        kwargs = {
            "max_length": max_tokens,
            "num_return_sequences": num_comps,
            "do_sample": do_sample,
            "eos_token_id": self.pipeline.tokenizer.eos_token_id,  # Stop generation when EOS token is reached
        }

        if do_sample:
            kwargs["temperature"] = temperature

        # Generate responses using the pipeline
        try:
            outputs = self.pipeline(messages, **kwargs)
        except TimeoutException:
            print("Generation timed out.")
            return []

        print('first output', outputs[0])

        # Handle pipeline output properly
        responses = []
        for output in outputs:
            if "generated_text" in output and output["generated_text"]:  # Validate output
                generated_messages = output["generated_text"]

                # we need to get the last message from content
                last_message = generated_messages[-1]

                content = last_message["content"]
                
                self.logger.info(f"Generated response:\n {content}")
                responses.append(LLMResponse(content=content, logprob=0.0))
            else:
                # Default to empty content if invalid
                responses.append(LLMResponse(content="", logprob=0.0))

        # Update stats
        input_tokens = sum(len(self.pipeline.tokenizer.encode(msg["content"])) for msg in messages)
        output_tokens = sum(len(self.pipeline.tokenizer.encode(r.content)) for r in responses if r.content)

        self.add_request(input_tokens, output_tokens)

        return responses

class Mistral7BInstruct(HFModelBase):
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Mistral7B-Instruct model.
        
        :param device: Device to load the model on ('cpu', 'cuda', or 'cuda:<device_id>').
        """
        super().__init__(model_name="mistralai/Mistral-7B-Instruct-v0.3", device=device)
        self.is_chat = True

    def __repr__(self) -> str:
        return "Mistral7BInstruct()"


class LLama8BInstruct(HFModelBase):
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the LLama7B-Instruct model.
        
        :param device: Device to load the model on ('cpu', 'cuda', or 'cuda:<device_id>').
        """
        super().__init__(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)
        self.is_chat = True

    def __repr__(self) -> str:
        return "LLama7BInstruct()"


class AnthropicClaude(ModelBase):
    def __init__(self, model_name: str):
        super().__init__(model_name, is_chat=True)
        self.client = anthropic.Anthropic()

    def _generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> list[str]:
        self.logger.debug(f"Temp: {temperature}")
        return self.claude_chat(self.name, messages, max_tokens, temperature, num_comps)

    def _generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> list[str]:
        return self.claude_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def claude_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
    ) -> list[str]:
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_strs,
            n=num_comps,
        )
        return [choice.text for choice in response.choices]  # type: ignore

    @retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
    def claude_chat(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps=1,
    ) -> list[str]:

        # add an assistant message after every system message
        for i, message in enumerate(messages):
            if message.role == "system":
                messages.insert(i + 1, Message(role="assistant", content="I understand."))
                message.role = "user"

        input = [dataclasses.asdict(message) for message in messages]
        # input = [{"role": "user", "content": "This is a test"}]
        # print(input)
        response = self.client.messages.create(
            model=model,
            messages=input,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
        )
        print('response', response.content)
        print('response text:\n', str(response.content))
        return [str(response.content)]