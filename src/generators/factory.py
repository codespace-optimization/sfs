from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .go_generate import GoGenerator
# from .generator_types import Generator
from .generator import GenericChatGenerator
from .model import ModelBase, GPTChat, AnthropicClaude, Mistral7BInstruct, LLama8BInstruct


def generator_factory(lang: str, ) -> GenericChatGenerator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    # elif lang == "rs" or lang == "rust":
        # return RsGenerator()
    # elif lang == "go" or lang == "golang":
        # return GoGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    # if model starts with gpt
    if model_name.startswith("gpt"):
        return GPTChat(model_name)
    elif model_name.startswith("claude"):
        return AnthropicClaude(model_name)
    elif model_name == "mistral7b-instruct":
        return Mistral7BInstruct()
    elif model_name == "llama8b-instruct":
        return LLama8BInstruct()
    # elif model_name == "starchat":
    #     return StarChat()
    # elif model_name.startswith("codellama"):
    #     # if it has `-` in the name, version was specified
    #     kwargs = {}
    #     if "-" in model_name:
    #         kwargs["version"] = model_name.split("-")[1]
    #     return CodeLlama(**kwargs)
    # elif model_name.startswith("text-davinci"):
    #     return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
