"""
Template System for Dataset Standardization

Provides a flexible template system inspired by LLaMA-Factory
to handle diverse dataset formats (Alpaca, ShareGPT, OpenAI, etc.)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .formatters import Formatter, get_formatter


@dataclass
class Template:
    """Template configuration for dataset formatting"""
    
    format_user: Formatter
    format_assistant: Formatter
    format_system: Formatter
    format_function: Formatter
    format_observation: Formatter
    format_tools: Formatter
    format_prefix: Formatter
    default_system: str
    stop_words: List[str]
    system_token: str = ""
    user_token: str = ""
    assistant_token: str = ""
    eos_token: str = ""
    bos_token: str = ""
    
    def format_conversation(self, messages: List[Dict[str, Any]], 
                          system_message: Optional[str] = None) -> str:
        """Format a conversation using this template"""
        formatted_parts = []
        
        # Add BOS token if specified
        if self.bos_token:
            formatted_parts.append(self.bos_token)
        
        # Add system message
        system_msg = system_message or self.default_system
        if system_msg:
            system_formatted = self.format_system.apply(
                system_msg,
                system=system_msg,
                system_token=self.system_token
            )
            if system_formatted.strip():
                formatted_parts.append(system_formatted)
        
        # Format conversation messages
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                user_formatted = self.format_user.apply(
                    content,
                    user=content,
                    user_token=self.user_token
                )
                formatted_parts.append(user_formatted)
                
            elif role == 'assistant':
                assistant_formatted = self.format_assistant.apply(
                    content,
                    assistant=content,
                    assistant_token=self.assistant_token
                )
                formatted_parts.append(assistant_formatted)
                
            elif role == 'function':
                function_formatted = self.format_function.apply(
                    content,
                    function=content
                )
                formatted_parts.append(function_formatted)
                
            elif role == 'observation':
                observation_formatted = self.format_observation.apply(
                    content,
                    observation=content
                )
                formatted_parts.append(observation_formatted)
        
        # Join parts and add EOS if needed
        result = "".join(formatted_parts)
        if self.eos_token:
            result += self.eos_token
            
        return result


def create_alpaca_template() -> Template:
    """Create Alpaca-style template"""
    return Template(
        format_user=get_formatter("### Instruction:\n{{user}}\n\n"),
        format_assistant=get_formatter("### Response:\n{{assistant}}"),
        format_system=get_formatter("{{system}}\n\n"),
        format_function=get_formatter(""),
        format_observation=get_formatter(""),
        format_tools=get_formatter(""),
        format_prefix=get_formatter("### Response:\n"),
        default_system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        stop_words=["### Instruction:", "### Response:"],
        eos_token="</s>"
    )


def create_chatml_template() -> Template:
    """Create ChatML template (OpenAI-style)"""
    return Template(
        format_user=get_formatter("<|im_start|>user\n{{user}}<|im_end|>\n"),
        format_assistant=get_formatter("<|im_start|>assistant\n{{assistant}}<|im_end|>\n"),
        format_system=get_formatter("<|im_start|>system\n{{system}}<|im_end|>\n"),
        format_function=get_formatter("<|im_start|>function\n{{function}}<|im_end|>\n"),
        format_observation=get_formatter("<|im_start|>observation\n{{observation}}<|im_end|>\n"),
        format_tools=get_formatter(""),
        format_prefix=get_formatter("<|im_start|>assistant\n"),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_start|>", "<|im_end|>"],
        eos_token="<|im_end|>"
    )


def create_vicuna_template() -> Template:
    """Create Vicuna conversation template"""
    return Template(
        format_user=get_formatter("USER: {{user}} "),
        format_assistant=get_formatter("ASSISTANT: {{assistant}}"),
        format_system=get_formatter("{{system}}\n\n"),
        format_function=get_formatter(""),
        format_observation=get_formatter(""),
        format_tools=get_formatter(""),
        format_prefix=get_formatter("ASSISTANT: "),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=["USER:", "ASSISTANT:"],
        eos_token="</s>"
    )


def create_llama2_chat_template() -> Template:
    """Create Llama-2 Chat template"""
    return Template(
        format_user=get_formatter("[INST] {{user}} [/INST]"),
        format_assistant=get_formatter("{{assistant}}</s>"),
        format_system=get_formatter("<<SYS>>\n{{system}}\n<</SYS>>\n\n"),
        format_function=get_formatter(""),
        format_observation=get_formatter(""),
        format_tools=get_formatter(""),
        format_prefix=get_formatter(""),
        default_system="You are a helpful, respectful and honest assistant.",
        stop_words=["[INST]", "[/INST]"],
        bos_token="<s>",
        eos_token="</s>"
    )


def create_qwen_template() -> Template:
    """Create Qwen template"""
    return Template(
        format_user=get_formatter("<|im_start|>user\n{{user}}<|im_end|>\n"),
        format_assistant=get_formatter("<|im_start|>assistant\n{{assistant}}<|im_end|>\n"),
        format_system=get_formatter("<|im_start|>system\n{{system}}<|im_end|>\n"),
        format_function=get_formatter("<|im_start|>function\n{{function}}<|im_end|>\n"),
        format_observation=get_formatter("<|im_start|>observation\n{{observation}}<|im_end|>\n"),
        format_tools=get_formatter(""),
        format_prefix=get_formatter("<|im_start|>assistant\n"),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_start|>", "<|im_end|>"],
        eos_token="<|im_end|>"
    )


# Template registry
TEMPLATE_REGISTRY: Dict[str, callable] = {
    "alpaca": create_alpaca_template,
    "chatml": create_chatml_template,
    "vicuna": create_vicuna_template,
    "llama2": create_llama2_chat_template,
    "qwen": create_qwen_template,
}


def get_template(template_name: str) -> Template:
    """Get template by name"""
    if template_name not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    
    return TEMPLATE_REGISTRY[template_name]()


def register_template(name: str, template_factory: callable) -> None:
    """Register a new template"""
    TEMPLATE_REGISTRY[name] = template_factory


def list_available_templates() -> List[str]:
    """List all available template names"""
    return list(TEMPLATE_REGISTRY.keys())