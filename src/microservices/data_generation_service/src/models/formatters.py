"""
Formatter Classes for Template System

Based on LLaMA-Factory's formatter architecture for standardizing
diverse dataset formats into consistent conversation templates.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class SLOTS:
    """Template slot placeholders"""
    system: str = "{{system}}"
    user: str = "{{user}}" 
    assistant: str = "{{assistant}}"
    function: str = "{{function}}"
    observation: str = "{{observation}}"
    tools: str = "{{tools}}"
    prefix: str = "{{prefix}}"


class Formatter(ABC):
    """Base formatter interface for template content"""
    
    @abstractmethod
    def apply(self, content: str, **kwargs) -> str:
        """Apply formatting to content"""
        pass


class EmptyFormatter(Formatter):
    """Fallback formatter that returns empty string"""
    
    def apply(self, content: str, **kwargs) -> str:
        return ""


class StringFormatter(Formatter):
    """Simple string template formatter with slot replacement"""
    
    def __init__(self, template: str):
        self.template = template
    
    def apply(self, content: str, **kwargs) -> str:
        """Apply string formatting with slot replacement"""
        result = self.template
        
        # Replace content slots
        result = result.replace(SLOTS.system, kwargs.get('system', ''))
        result = result.replace(SLOTS.user, kwargs.get('user', content))
        result = result.replace(SLOTS.assistant, kwargs.get('assistant', content))
        result = result.replace(SLOTS.function, kwargs.get('function', content))
        result = result.replace(SLOTS.observation, kwargs.get('observation', content))
        result = result.replace(SLOTS.tools, kwargs.get('tools', ''))
        result = result.replace(SLOTS.prefix, kwargs.get('prefix', ''))
        
        return result


class FunctionFormatter(Formatter):
    """Formatter for function calling content"""
    
    def __init__(self, template: str):
        self.template = template
    
    def apply(self, content: str, **kwargs) -> str:
        """Format function calling content"""
        # Handle function call formatting
        if isinstance(content, dict):
            if 'function_call' in content:
                func_call = content['function_call']
                name = func_call.get('name', '')
                arguments = func_call.get('arguments', '{}')
                
                formatted = self.template.replace('{{function_name}}', name)
                formatted = formatted.replace('{{function_args}}', arguments)
                return formatted
        
        return self.template.replace('{{function}}', str(content))


class ToolFormatter(Formatter):
    """Formatter for tool usage content"""
    
    def __init__(self, template: str = ""):
        self.template = template or "Tools: {{tools}}"
    
    def apply(self, content: str, **kwargs) -> str:
        """Format tool usage content"""
        tools = kwargs.get('tools', [])
        
        if isinstance(tools, list) and tools:
            tools_str = "\n".join([
                f"- {tool['name']}: {tool.get('description', '')}" 
                for tool in tools
            ])
        else:
            tools_str = str(tools) if tools else ""
        
        return self.template.replace('{{tools}}', tools_str)


def get_formatter(template: str, formatter_type: str = "string") -> Formatter:
    """Factory function to create formatters"""
    
    if formatter_type == "empty" or not template.strip():
        return EmptyFormatter()
    elif formatter_type == "function":
        return FunctionFormatter(template)
    elif formatter_type == "tool":
        return ToolFormatter(template)
    else:
        return StringFormatter(template)