"""
Agent Orchestrator Service

Manages custom system prompts and orchestrates different AI agents for enhanced fine-tuned model interactions.
Provides context-aware prompt engineering and agent selection for optimal model performance.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from utils.logging import get_logger
from config.settings import settings


class PromptTemplate:
    """Template for system prompts with dynamic variables"""
    
    def __init__(self, name: str, template: str, description: str, variables: List[str] = None):
        self.name = name
        self.template = template
        self.description = description
        self.variables = variables or []
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")


class AgentProfile:
    """Defines an AI agent with specific capabilities and system prompt"""
    
    def __init__(self, agent_id: str, name: str, role: str, system_prompt: str, 
                 capabilities: List[str], use_cases: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.capabilities = capabilities
        self.use_cases = use_cases
        self.usage_count = 0
        self.last_used = None


class AgentOrchestrator:
    """Orchestrates AI agents and manages custom system prompts for fine-tuned models"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.agents: Dict[str, AgentProfile] = {}
        self.session_agents: Dict[str, str] = {}  # session_id -> agent_id
        self.session_prompts: Dict[str, str] = {}  # session_id -> custom_prompt
        
        # Load default templates and agents
        self._load_default_templates()
        self._load_default_agents()
        
        self.logger.info("Agent Orchestrator initialized")
    
    def _load_default_templates(self):
        """Load default system prompt templates"""
        default_templates = {
            "helpful_assistant": PromptTemplate(
                name="Helpful Assistant",
                template="You are a helpful, knowledgeable assistant that provides detailed and practical responses. Always be specific and helpful. {context}",
                description="General purpose helpful assistant",
                variables=["context"]
            ),
            "coding_expert": PromptTemplate(
                name="Coding Expert",
                template="You are an expert software developer with deep knowledge of {language} programming. Provide clean, efficient, and well-documented code solutions. Focus on best practices and explain your reasoning. {context}",
                description="Expert coding assistant for programming tasks",
                variables=["language", "context"]
            ),
            "data_analyst": PromptTemplate(
                name="Data Analyst",
                template="You are a skilled data analyst with expertise in {domain} analysis. Provide insights backed by data and statistical reasoning. Always explain your methodology and assumptions. {context}",
                description="Data analysis and insights specialist",
                variables=["domain", "context"]
            ),
            "creative_writer": PromptTemplate(
                name="Creative Writer",
                template="You are a creative writer with expertise in {genre} writing. Craft engaging, imaginative content that captures the reader's attention. Use vivid descriptions and compelling narratives. {context}",
                description="Creative writing specialist",
                variables=["genre", "context"]
            ),
            "research_assistant": PromptTemplate(
                name="Research Assistant",
                template="You are a thorough research assistant specializing in {field} research. Provide comprehensive, well-sourced information and analysis. Always cite your reasoning and acknowledge limitations. {context}",
                description="Academic and research support specialist",
                variables=["field", "context"]
            ),
            "custom_domain": PromptTemplate(
                name="Custom Domain Expert",
                template="You are an expert in {domain} with {years} years of experience. Your expertise includes {specializations}. {personality_traits} {context}",
                description="Customizable domain expert template",
                variables=["domain", "years", "specializations", "personality_traits", "context"]
            )
        }
        
        for template_id, template in default_templates.items():
            self.prompt_templates[template_id] = template
            
        self.logger.info(f"Loaded {len(default_templates)} default prompt templates")
    
    def _load_default_agents(self):
        """Load default agent profiles"""
        default_agents = {
            "general_assistant": AgentProfile(
                agent_id="general_assistant",
                name="General Assistant",
                role="helpful_assistant",
                system_prompt="You are a helpful, knowledgeable assistant that provides detailed and practical responses. Always be specific and helpful.",
                capabilities=["general_qa", "explanations", "recommendations"],
                use_cases=["general questions", "explanations", "advice"]
            ),
            "code_reviewer": AgentProfile(
                agent_id="code_reviewer",
                name="Code Reviewer",
                role="coding_expert",
                system_prompt="You are an expert code reviewer with deep knowledge of software development best practices. Review code for quality, security, performance, and maintainability. Provide specific, actionable feedback.",
                capabilities=["code_review", "security_analysis", "performance_optimization"],
                use_cases=["code review", "debugging", "optimization"]
            ),
            "technical_writer": AgentProfile(
                agent_id="technical_writer",
                name="Technical Writer",
                role="documentation_expert",
                system_prompt="You are a technical writer specializing in clear, comprehensive documentation. Create well-structured documentation that is easy to understand for the target audience.",
                capabilities=["documentation", "technical_writing", "user_guides"],
                use_cases=["documentation", "tutorials", "API docs"]
            ),
            "business_analyst": AgentProfile(
                agent_id="business_analyst",
                name="Business Analyst",
                role="data_analyst",
                system_prompt="You are a business analyst with expertise in data-driven decision making. Analyze business problems and provide strategic insights backed by data and market knowledge.",
                capabilities=["business_analysis", "market_research", "strategic_planning"],
                use_cases=["business planning", "market analysis", "strategy"]
            ),
            "creative_companion": AgentProfile(
                agent_id="creative_companion",
                name="Creative Companion",
                role="creative_writer",
                system_prompt="You are a creative companion that helps with brainstorming, creative writing, and artistic projects. Think outside the box and inspire innovative ideas.",
                capabilities=["creative_writing", "brainstorming", "idea_generation"],
                use_cases=["creative writing", "brainstorming", "content creation"]
            )
        }
        
        for agent_id, agent in default_agents.items():
            self.agents[agent_id] = agent
            
        self.logger.info(f"Loaded {len(default_agents)} default agent profiles")
    
    def set_session_agent(self, session_id: str, agent_id: str) -> bool:
        """Set the active agent for a session"""
        if agent_id not in self.agents:
            self.logger.error(f"Agent {agent_id} not found")
            return False
        
        self.session_agents[session_id] = agent_id
        agent = self.agents[agent_id]
        agent.usage_count += 1
        agent.last_used = datetime.now().isoformat()
        
        self.logger.info(f"Set agent {agent_id} for session {session_id}")
        return True
    
    def set_session_custom_prompt(self, session_id: str, custom_prompt: str) -> bool:
        """Set a custom system prompt for a session"""
        if not custom_prompt or not custom_prompt.strip():
            self.logger.error("Custom prompt cannot be empty")
            return False
        
        self.session_prompts[session_id] = custom_prompt.strip()
        self.logger.info(f"Set custom prompt for session {session_id}")
        return True
    
    def get_system_prompt_for_session(self, session_id: str, context: str = "") -> str:
        """Get the appropriate system prompt for a session"""
        # Priority 1: Custom prompt
        if session_id in self.session_prompts:
            prompt = self.session_prompts[session_id]
            if context:
                prompt += f"\n\nContext: {context}"
            return prompt
        
        # Priority 2: Agent-based prompt
        if session_id in self.session_agents:
            agent_id = self.session_agents[session_id]
            agent = self.agents.get(agent_id)
            if agent:
                prompt = agent.system_prompt
                if context:
                    prompt += f"\n\nContext: {context}"
                return prompt
        
        # Fallback: Default helpful assistant
        default_prompt = "You are a helpful, knowledgeable assistant that provides detailed and practical responses. Always be specific and helpful."
        if context:
            default_prompt += f"\n\nContext: {context}"
        return default_prompt
    
    def suggest_agent_for_query(self, query: str) -> Optional[str]:
        """Suggest the most appropriate agent based on query analysis"""
        query_lower = query.lower()
        
        # Simple keyword-based agent suggestion
        if any(keyword in query_lower for keyword in ["code", "programming", "bug", "debug", "function"]):
            return "code_reviewer"
        elif any(keyword in query_lower for keyword in ["write", "story", "creative", "poem", "brainstorm"]):
            return "creative_companion"
        elif any(keyword in query_lower for keyword in ["business", "market", "strategy", "analyze", "data"]):
            return "business_analyst"
        elif any(keyword in query_lower for keyword in ["documentation", "tutorial", "explain", "guide"]):
            return "technical_writer"
        else:
            return "general_assistant"
    
    def create_custom_agent(self, agent_id: str, name: str, system_prompt: str, 
                           capabilities: List[str], use_cases: List[str]) -> bool:
        """Create a new custom agent"""
        if agent_id in self.agents:
            self.logger.error(f"Agent {agent_id} already exists")
            return False
        
        agent = AgentProfile(
            agent_id=agent_id,
            name=name,
            role="custom",
            system_prompt=system_prompt,
            capabilities=capabilities,
            use_cases=use_cases
        )
        
        self.agents[agent_id] = agent
        self.logger.info(f"Created custom agent: {agent_id}")
        return True
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "role": agent.role,
            "system_prompt": agent.system_prompt,
            "capabilities": agent.capabilities,
            "use_cases": agent.use_cases,
            "usage_count": agent.usage_count,
            "last_used": agent.last_used
        }
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available agents"""
        return [self.get_agent_info(agent_id) for agent_id in self.agents.keys()]
    
    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available prompt templates"""
        return [
            {
                "template_id": template_id,
                "name": template.name,
                "description": template.description,
                "variables": template.variables
            }
            for template_id, template in self.prompt_templates.items()
        ]
    
    def render_template(self, template_id: str, **variables) -> str:
        """Render a prompt template with variables"""
        template = self.prompt_templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        return template.render(**variables)
    
    def get_session_config(self, session_id: str) -> Dict[str, Any]:
        """Get current session configuration"""
        config = {
            "session_id": session_id,
            "has_custom_prompt": session_id in self.session_prompts,
            "has_agent": session_id in self.session_agents,
            "current_prompt": self.get_system_prompt_for_session(session_id)
        }
        
        if session_id in self.session_agents:
            agent_info = self.get_agent_info(self.session_agents[session_id])
            config["current_agent"] = agent_info
        
        return config
    
    def clear_session_config(self, session_id: str):
        """Clear session configuration"""
        if session_id in self.session_agents:
            del self.session_agents[session_id]
        if session_id in self.session_prompts:
            del self.session_prompts[session_id]
        
        self.logger.info(f"Cleared configuration for session {session_id}")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for agents"""
        total_usage = sum(agent.usage_count for agent in self.agents.values())
        
        agent_stats = []
        for agent in self.agents.values():
            usage_percentage = (agent.usage_count / total_usage * 100) if total_usage > 0 else 0
            agent_stats.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "usage_count": agent.usage_count,
                "usage_percentage": round(usage_percentage, 2),
                "last_used": agent.last_used
            })
        
        # Sort by usage count
        agent_stats.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "total_sessions": len(set(list(self.session_agents.keys()) + list(self.session_prompts.keys()))),
            "total_agent_usage": total_usage,
            "custom_prompt_sessions": len(self.session_prompts),
            "agent_based_sessions": len(self.session_agents),
            "agent_statistics": agent_stats
        }


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()