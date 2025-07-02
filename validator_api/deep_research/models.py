from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

from pydantic import BaseModel


class LLMQuery(BaseModel):
    """Records a single LLM API call with its inputs and outputs"""

    messages: list[dict[str, Any]]  # The input messages
    raw_response: str  # The raw response from the LLM
    parsed_response: Any | None = None  # The parsed response (if applicable)
    step_name: str  # Name of the step that made this query
    timestamp: float  # When the query was made
    model: str  # Which model was used


class Step(BaseModel):
    title: str
    content: str
    next_step: str | None = None
    summary: str | None = None

    def __str__(self):
        return f"Title: {self.title}\nContent: {self.content}\nNext Step: {self.next_step}\nSummary: {self.summary}"


class Tool(ABC):
    """Base class for tools that can be used by the orchestrator"""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does and how to use it"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters"""
        pass


class ToolRequest(BaseModel):
    """A request to execute a specific tool"""

    tool_name: str
    parameters: dict
    purpose: str  # Why this tool execution is needed for the current step


class ToolResult(BaseModel):
    """Result of executing a tool"""

    tool_name: str
    parameters: dict
    result: Any
    purpose: str


# LangGraph State Definition
class ResearchState(TypedDict):
    """State that flows through the LangGraph workflow"""

    user_messages: list[dict[str, Any]]
    question: str
    is_suitable_for_research: bool
    direct_answer: Optional[str]
    todo_list: Optional[str]
    current_step: int
    max_steps: int
    completed_steps: list[Step]
    query_history: list[LLMQuery]
    tool_history: list[ToolResult]
    final_answer: Optional[dict]
    should_continue: bool
    stream_output: list[str]  # Collect streaming output
