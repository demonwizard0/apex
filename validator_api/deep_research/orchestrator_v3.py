import time
from typing import Any, Awaitable, Callable, Optional

from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from .models import ResearchState, Tool, ToolResult


def make_chunk(content: str) -> str:
    """Simple chunk maker - moved inline to avoid utils.py dependencies"""
    return f"data: {content}\n\n"


from validator_api.serializers import CompletionsRequest

from .nodes_clean import (
    analyze_step_node,
    assess_question_node,
    direct_answer_node,
    execute_tools_node,
    generate_answer_node,
    generate_todo_node,
    plan_tools_node,
    update_todo_node,
)


class OrchestratorV3:
    """LangGraph-based orchestrator for deep research"""

    def __init__(self, completions: Optional[Callable[[CompletionsRequest], Awaitable[StreamingResponse]]] = None):
        self.completions = completions
        self.tools = {}
        self.graph = None

        # Register default web search tool
        try:
            from .tools import WebSearchTool

            self.register_tool(WebSearchTool(completions=completions))
        except ImportError as e:
            logger.warning(f"WebSearchTool not available: {e}")

        # Build initial graph
        self._build_graph()

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool dynamically and rebuild the graph"""
        logger.info(f"Registering tool: {tool.name}")
        self.tools[tool.name] = tool

        if self.graph is not None:
            self._build_graph()

    def unregister_tool(self, tool_name: str) -> bool:
        """Remove a tool and rebuild the graph"""
        if tool_name in self.tools:
            logger.info(f"Unregistering tool: {tool_name}")
            del self.tools[tool_name]
            self._build_graph()
            return True
        return False

    def list_tools(self) -> dict[str, str]:
        """List all registered tools and their descriptions"""
        return {name: tool.description for name, tool in self.tools.items()}

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name"""
        return self.tools.get(name)

    def _build_graph(self) -> None:
        """Build the LangGraph workflow with dynamic tool integration"""
        logger.info(f"Building graph with {len(self.tools)} registered tools")

        workflow = StateGraph(ResearchState)
        workflow.add_node("assess_question", lambda state: assess_question_node(state, self.completions))
        workflow.add_node("generate_todo", lambda state: generate_todo_node(state, self.completions))
        workflow.add_node("plan_tools", lambda state: plan_tools_node(state, self.completions, self.tools))
        workflow.add_node("execute_tools", lambda state: execute_tools_node(state, self.tools))
        workflow.add_node("analyze_step", lambda state: analyze_step_node(state, self.completions))
        workflow.add_node("update_todo", lambda state: update_todo_node(state, self.completions))
        workflow.add_node("generate_answer", lambda state: generate_answer_node(state, self.completions))
        workflow.add_node("provide_direct_answer", direct_answer_node)

        for tool_name, tool in self.tools.items():
            if hasattr(tool, "requires_custom_node") and tool.requires_custom_node:
                node_name = f"tool_{tool_name}"
                workflow.add_node(node_name, self._create_tool_node(tool))
                logger.info(f"Added custom node for tool: {tool_name}")

        workflow.add_edge(START, "assess_question")
        workflow.add_conditional_edges(
            "assess_question",
            self._should_do_research,
            {"research": "generate_todo", "direct": "provide_direct_answer"},
        )

        workflow.add_edge("generate_todo", "plan_tools")
        workflow.add_edge("plan_tools", "execute_tools")
        workflow.add_edge("execute_tools", "analyze_step")
        workflow.add_edge("analyze_step", "update_todo")
        workflow.add_conditional_edges(
            "update_todo", self._should_continue_research, {"continue": "plan_tools", "finish": "generate_answer"}
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("provide_direct_answer", END)

        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)
        logger.info("Graph compilation complete")

    def _create_tool_node(self, tool: Tool):
        """Create a specialized node for a tool that requires custom workflow"""

        async def tool_node(state: ResearchState) -> ResearchState:
            logger.info(f"Executing custom node for tool: {tool.name}")

            try:
                tool_params = state.get(f"{tool.name}_params", {})
                result = await tool.execute(**tool_params)

                if "tool_history" not in state:
                    state["tool_history"] = []

                tool_result = ToolResult(
                    tool_name=tool.name,
                    parameters=tool_params,
                    result=result,
                    purpose=f"Custom execution of {tool.name}",
                )
                state["tool_history"].append(tool_result)

                if "stream_output" not in state:
                    state["stream_output"] = []
                state["stream_output"].append(f"\n### Custom Tool Execution\n{tool.name} completed successfully")

            except Exception as e:
                logger.error(f"Custom tool node failed for {tool.name}: {e}")
                if "stream_output" not in state:
                    state["stream_output"] = []
                state["stream_output"].append(f"\n### Tool Error\n{tool.name} failed: {str(e)}")

            return state

        return tool_node

    def _should_do_research(self, state: ResearchState) -> str:
        """Conditional edge function to determine if research is needed"""
        return "research" if state.get("is_suitable_for_research", True) else "direct"

    def _should_continue_research(self, state: ResearchState) -> str:
        """Conditional edge function to determine if research should continue"""
        should_continue = state.get("should_continue", True) and state.get("current_step", 0) < state.get(
            "max_steps", 10
        )
        return "continue" if should_continue else "finish"

    async def run(self, messages: list[dict[str, Any]]):
        """Run the orchestrator with streaming output"""
        logger.info("Starting LangGraph orchestration run")
        initial_state = ResearchState(
            user_messages=messages,
            question="",
            is_suitable_for_research=True,
            direct_answer=None,
            todo_list=None,
            current_step=0,
            max_steps=10,
            completed_steps=[],
            query_history=[],
            tool_history=[],
            final_answer=None,
            should_continue=True,
            stream_output=[],
        )

        # Create a unique thread ID for this conversation
        config = {"configurable": {"thread_id": f"research_{int(time.time())}"}}

        # Run the graph
        async for output in self.graph.astream(initial_state, config=config):
            # Extract the state from each node output
            for node_name, node_state in output.items():
                # Stream any new output that was added
                stream_output = node_state.get("stream_output", [])
                for chunk in stream_output:
                    yield make_chunk(chunk + "\n")

        yield "data: [DONE]\n\n"
