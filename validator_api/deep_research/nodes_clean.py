import asyncio
import json
from datetime import datetime
from typing import Any

from loguru import logger

from .models import ResearchState, ToolRequest, ToolResult


def get_current_datetime_str() -> str:
    """Returns a nicely formatted string of the current date and time"""
    return datetime.now().strftime("%B %d, %Y")


def parse_llm_json(json_str: str, allow_empty: bool = True) -> dict[str, Any]:
    """Parse JSON output from LLM - simplified version"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to extract from code blocks
        if "```" in json_str:
            start = json_str.find("```")
            end = json_str.rfind("```")
            if start != -1 and end != -1 and start != end:
                json_content = json_str[start + 3 : end].strip()
                if json_content.startswith("json"):
                    json_content = json_content[4:].strip()
                return json.loads(json_content)
        raise


async def make_mistral_request(messages, step_name, completions):
    """Simplified mock request - replace with actual implementation"""
    return "Mock response", None


async def make_mistral_request_with_json(messages, step_name, completions):
    """Simplified mock JSON request - replace with actual implementation"""
    response, query_record = await make_mistral_request(messages, step_name, completions)
    return response, query_record


async def assess_question_node(state: ResearchState, completions) -> ResearchState:
    """Node to assess if question needs deep research"""
    logger.info("Assessing question suitability")

    question = state["user_messages"][-1]["content"]
    state["question"] = question

    state["is_suitable_for_research"] = True
    state["direct_answer"] = None
    state["query_history"] = []

    if "stream_output" not in state:
        state["stream_output"] = []

    state["stream_output"].append("Question assessed for research")

    return state


async def generate_todo_node(state: ResearchState, completions) -> ResearchState:
    """Node to generate initial todo list"""
    logger.info("Generating todo list")

    if "stream_output" not in state:
        state["stream_output"] = []
    state["stream_output"].append("## Generating Research Plan")

    todo_list = """
1. [Analysis]: Analyze the user's question
2. [Research]: Gather relevant information
3. [Synthesis]: Combine findings into answer
"""

    state["todo_list"] = todo_list
    state["current_step"] = 0
    state["max_steps"] = state.get("max_steps", 10)
    state["completed_steps"] = []
    state["tool_history"] = []
    if "query_history" not in state:
        state["query_history"] = []

    state["stream_output"].append(f"## Research Plan\n{todo_list}")

    return state


async def plan_tools_node(state: ResearchState, completions, tools) -> ResearchState:
    """Node to plan tool executions"""
    state["current_step"] += 1
    logger.info(f"Planning tools for step {state['current_step']}")

    if "stream_output" not in state:
        state["stream_output"] = []
    state["stream_output"].append(f"\n## Step {state['current_step']}: Planning Tools")

    planned_tools = []

    state["planned_tools"] = planned_tools
    logger.info(f"Planned {len(planned_tools)} tool executions")

    return state


async def execute_tools_node(state: ResearchState, tools) -> ResearchState:
    """Node to execute planned tools"""
    logger.info("Executing tools")

    planned_tools = state.get("planned_tools", [])
    if not planned_tools:
        return state

    if "stream_output" not in state:
        state["stream_output"] = []
    if "tool_history" not in state:
        state["tool_history"] = []

    async def execute_single_tool(request: ToolRequest) -> ToolResult | None:
        logger.info(f"Executing {request.tool_name} - Purpose: {request.purpose}")
        state["stream_output"].append(f"\n## Executing {request.tool_name}\n{request.purpose}")

        try:
            tool = tools.get(request.tool_name)
            if not tool:
                raise ValueError(f"Tool '{request.tool_name}' not found")

            result = await tool.execute(**request.parameters)

            return ToolResult(
                tool_name=request.tool_name, parameters=request.parameters, result=result, purpose=request.purpose
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                tool_name=request.tool_name,
                parameters=request.parameters,
                result={"error": str(e)},
                purpose=request.purpose,
            )

    tasks = [execute_single_tool(request) for request in planned_tools]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, ToolResult):
            state["tool_history"].append(result)
            state["stream_output"].append(f"Tool {result.tool_name} completed")
        else:
            logger.error(f"Tool execution error: {result}")

    return state


async def analyze_step_node(state: ResearchState, completions) -> ResearchState:
    """Node to analyze the current step"""
    logger.info("Analyzing step")

    if "stream_output" not in state:
        state["stream_output"] = []

    state["stream_output"].append("\n## Analyzing Step Results")

    tool_results = state.get("tool_history", [])
    if tool_results:
        analysis = f"Completed {len(tool_results)} tool executions"
    else:
        analysis = "No tools executed in this step"

    state["stream_output"].append(analysis)
    return state


async def update_todo_node(state: ResearchState, completions) -> ResearchState:
    """Node to update todo list and determine continuation"""
    logger.info("Updating todo list")

    if "stream_output" not in state:
        state["stream_output"] = []

    state["stream_output"].append("\n## Updating Research Plan")

    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 3)

    if current_step >= max_steps:
        state["should_continue"] = False
        state["stream_output"].append("Research plan completed")
    else:
        state["should_continue"] = True
        state["stream_output"].append("Continuing research")

    return state


async def generate_answer_node(state: ResearchState, completions) -> ResearchState:
    """Node to generate final answer"""
    logger.info("Generating final answer")

    if "stream_output" not in state:
        state["stream_output"] = []

    state["stream_output"].append("\n## Generating Final Answer")

    question = state.get("question", "Unknown question")
    tool_history = state.get("tool_history", [])

    answer = f"Based on analysis of: {question}\n"
    if tool_history:
        answer += f"Used {len(tool_history)} tools for research.\n"
    answer += "This is a simplified answer from the clean orchestrator."

    state["final_answer"] = {"answer": answer, "sources": [], "methodology": "Simplified analysis"}

    state["stream_output"].append(answer)
    return state


async def direct_answer_node(state: ResearchState) -> ResearchState:
    """Node to handle direct answers without research"""
    logger.info("Providing direct answer")

    if "stream_output" not in state:
        state["stream_output"] = []

    direct_answer = state.get("direct_answer", "This question can be answered directly.")
    state["stream_output"].append(f"## Direct Answer\n{direct_answer}")

    state["final_answer"] = {"answer": direct_answer, "sources": [], "methodology": "Direct response"}

    return state
