import json
import re
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Awaitable, Callable

from fastapi.responses import StreamingResponse
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from validator_api.serializers import CompletionsRequest, WebRetrievalRequest
from validator_api.web_retrieval import web_retrieval

from .models import LLMQuery

STEP_MAX_RETRIES = 10


def parse_llm_json(json_str: str, allow_empty: bool = True) -> dict[str, Any]:
    """Parse JSON output from LLM that may contain code blocks, newlines and other formatting.

    Extracts JSON from code blocks if present, or finds JSON objects/arrays within text.

    Args:
        json_str (str): The JSON string to parse.
        allow_empty (bool): Whether to allow empty JSON objects.

    Returns:
        dict: The parsed JSON object
    """
    # First try to extract JSON from code blocks if they exist.
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_block_matches = re.findall(code_block_pattern, json_str)
    if code_block_matches:
        # Use the first code block found.
        json_str = code_block_matches[0]
    else:
        # Try to find JSON objects or arrays within the string.
        json_candidates = []

        # Look for JSON objects {...}.
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        object_matches = re.findall(brace_pattern, json_str)
        json_candidates.extend(object_matches)

        # Look for JSON arrays [...].
        bracket_pattern = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
        array_matches = re.findall(bracket_pattern, json_str)
        json_candidates.extend(array_matches)

        # Try to parse each candidate and use the first valid one.
        for candidate in json_candidates:
            try:
                candidate = candidate.strip()
                json.loads(candidate)
                json_str = candidate
                break
            except json.JSONDecodeError:
                continue
        else:
            # If no valid JSON found in candidates, try the original string.
            pass

    # Replace escaped newlines with actual newlines.
    json_str = json_str.replace("\\n", "\n")

    # Remove any redundant newlines/whitespace while preserving content.
    json_str = " ".join(line.strip() for line in json_str.splitlines())

    # Parse the cleaned JSON string.
    result = json.loads(json_str)

    if not allow_empty and not result:
        raise json.JSONDecodeError("Empty JSON string", json_str, 0)

    return result


def with_retries(max_retries: int = 3):
    """
    A decorator that retries a function on failure and logs attempts using loguru.

    Args:
        max_retries (int): Maximum number of retry attempts before giving up
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Get the full stack trace
                    stack_trace = traceback.format_exc()
                    # If this is the last attempt, log as critical with full stack trace
                    if attempt == max_retries - 1:
                        logger.exception(
                            f"Function '{func.__name__}' failed on final attempt {attempt + 1}/{max_retries}. "
                            f"Error: {str(e)}\nStack trace:\n{stack_trace}"
                        )
                        raise  # Re-raise the exception after logging
                    # Otherwise log as error without stack trace
                    logger.error(
                        f"Function '{func.__name__}' failed on attempt {attempt + 1}/{max_retries}. "
                        f"Error: {str(e)}. Retrying..."
                    )
            return None  # In case all retries fail

        return wrapper

    return decorator


def convert_to_gemma_messages(messages):
    """Convert a list of messages to a list of gemma messages by alternating roles and adding empty messages."""
    gemma_messages = []
    for message in messages:
        if gemma_messages and gemma_messages[-1]["role"] == message["role"]:
            # Gemma requires alternating roles, so we need to add an empty message with the opposite role
            gemma_messages.append(
                {"type": "text", "content": "", "role": "assistant" if message["role"] == "user" else "user"}
            )
        gemma_messages.append({"type": "text", "role": message["role"], "content": message["content"]})
    return gemma_messages


async def extract_content_from_stream(streaming_response: StreamingResponse) -> str:
    full_content = ""

    async for chunk in streaming_response.body_iterator:
        # Decode bytes to string if necessary
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")

        # Remove any 'data: ' prefixes and skip empty lines
        for line in chunk.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            try:
                data = json.loads(line.removeprefix("data:").strip())
                delta = data.get("choices", [{}])[0].get("delta", {})
                content_piece = delta.get("content")
                if content_piece:
                    full_content += content_piece
            except json.JSONDecodeError:
                continue  # Optionally log/handle malformed chunks

    return full_content


def make_chunk(text: str) -> str:
    """Create a streaming chunk for SSE response"""
    chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
    return f"data: {chunk}\n\n"


def get_current_datetime_str() -> str:
    """Returns a nicely formatted string of the current date and time"""
    return datetime.now().strftime("%B %d, %Y")


@retry(
    stop=stop_after_attempt(STEP_MAX_RETRIES),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(json.JSONDecodeError),
)
async def make_mistral_request_with_json(
    messages: list[dict[str, Any]],
    step_name: str,
    completions: Callable[[CompletionsRequest], Awaitable[StreamingResponse]],
) -> tuple[str, LLMQuery]:
    """Makes a request to Mistral API and ensures response is valid JSON"""

    raw_response, query_record = await make_mistral_request(
        messages,
        step_name,
        completions,
    )
    try:
        parse_llm_json(raw_response)  # Test if the response is jsonable
        return raw_response, query_record
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Mistral API response as JSON: {e}")
        raise


@retry(
    stop=stop_after_attempt(STEP_MAX_RETRIES),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(BaseException),
)
async def make_mistral_request(
    messages: list[dict[str, Any]],
    step_name: str,
    completions: Callable[[CompletionsRequest], Awaitable[StreamingResponse]],
) -> tuple[str, LLMQuery]:
    """Makes a request to Mistral API and records the query"""

    model = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"
    temperature = 0.15
    top_p = 1
    max_tokens = 6144
    sample_params: dict[str, Any] = {
        "top_p": top_p,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": False,
    }
    request = CompletionsRequest(
        messages=messages,
        model=model,
        stream=True,
        sampling_parameters=sample_params,
        timeout=120,
    )
    response = await completions(request)

    response_content = await extract_content_from_stream(response)

    logger.debug(f"Response content: {response_content}")
    if not response_content:
        raise ValueError(f"No response content received from Mistral API, response: {response}")
    if "Error" in response_content:
        raise ValueError(f"Error in Mistral API response: {response_content}")

    query_record = LLMQuery(
        messages=messages, raw_response=response_content, step_name=step_name, timestamp=time.time(), model=model
    )

    return response_content, query_record


async def search_web(question: str, n_results: int = 2, completions=None) -> dict:
    """
    Takes a natural language question, generates an optimized search query, performs web search,
    and returns a referenced answer based on the search results.
    """
    # Generate optimized search query
    query_prompt = """Given a natural language question, generate an optimized web search query.
Focus on extracting key terms and concepts while removing unnecessary words.
Format your response as a single line containing only the optimized search query."""

    messages = [{"role": "system", "content": query_prompt}, {"role": "user", "content": question}]

    optimized_query, query_record = await make_mistral_request(
        messages, "optimize_search_query", completions=completions
    )

    # Perform web search
    search_results = None
    for i in range(STEP_MAX_RETRIES):
        try:
            search_results = await web_retrieval(WebRetrievalRequest(search_query=optimized_query, n_results=n_results))
            if search_results.results:
                break
        except BaseException:
            logger.warning(f"Try {i+1} failed")
    if search_results is None or not search_results.results:
        search_results = {"results": []}

    # Generate referenced answer
    answer_prompt = f"""Based on the provided search results, generate a concise but well-structured answer to the question.
Include inline references to sources using markdown format [n] where n is the source number.

Question: {question}

Search Results:
{json.dumps([{
    'index': i + 1,
    'content': result.content,
    'url': result.url
} for i, result in enumerate(search_results.results)], indent=2)}

Format your response as a JSON object with the following structure:
{{
    "answer": "Your detailed answer with inline references [n]",
    "references": [
        {{
            "number": n,
            "url": "Source URL"
        }}
    ]
}}"""

    messages = [
        {"role": "system", "content": answer_prompt},
        {"role": "user", "content": "Please generate a referenced answer based on the search results."},
    ]

    raw_answer, answer_record = await make_mistral_request_with_json(
        messages, "generate_referenced_answer", completions=completions
    )
    try:
        answer_data = parse_llm_json(raw_answer)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Mistral API response as JSON: {e}")
        raise

    return {
        "question": question,
        "optimized_query": optimized_query,
        "answer": answer_data["answer"],
        "references": answer_data["references"],
        "raw_results": [{"snippet": r.content, "url": r.url} for r in search_results.results],
    }
