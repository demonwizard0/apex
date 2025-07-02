from loguru import logger

from ..models import Tool


async def search_web(question: str, n_results: int = 2, completions=None) -> dict:
    """
    Takes a natural language question, generates an optimized search query, performs web search,
    and returns a referenced answer based on the search results.

    Args:
        question: The natural language question to answer
        n_results: Number of search results to retrieve
        completions: Function to make completions request

    Returns:
        dict containing the answer, references, and search metadata
    """
    try:
        from validator_api.serializers import WebRetrievalRequest
        from validator_api.web_retrieval import web_retrieval
    except ImportError as e:
        logger.error(f"Web retrieval dependencies not available: {e}")
        return {
            "question": question,
            "optimized_query": question,
            "answer": "Web search functionality is not available due to missing dependencies.",
            "references": [],
            "raw_results": [],
            "error": str(e),
        }

    if completions is None:
        optimized_query = question
    else:
        try:
            # TODO: Implement query optimization with completions
            # query_prompt = """Given a natural language question, generate an optimized web search query.
            # Focus on extracting key terms and concepts while removing unnecessary words.
            # Format your response as a single line containing only the optimized search query."""
            # messages = [{"role": "system", "content": query_prompt}, {"role": "user", "content": question}]
            optimized_query = question  # Placeholder
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            optimized_query = question

    search_results = None
    max_retries = 3

    for i in range(max_retries):
        try:
            search_request = WebRetrievalRequest(search_query=optimized_query, n_results=n_results)
            search_results = await web_retrieval(search_request)
            if search_results and hasattr(search_results, "results") and search_results.results:
                break
        except Exception as e:
            logger.warning(f"Web search attempt {i+1} failed: {e}")
            if i == max_retries - 1:
                return {
                    "question": question,
                    "optimized_query": optimized_query,
                    "answer": f"Web search failed after {max_retries} attempts: {str(e)}",
                    "references": [],
                    "raw_results": [],
                    "error": str(e),
                }

    if not search_results or not hasattr(search_results, "results") or not search_results.results:
        return {
            "question": question,
            "optimized_query": optimized_query,
            "answer": "No search results found.",
            "references": [],
            "raw_results": [],
        }

    if completions is None:
        answer_data = {
            "answer": f"Search results found for: {question}. "
            + " ".join([f"[{i+1}] {result.content[:200]}..." for i, result in enumerate(search_results.results)]),
            "references": [{"number": i + 1, "url": result.url} for i, result in enumerate(search_results.results)],
        }
    else:
        try:
            # TODO: Use completions to generate referenced answer with this prompt:
            # answer_prompt = f"""Based on the provided search results, generate a concise but well-structured answer to the question.
            # Include inline references to sources using markdown format [n] where n is the source number.
            # Question: {question}
            # Search Results: {json.dumps([{'index': i + 1, 'content': result.content, 'url': result.url} for i, result in enumerate(search_results.results)], indent=2)}
            # Format your response as a JSON object with the following structure:
            # {{"answer": "Your detailed answer with inline references [n]", "references": [{{"number": n, "url": "Source URL"}}]}}"""
            answer_data = {
                "answer": f"Based on search results for: {question}",
                "references": [{"number": i + 1, "url": result.url} for i, result in enumerate(search_results.results)],
            }
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            answer_data = {"answer": f"Search completed but answer generation failed: {str(e)}", "references": []}

    return {
        "question": question,
        "optimized_query": optimized_query,
        "answer": answer_data["answer"],
        "references": answer_data["references"],
        "raw_results": [{"snippet": r.content, "url": r.url} for r in search_results.results],
    }


class WebSearchTool(Tool):
    """Tool for performing web searches and getting referenced answers"""

    def __init__(self, completions=None):
        self.completions = completions

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return """Searches the web to answer a question. Provides a referenced answer with citations.
Input parameters:
- question: The natural language question to answer
- n_results: (optional) Number of search results to use (default: 2)

Returns a dictionary containing:
- question: Original question asked
- optimized_query: Search query used
- answer: Detailed answer with inline references [n]
- references: List of numbered references with URLs
- raw_results: Raw search results used"""

    async def execute(self, question: str, n_results: int = 2) -> dict:
        return await search_web(question=question, n_results=n_results, completions=self.completions)
