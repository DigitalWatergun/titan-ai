# src/titan/tools/search.py
import logging
import os

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")


@tool(args_schema=WebSearchInput)
def web_search(query: str, num_results: int = 3) -> str:
    """Search the web for current information using SearXNG.
    Use this when you need up-to-date information, documentation, or anything
    not found in the local codebase."""
    try:
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", [])[:num_results]:
            results.append(
                f"**{r['title']}**\n{r.get('content', 'No snippet')}\nURL: {r['url']}\n"
            )

        return "\n".join(results) if results else "No results found."
    except Exception as e:
        logger.exception(f"Web search failed for query: {query}")
        return f"Search error: {e}"
