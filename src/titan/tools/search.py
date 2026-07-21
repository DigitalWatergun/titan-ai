import logging
import os

import httpx
import trafilatura
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")
MAX_CONTENT_CHARS = 20000
MAX_FETCH_BYTES = 2_000_000
FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TitanAgent/1.0)"}


class WebSearchInput(BaseModel):
    """Search the web for current information using SearXNG.
    Use this when you need up-to-date information, documentation, or anything
    not found in the local codebase."""

    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")


class WebFetchInput(BaseModel):
    """Fetch a web page and return its main content as clean markdown.
    Use this after web_search to read the full body of a promising result -
    web_search returns only the titles and short snippets, not the page contents.
    Also use it when the user gives you a URL to read directly."""

    url: str = Field(description="The URL of the page to fetch")


def web_search(args: WebSearchInput) -> str:
    try:
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": args.query, "format": "json", "categories": "general"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", [])[: args.num_results]:
            results.append(
                f"**{r['title']}**\n{r.get('content', 'No snippet')}\nURL: {r['url']}\n"
            )

        return "\n".join(results) if results else "No results found."
    except Exception as e:
        logger.exception(f"Web search failed for query: {args.query}")
        return f"Search error: {e}"


def web_fetch(args: WebFetchInput) -> str:
    try:
        with httpx.stream(
            "GET",
            args.url,
            headers=FETCH_HEADERS,
            timeout=15,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "text/html").lower()
            if "html" not in content_type and not content_type.startswith("text/"):
                return f"Unsupported content type ({content_type}): {args.url}"
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_FETCH_BYTES:
                    break
        html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")

        content = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            include_comments=False,
        )

        if not content:
            return f"Could not extract readable content from {args.url}"

        if len(content) > MAX_CONTENT_CHARS:
            content = (
                content[:MAX_CONTENT_CHARS]
                + f"\n\n[Truncated at {MAX_CONTENT_CHARS} characters]"
            )

        return content
    except Exception as e:
        logger.exception(f"Web fetch failed for URL: {args.url}")
        return f"Fetch error: {e}"
