import os
import json
from typing import Literal
from langchain_core.tools import tool
from serpapi import GoogleSearch
from tavily import TavilyClient
from langchain_community.tools import DuckDuckGoSearchRun
import time


def _search_serp(queries: list[str], num_results: int) -> str:
    """Search using SerpAPI backend.
    
    Args:
        queries: List of search queries
        num_results: Number of results per query
        
    Returns:
        str: Formatted search results
    """
    
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return "ERROR: SERP_API_KEY not found in environment variables"
    
    all_results = []
    for query in queries:
        params = {
            "q": query,
            "api_key": api_key,
            "num": min(num_results, 10)
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results:
            for result in results["organic_results"][:num_results]:
                all_results.append({
                    "query": query,
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
    
    return json.dumps(all_results, indent=2)


def _search_tavily(queries: list[str], num_results: int) -> str:
    """Search using Tavily backend.
    
    Args:
        queries: List of search queries
        num_results: Number of results per query
        
    Returns:
        str: Formatted search results
    """
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "ERROR: TAVILY_API_KEY not found in environment variables"
    
    client = TavilyClient(api_key=api_key)
    all_results = []
    
    for query in queries:
        response = client.search(query=query, max_results=num_results)
        
        for result in response.get("results", [])[:num_results]:
            all_results.append({
                "query": query,
                "title": result.get("title", ""),
                "link": result.get("url", ""),
                "snippet": result.get("content", "")
            })
    
    return json.dumps(all_results, indent=2)


def _search_duckduckgo(queries: list[str], num_results: int) -> str:
    """Search using DuckDuckGo backend.
    
    Args:
        queries: List of search queries
        num_results: Number of results per query
        
    Returns:
        str: Formatted search results
    """
    
    search = DuckDuckGoSearchRun()
    all_results = []
    
    for query in queries:
        try:
            results = search.run(query)
            all_results.append({
                "query": query,
                "results": results
            })
            time.sleep(1)
        except Exception as e:
            error_msg = str(e).lower()
            if "ratelimit" in error_msg or "blocked" in error_msg or "429" in error_msg:
                return "ERROR: DuckDuckGo rate limited. Please try again later or use 'tavily' or 'serp' backend."
            all_results.append({
                "query": query,
                "error": str(e)
            })
    
    return json.dumps(all_results, indent=2)


@tool
def search_web(
    queries: list[str],
    backend: Literal["serp", "tavily", "duckduckgo"] = "duckduckgo",
    num_results: int = 5
) -> str:
    """Search the web using specified backend.
    
    Args:
        queries: List of search queries to execute
        backend: Search backend - 'serp', 'tavily', or 'duckduckgo'
        num_results: Number of results per query (default: 5)
        
    Returns:
        str: JSON formatted search results from all queries
    """
    if not queries:
        return "ERROR: No queries provided"
    
    if not isinstance(queries, list):
        queries = [queries]
    
    num_results = max(1, min(num_results, 20))
    
    try:
        if backend == "serp":
            return _search_serp(queries, num_results)
        elif backend == "tavily":
            return _search_tavily(queries, num_results)
        elif backend == "duckduckgo":
            return _search_duckduckgo(queries, num_results)
        else:
            return f"ERROR: Unknown backend '{backend}'. Use 'serp', 'tavily', or 'duckduckgo'"
    except Exception as e:
        return f"ERROR during search: {str(e)}"

