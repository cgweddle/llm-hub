from typing import List
import requests
from langchain_community.document_loaders import BraveSearchLoader
import os

def brave_search_function(query: str) -> str:
    """
    Search the web for current information about a given query using Brave Search.
    Args:
        query: The search query to look up on the web
    Returns:
        A string containing relevant search results and information
    """
    try:
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return "Brave Search API key not set. Please set BRAVE_SEARCH_API_KEY environment variable."
        loader = BraveSearchLoader(query=query, api_key=api_key, search_kwargs={"count": 5})
        docs = loader.load()
        if not docs:
            return f"No results found for query: {query}"
        formatted_results = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'No title')
            url = doc.metadata.get('link', 'No url')
            snippet = doc.page_content[:300] if doc.page_content else ''
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error performing Brave search: {str(e)}"
