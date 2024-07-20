import os
import aiohttp
import json

def format_response(res):
    """
    Format response from search_web function for reading by LLM.
    """
    return {
        "query": res.get("query"),
        "answer": res.get("answer"),
        "results": [{
            "title": result.get("title"),
            "summary": result.get("content"),
            "raw_content": result.get("raw_content"),
            "url": result.get("url"),
        } for result in res["results"]]
    }

async def search_web(query: str, include_answer: bool = True, search_depth: str = 'basic', include_raw_content: bool = False, max_results: int = 5) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.tavily.com/search',
            headers={
                'Content-Type': 'application/json',
            },
            data=json.dumps({
                'api_key': os.getenv('TAVILY_API_KEY'),
                'query': query,
                'include_answer': include_answer,
                'search_depth': search_depth,
                'include_raw_content': include_raw_content,
                'max_results': max_results,
            })
        ) as response:
            if response.status != 200:
                print("Error searching web: ", await response.json())
                return None
            data = await response.json()
            return format_response(data)
