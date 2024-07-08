import os
import aiohttp
import json

# from datetime import datetime

def format_response(res):
    """
    Format response from search_web function for reading by LLM
    """
    return {
        "query": res["query"],
        "answer": res["answer"],
        "results": [{
            "title": result["title"],
            "content": result["content"],
            "url": result["url"],
        } for result in res["results"]]
    }

async def search_web(query: str) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.tavily.com/search',
                headers={
                    'Content-Type': 'application/json',
                },
                data=json.dumps({
                    'api_key': os.getenv('TAVILY_API_KEY'),
                    'query': query,
                    'search_depth': 'basic',
                    'include_answer': True,
                    'max_results': 5,
                })
            ) as response:

                if response.status != 200:
                    raise Exception("Error searching web")

                data = await response.json()
                return format_response(data)
    except Exception as e:
        return None
