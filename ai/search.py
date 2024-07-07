import os
import aiohttp
import json

from datetime import datetime

async def search_web(query: str, date: str = None) -> dict:
    try:
        # Adding date to query helps to get current information
        if date is None:
            date = datetime.now().strftime('%#d %B %Y')

        query = f"{query} {date}"

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
                return data
    except Exception as e:
        return None
