from datetime import datetime
import json

import asyncio

from upstash_redis.asyncio import Redis

from ai.functions import openai_call
from ai.search import search_web

# caching to reduce API usage
redis = Redis.from_env()

GENERAL_QUERIES = [
    "What's happening in the stock market today{region_modifier} {date_modifier}?",
    "What factors are influencing the stock market{region_modifier} {date_modifier}?",
    "What are investors thinking about this week{region_modifier} {date_modifier}?",
    "What is the stock market outlook{region_modifier} {date_modifier}?",
    "What is the latest economic news{region_modifier} {date_modifier}?"
]

async def get_general_market_update(region: str = "US"):
    # check cache
    # use date suffix to ensure response is current
    today = datetime.now()
    key = "GENERAL_MARKET_UPDATE_{}".format(today.strftime('%#d_%m_%Y'))
    cached_response = await redis.get(key)

    if cached_response:
        return cached_response
    
    # search web for market news
    tasks = []
    for query in GENERAL_QUERIES:
        tasks.append(
            asyncio.create_task(search_web(
                query=query.format(
                    date_modifier=today.strftime('%#d %B %Y'), # e.g 20 July 2024
                    region_modifier=" in Australia" if region == "AUS" else "", # specify Australia
                ),
                include_answer=True,
                include_raw_content=True
            ))
        )

    web_results = await asyncio.gather(*tasks)

    # get AI summary
    messages = [
        {"role": "system", "content": "You are an intelligent investment adviser. Use the below info to answer the user's query.\n\n{}".format('\n\n'.join([json.dumps(item) for item in web_results if item]))},            
        {"role": "user", "content": "What's happening in the stock market? I am especially interested in any economic updates. Use at least 300 words and reference any sources used."}
    ]
    response = await openai_call(messages)
    # add response to cache
    await redis.set(key, response, ex=24*60*60)
    return response

async def get_stock_update_by_symbol(symbol: str):
    # check cache
    cached_response = await redis.get(f"STOCK_UPDATE_{symbol.upper()}")

    if cached_response:
        return cached_response
    
    # search web for stock news
    web_results = await search_web(
        query="What's happening with {symbol} stock {date}?".format(
            symbol=symbol,
            date=datetime.now().strftime('%#d %B %Y'),
        ),
        include_answer=True,
        include_raw_content=True
    )

    # get AI summary
    messages = [
        {"role": "system", "content": f"Use the following info to answer the user's query:\n\n{web_results}"},
        {"role": "user", "content": f"What's happening with {symbol} stock?"}
    ]
    response = await openai_call(messages)

    # add response to cache, with 24 hour expiry
    await redis.set(f"STOCK_UPDATE_{symbol.upper()}", response, ex=24*60*60)
    # return summary
    return response