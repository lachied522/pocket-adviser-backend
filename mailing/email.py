import asyncio

from database import SessionLocal
from helpers import get_user_data
from universe import Universe
from ai.functions import openai_call
from ai.search import search_web
from advice.functions import get_recom_transactions
from data.fmp import ApiClient

async def construct_body(symbols: list[str]):
    # we will search web for general market update and an update on the user's stocks
    tasks = [
        search_web("What's happening in the stock market?")
    ]
    for symbol in symbols:
        tasks.append(search_web("What's happening with {} stock?".format(symbol)))

    context = ""
    for result in await asyncio.gather(*tasks):
        context += result["answer"]
        context += "\n\n"

    context = context.strip()

    messages = [
        {"role": "user", "content": f"Write market update for the user, informing them of updates in the stock market and their portfolio. The update will appear in the body of an email newsletter. Below is some information on the stock market and the user's portfolio:\n\n{context}"}
    ]

    return openai_call(messages)

async def construct_email(userId: str, db = SessionLocal):
    portfolio, profile, prev_advice = get_user_data(userId, db)

    # Step 0: extract symbols from portfolio
    symbols = []
    for _, holding in portfolio.iterrows():
        stock = Universe().get_stock_by_id(holding['stockId'])
        symbols.append(stock['symbol'])

    # Step 1: get portfolio performance
    # TO DO

    # Step 2: get market update
    body = await construct_body(symbols)

    # Step 3: get recommendations
    transactions = get_recom_transactions(portfolio, profile, prev_advice)["transactions"]

    # Step 4: get news articles
    symbols = []
    articles = await ApiClient().get_news_articles(symbols)

    return {
        "body": body,
        "transactions": transactions,
        "articles": articles
    }