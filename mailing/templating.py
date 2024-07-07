import asyncio

from jinja2 import Environment, FileSystemLoader
from markdown import markdown

from database import SessionLocal
from helpers import get_user_data
from universe import Universe
from ai.functions import openai_call
from ai.search import search_web
from advice.functions import get_recom_transactions
from data.fmp import ApiClient

async def get_body(symbols: list[str]):
    # we will search web for general market update and an update on the user's stocks
    tasks = [
        search_web("What's happening in the stock market?")
    ]
    for symbol in symbols:
        tasks.append(
            asyncio.create_task(search_web("What's happening with {} stock?".format(symbol)))
        )

    context = ""
    for result in await asyncio.gather(*tasks):
        context += result["answer"]
        context += "\n\n"

    context = context.strip()

    messages = [
        {"role": "user", "content": f"Write market update for the user, informing them of updates in the stock market and their portfolio. The update will appear in the body of an email newsletter. Below is some information on the stock market and the user's portfolio:\n\n{context}"}
    ]

    return openai_call(messages)

async def get_contents(portfolio, profile, prev_advice):
    # Step 0: extract symbols from portfolio
    symbols = []
    for _, holding in portfolio.iterrows():
        stock = Universe().get_stock_by_id(holding['stockId'])
        if stock is not None:
            symbols.append(stock['symbol'])

    # Step 1: get portfolio performance
    # TO DO

    # Step 2: get market update
    body_task = asyncio.create_task(get_body(symbols))

    # Step 3: get news articles
    symbols = []
    article_task = asyncio.create_task(ApiClient().get_news_articles(symbols))

    # Step 4: get recommended transactions
    transactions = get_recom_transactions(portfolio, profile, prev_advice)["transactions"]
    # populate transactions
    for transaction in transactions:
        transaction["transaction"] = "Buy" if transaction["units"] > 0 else "Sell"
        transaction["value"] = transaction["price"] * transaction["units"]

    body, articles = await asyncio.gather(body_task, article_task)

    return {
        "body": body,
        "transactions": transactions,
        "articles": articles
    }

async def construct_html(portfolio, profile, prev_advice, file_path = 'output.html'):
    # Step 1: get contents
    contents = await get_contents(portfolio, profile, prev_advice)
    # Step 1.5: convert body text to markdown
    contents["body"] = markdown(contents["body"])
    # Step 2: load template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')
    # Step 3: render template
    html_output = template.render(**contents)
    # Step 4: write to output file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_output)

    return file_path