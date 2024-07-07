import asyncio

from jinja2 import Environment, FileSystemLoader
from markdown import markdown

from schemas import User
from universe import Universe

from ai.functions import openai_call
from ai.search import search_web
from advice.portfolio import get_recom_transactions
from data.fmp import ApiClient

client = ApiClient()

freq_map = {
    "DAILY": "1D",
    "WEEKLY": "5D",
    "MONTHLY": "1M"
}

async def get_main_text(symbols: list[str], region: str):
    # we will search web for general market update and an update on the user's stocks
    tasks = [
        search_web("What's happening in the stock market{}?".format(" in Australia" if region == "AUS" else ""))
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

async def calculate_portfolio_changes(
    holdings: list,
    symbols: list[str],
    freq: str = "daily" # daily, weekly, monthly
):
    if not freq in ["daily", "weekly", "monthly"]:
        raise Exception("Period must be one of daily, weekly, monthly")

    # get period as '1D', '5D', or '1M'
    period = freq_map[freq]

    performances = await client.get_performance(symbols)
    performance_map = {d['symbol']: d for d in performances}

    change_map = {
        "total": {
            "dollar_change": 0,
            "percent_change": 0
        }
    }

    total_current_value = 0 # used for calculating total percent change
    for holding in holdings:
        stock = Universe().get_stock_by_id(holding['stockId'])
        if stock is None:
            continue
        symbol = stock["symbol"]
        if symbol not in performance_map:
            continue
        current_value = holding["units"] * stock["price"]
        dollar_change = holding["units"] * performance_map[symbol][period]
        # increment total
        total_current_value += current_value
        # update changes map
        change_map[symbol] = {
            "dollar_change": dollar_change,
            "percent_change": 100 * dollar_change / (current_value - dollar_change)
        }

    # calculate percent change
    change_map["total"]["percent_change"] = 100 * change_map["total"]["dollar_change"] / (total_current_value - change_map["total"]["dollar_change"])

    return change_map

async def get_content(user: User):
    # Step 0: extract symbols from portfolio
    symbols = []
    for holding in user.holdings:
        stock = Universe().get_stock_by_id(holding['stockId'])
        if stock is not None:
            symbols.append(stock['symbol'])

    # Step 1: get portfolio changes
    changes = await calculate_portfolio_changes(user.holdings, symbols)

    # Step 2: get recommended transactions
    # want to get
    transactions = get_recom_transactions(user)["transactions"]
    # populate transactions
    for transaction in transactions:
        transaction["transaction"] = "Buy" if transaction["units"] > 0 else "Sell"
        transaction["value"] = transaction["price"] * transaction["units"]

    # Step 3: get market update
    # get a subset of symbols for AI to talk about
    # we want a mix of stocks from the user's portfolio and the recommended transactions
    sorted_changes = sorted([(key, abs(value['percent_change'])) for key, value in changes.items() if key != "total"], key=lambda item: item[1])
    sorted_transactions = sorted([(transaction['symbol'], abs(transaction['value'])) for transaction in transactions], key=lambda item: item[1])
    symbols = list(set(
        [item[0] for item in sorted_changes][:10 - len(sorted_transactions)] + # limit this to 10
        [item[0] for item in sorted_transactions]
    ))
    # use user profile to determine region as Australia or US
    region = "US" if user.profile.get("international", 1) > 0.5 else "AUS"
    body_task = asyncio.create_task(get_main_text(symbols, region))

    # Step 4: get news articles
    symbols = []
    article_task = asyncio.create_task(client.get_news_articles(symbols))

    body, articles = await asyncio.gather(body_task, article_task)

    return {
        "body": body,
        "transactions": transactions,
        "articles": articles
    }

async def construct_html_body_for_email(
    user: User,
    file_path: str = 'output.html',
):
    # Step 1: get contents
    content = await get_content(user)
    # Step 1.5: convert body text to markdown
    content["body"] = markdown(content["body"])
    # Step 2: load template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')
    # Step 3: render template
    html_output = template.render(**content)
    # Step 4: write to output file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_output)

    return file_path