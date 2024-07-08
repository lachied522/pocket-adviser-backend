from datetime import datetime
import json

import asyncio
import pandas as pd

from jinja2 import Environment, FileSystemLoader
from markdown import markdown

from schemas import User, Holding
from universe import Universe
from helpers import get_portfolio_as_dataframe

from ai.functions import openai_call
from ai.search import search_web
from advice.portfolio import get_recom_transactions
from data.financial_modelling_prep import ApiClient

client = ApiClient()

freq_map = {
    "DAILY": "1D",
    "WEEKLY": "5D",
    "MONTHLY": "1M"
}

query_map = {
    "DAILY": [
        "What's happening in the stock market{region}{date}?",
        "What factors are influencing the stock market{region}{date}?"
    ],
    "WEEKLY": [
        "What's happening in the stock market this week{region}{date}?",
        "What are investors thinking about this week{region}{date}?",
    ],
    "MONTHLY": [
        "What happened in the stock market last month{region}{date}?",
        "What is the stock market outlook{region}{date}?",
    ]
}

SYSTEM_MESSAGE = (
    "You are an enthusiastic investment advisor. You are assiting the user with their investments in the stock market. " +
    "Feel free to use emojis in your messages. "
)

def get_general_search_queries(freq: str = "DAILY", region: str = "US"):
    if freq == "MONTHLY":
        date = datetime.now().strftime('%B %Y')
    else:
        date = datetime.now().strftime('%#d %B %Y')

    queries = []
    for query in query_map[freq]:
        queries.append(query.format(**{
            "region": " in Australia" if region == "AUS" else "",
            "date": " " + date
        }))

    return queries

async def get_main_text(
        portfolio: pd.DataFrame,
        change_map: dict,
        transactions: list[dict],
        freq: str = "DAILY",
        region: str = "US"
    ):
    # search web for general market news
    tasks = []
    for query in get_general_search_queries(freq, region):
        tasks.append(
            asyncio.create_task(search_web(query))
        )
    
    # get a subset of symbols for AI to talk about
    # we want a mix of stocks from the user's portfolio and the recommended transactions
    # limit to 10
    sorted_changes = sorted([(key, abs(value['percent_change'])) for key, value in change_map.items() if key != "total"], key=lambda item: item[1])
    sorted_transactions = sorted([(transaction['symbol'], abs(transaction['value'])) for transaction in transactions], key=lambda item: item[1])
    symbols_of_interest = list(set(
        [item[0] for item in sorted_changes][:10 - len(sorted_transactions)] +
        [item[0] for item in sorted_transactions]
    ))

    for symbol in symbols_of_interest:
        tasks.append(
            asyncio.create_task(search_web("What's happening with {} stock?".format(symbol)))
        )

    # initialise message
    content = (
        "Give the user a detailed update on the general stock market. Include a paragraph about what factors are influencing the stock market. " + 
        "Below is some information from the web about the general market and some stocks the user might be interested in. " +
        "Reference these sources in your response.\n\n" +
        "Do not provide any recommendations.\n\n"
    )

    # add portfolio to context    
    content += "Market news:\n\n"
    # add results of web search to context
    for result in await asyncio.gather(*tasks):
        content += json.dumps(result)
        content += "\n\n"

    # add some basic stock info
    content += "Stock info:\n\n{}\n\n"
    for symbol in symbols_of_interest:
        stock = Universe().get_stock_by_symbol(symbol)
        content += json.dumps({
            "symbol": stock["symbol"],
            "name": stock["name"],
            "price": stock["previousClose"],
            f"{freq.lower()}_change": change_map[symbol] if symbol in change_map else "N/A"
        })

    content += "User's portfolio:\n\n{}".format(portfolio[["symbol", "name", "units"]].to_dict(orient="records"))

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content}
    ]

    return openai_call(messages)

async def calculate_portfolio_changes(
    portfolio: pd.DataFrame,
    freq: str = "DAILY" # daily, weekly, monthly
):
    if not freq.upper() in ["DAILY", "WEEKLY", "MONTHLY"]:
        raise Exception("Period must be one of DAILY, WEEKLY, MONTHLY")

    # get period as '1D', '5D', or '1M'
    period = freq_map[freq]
    # fetch stock performance data
    performances = await client.get_performance(portfolio["symbol"].values.tolist())
    performance_map = {d['symbol']: d[period] for d in performances}

    change_map = {
        "total": {
            "dollar_change": 0,
            "percent_change": 0
        }
    }

    total_current_value = 0 # used for calculating total percent change
    for _, stock in portfolio.iterrows():
        if stock["symbol"] not in performance_map:
            continue
        # increment total
        total_current_value +=  stock["units"] * stock["previousClose"]
        # update changes map
        change_map[stock["symbol"]] = {
            "dollar_change": round(stock["units"] * performance_map[stock["symbol"]], 2),
            "percent_change": round(100 * performance_map[stock["symbol"]] / (stock["previousClose"] - performance_map[stock["symbol"]]), 2)
        }

    # calculate percent change
    change_map["total"]["percent_change"] = 100 * change_map["total"]["dollar_change"] / (total_current_value - change_map["total"]["dollar_change"])

    return change_map

async def get_content(user: User):
    # Step 0: populate user's portfolio with stock info
    portfolio = Universe().merge_with_portfolio(get_portfolio_as_dataframe(user))
    portfolio = portfolio[portfolio["units"] > 0]

    # Step 1: get portfolio changes
    changes = await calculate_portfolio_changes(portfolio, user.mailFrequency)

    # Step 2: get recommended transactions
    # want to get
    transactions = get_recom_transactions(user)["transactions"]
    # populate transactions
    for transaction in transactions:
        transaction["transaction"] = "Buy" if transaction["units"] > 0 else "Sell"
        transaction["value"] = transaction["price"] * transaction["units"]

    # Step 3: get market update
    # use user profile to determine region as Australia or US
    region = "US" if user.profile[0].international > 0.5 else "AUS"
    body = await get_main_text(portfolio, changes, transactions, user.mailFrequency, region)

    # Step 4: get news articles
    # article_task = asyncio.create_task(client.get_news_articles(symbols_of_interest))

    # body, articles = await asyncio.gather(body_task,  article_task)

    return {
        "body": body,
        "transactions": transactions,
        # "articles": articles
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
    html_output = template.render(name=user.name, freq=user.mailFrequency.lower(), **content)
    # Step 4: write to output file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_output)

    return file_path