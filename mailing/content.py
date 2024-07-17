import asyncio
import pandas as pd

from markdown import markdown

from schemas import User, Holding
from universe import Universe
from helpers import get_portfolio_from_user, get_profile_from_user

from ai.functions import openai_call
from ai.helpers import get_general_market_update, get_stock_update_by_symbol
from advice.portfolio import get_recom_transactions
from data.financial_modelling_prep import ApiClient

client = ApiClient()

freq_map = {
    "DAILY": "1D",
    "WEEKLY": "5D",
    "MONTHLY": "1M"
}

async def get_main_text(
        portfolio: pd.DataFrame,
        change_map: dict,
        transactions: list[dict],
        freq: str = "DAILY",
        region: str = "US"
    ):
    # initialise tasks list
    tasks = []
    # append general market update
    tasks.append(
        asyncio.create_task(get_general_market_update(region))
    )
    
    # get a subset of symbols for AI to talk about
    # we want a mix of stocks from the user's portfolio and the recommended transactions
    # limit to 5
    max_symbols = 5
    sorted_changes = sorted([(key, abs(value['percent_change'])) for key, value in change_map.items() if key != "total"], key=lambda item: item[1])
    sorted_transactions = sorted([(transaction['symbol'], abs(transaction['units'] * transaction['price'])) for transaction in transactions], key=lambda item: item[1])
    symbols_of_interest = list(set(
        [item[0] for item in sorted_changes][:max(max_symbols - len(sorted_transactions), 1)] +
        [item[0] for item in sorted_transactions]
    ))

    for symbol in symbols_of_interest:
        tasks.append(
            asyncio.create_task(get_stock_update_by_symbol(symbol))
        )

    sections = await asyncio.gather(*tasks)

    system = "You are an insightful, intelligent, and enthusastic investment adviser."
    
    content = (
        "Use the below information to write a comprehensive article about the stock market. " +
        "The article should be split into sections: General Market Update and Stocks You Might Be Interested In. " +
        "The first section should include an update on the overall stock market, what is driving the market, and important economic news. " +
        "The second section should contain a brief update on each stock, including the potential impact of the general market on the stock. " +
        "You can omit a stock if there is not enough information to provide a meaningful update."
        "\n'''\n" + # helps to separate info from above instruction
        "\n\n".join(sections)
    )

    messages = [
        {"role":"system", "content": system},
        {"role":"user", "content": content}
    ]

    return await openai_call(messages)

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

def format_transactions(transactions: list[dict]):
    # create copy of transactions
    formatted_transactions = transactions.copy()
    # format transactions
    for transaction in formatted_transactions:
        transaction["transaction"] = "📈 Buy" if transaction["units"] > 0 else "📉 Sell"
        transaction["value"] = "$  {:,.2f}".format(transaction["price"] * transaction["units"])
        transaction["price"] = "$  {:,.2f}".format(transaction["price"])

    return formatted_transactions

async def get_content(user: User):
    # Step 0: populate user's portfolio with stock info
    portfolio = Universe().merge_with_portfolio(get_portfolio_from_user(user))
    portfolio = portfolio[portfolio["units"] > 0]

    # Step 1: get portfolio changes
    changes = await calculate_portfolio_changes(portfolio, user.mailFrequency)

    # Step 2: get recommended transactions
    # want to get
    advice = get_recom_transactions(user)

    # Step 3: get market update
    # use profile to determine region as Australia or US
    profile = get_profile_from_user(user)
    region = "US" if profile.international > 0.5 else "AUS"
    body = await get_main_text(portfolio, changes, advice["transactions"], user.mailFrequency, region)

    return {
        "body": markdown(body), # convert markdown to html
        "formatted_transactions": format_transactions(advice["transactions"]),
        "advice": advice,
    }