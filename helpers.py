
import numpy as np
import pandas as pd

from universe import Universe
from schemas import User
from data.helpers import get_aggregated_stock_data

async def get_stock_by_symbol(symbol: str):
    # attempt to fetch stock from universe
    universe = Universe()
    stock = universe.get_stock_by_symbol(symbol)
    if stock is not None:
        return stock
    
    # symbol not in universe, fetch from data provider instead
    stock = await get_aggregated_stock_data(symbol)

    if stock:
        stock = universe.add_stock(stock)

    return stock

def get_portfolio_value(portfolio: pd.DataFrame|list[dict]):
    """
    Get value of portfolio
    """
    value = 0
    if type(portfolio) == pd.DataFrame:
        portfolio = portfolio.to_dict(orient='records')

    for holding in portfolio:
        stock = Universe().get_stock_by_id(holding["stockId"])
        if stock:
            value += holding["units"] * stock["previousClose"]

    return value

def get_portfolio_as_dataframe(user: User):
    """
    Get all holdings and profile that belong to a user.
    """
    if (len(user.holdings) > 0):
        df = pd.DataFrame.from_records([m.__dict__ for m in user.holdings], index='id').drop('_sa_instance_state', axis=1)
    else:
        df = pd.DataFrame(columns=["stockId", "units"])

    return df