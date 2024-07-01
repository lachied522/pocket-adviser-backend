from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sqlalchemy.orm import Session

import crud
from universe import Universe

def get_portfolio_value(portfolio: pd.DataFrame):
    """
    Get value of portfolio
    """
    value = 0
    for _, row in portfolio.iterrows():
        stock = Universe().get_stock_by_id(row["stockId"])
        if stock:
            value += row["units"] * stock["previousClose"]
    return value

def get_sector_allocation(portfolio: pd.DataFrame, sector: str):
    """
    Get current sector allocation for a portfolio.
    """
    value = 0
    for _, row in portfolio.iterrows():
        stock = Universe().get_stock_by_id(row["stockId"])
        if stock and stock["sector"] == sector:
            value += row["units"] * stock["previousClose"]
    return value

def get_riskfree_rate():
    """
    Risk free rate for use in optimisation models. Taken as 10-year US treasury yield.
    """
    return 0.05 # TO DO

def get_user_data(userId: str|None, db: Session):
    """
    Get all holdings and profile that belong to a user.
    """
    if userId is None:
        # return empty portfolio and profile
        return (
            pd.DataFrame(columns=["stockId", "units"]),
            None
        )

    # could fetch through a join table by sqlalchemy throws an error
    user = crud.get_user_record(userId, db)
    if user is None:
        raise Exception("User not found")

    if (len(user.holdings) > 0):
        portfolio = pd.DataFrame.from_records([m.__dict__ for m in user.holdings], index='id').drop('_sa_instance_state', axis=1)
    else:
        portfolio = pd.DataFrame(columns=["stockId", "units"])

    profile = None
    if user.profile:
        profile = user.profile[0]

    advice = []
    if (len(user.advice) > 0):
        now = datetime.now()
        advice = [record for record in user.advice if now - record.createdAt < timedelta(days=1)]

    return (
        portfolio,
        profile,
        advice,
    )

def merge_portfolio_with_universe(portfolio: pd.DataFrame):
    universe = Universe().get()

    if portfolio.empty:
        merged = universe.copy()
        merged["stockId"] = universe.index
        merged["units"] = np.zeros(len(merged))
    else:
        merged = pd.merge(universe, portfolio[["stockId", "units"]], left_index=True, right_on='stockId', how='left').reset_index(drop=True)
        merged["units"] = merged["units"].fillna(0)

    return merged
