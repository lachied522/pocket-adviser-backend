import numpy as np
import pandas as pd

from sqlalchemy.orm import Session

from database import SessionLocal
import crud

universe: pd.DataFrame | None = None
r_f: float | None = None # risk-free rate
MRP: float | None = None # market risk premium

def get_universe():
    if universe: return universe

    db: Session = SessionLocal()
    data = crud.get_all_stocks(db)
    df = pd.DataFrame.from_records([m.__dict__ for m in data], index='id').drop('_sa_instance_state', axis=1)
    # add 'expected return column'
    df["expReturn"] = df.apply(lambda x: x["priceTarget"] / x["previousClose"] - 1, axis=1)
    return df

def get_stock_by_id(_id: int):
    df = get_universe()

    if _id not in df.index.values:
        return None

    _stock = df.iloc[_id]
    return {
        "id": _stock.index[0],
        **_stock.to_dict()
    }

def get_stock_by_symbol(symbol: str):
    df = get_universe()
    # ensure symbol is uppercase
    symbol = symbol.upper()
    if symbol not in df["symbol"].values:
        return None
    
    _id = df[df["symbol"] == symbol].index[0]
    _stock = df.iloc[_id]
    return {
        "id": _id,
        **_stock.to_dict()
    }

def get_sector_allocation(portfolio: pd.DataFrame, sector: str):
    """
    Get current sector allocation for a portfolio.
    """
    value = 0
    for _, row in portfolio.iterrows():
        stock = get_stock_by_id(row["stockId"])
        if stock and stock["sector"] == sector:
            value += row["units"] * stock["previousClose"]
    return value

def get_riskfree_rate():
    """
    Risk free rate for use in optimisation models. Taken as 10-year US treasury yield.
    """
    return 0.05 # TO DO

def get_holdings_and_profile(userId: str, db: Session = SessionLocal()):
    """
    Get all holdings and profile that belong to a user.
    """
    # could fetch through a join table by sqlalchemy throws an error
    holdings = crud.get_holdings_by_user_id(userId, db)
    profile = crud.get_profile_by_user_id(userId, db)
    return (
        pd.DataFrame.from_records([m.__dict__ for m in holdings], index='id').drop('_sa_instance_state', axis=1),
        profile,
    )

def merge_portfolio_with_universe(universe, portfolio):
    merged = pd.merge(universe, portfolio[["stockId", "units"]], left_on='id', right_on='stockId', how='left')
    merged["previousClose"] = merged["previousClose"].fillna(0)
    merged["units"] = merged["units"].fillna(0)
    return merged
