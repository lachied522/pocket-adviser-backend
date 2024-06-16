import pandas as pd

from sqlalchemy.orm import Session

from database import SessionLocal
import crud

universe: pd.DataFrame | None = None

def get_universe():
    if universe: return universe

    db: Session = SessionLocal()
    data = crud.get_all_stocks(db)
    df = pd.DataFrame.from_records([m.__dict__ for m in data], index='id').drop('_sa_instance_state', axis=1)
    # add 'expected return column'
    df['exp_return'] = df.apply(lambda x: x['priceTarget'] / x['previousClose'] - 1, axis=1)
    return df

def get_holdings_and_profile(userId: str, db: Session = SessionLocal()):
    holdings = crud.get_holdings_by_user_id(userId, db)
    profile = crud.get_profile_by_user_id(userId, db)
    return (
        pd.DataFrame.from_records([m.__dict__ for m in holdings], index='id').drop('_sa_instance_state', axis=1),
        profile,
    )

def merge_portfolio_with_universe(universe, portfolio):
    merged = pd.merge(universe, portfolio[["stockId", "units"]], left_on='id', right_on='stockId', how='left')
    merged["units"].fillna(0, inplace=True)
    return merged
