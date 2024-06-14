import pandas as pd

import crud

def get_universe():
    data = crud.get_all_stocks()
    return pd.DataFrame.from_records([m.__dict__ for m in data], index='id').drop('_sa_instance_state', axis=1)

def get_portfolio():
    data = crud.get_holdings()
    return pd.DataFrame.from_records([m.__dict__ for m in data], index='id').drop('_sa_instance_state', axis=1)

def merge_portfolio_with_universe(universe, portfolio):
    merged = pd.merge(universe, portfolio, left_on='id', right_on='stockId', how='left')
    merged["units"].fillna(0, inplace=True)
    return merged
