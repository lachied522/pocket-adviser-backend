import numpy as np
import pandas as pd

from optimiser import Optimser
from database import SessionLocal
import crud

from pydantic import BaseModel

# {
#     body: {
#         target: int
#         holdings: []
#     }
# }

class EventBody(BaseModel):
    holdings: list
    target: int

def handler(event, context):
    """
    Generate an array of recommended transactions by comparing current portfolio to an optimal portfolio.
    """
    print("event", event)
    print("context", context)

    try:
        target = event["body"]["target"] # target funds to raise, defaults to zero

        db = SessionLocal()
        current_portfolio, universe = pd.DataFrame.from_records(crud.get_holdings(db)), pd.DataFrame.from_records(crud.get_stocks(db))

        # initialise optimiser
        optimiser = Optimser(current_portfolio, universe)
        # get optimal portfolio
        optimal_portfolio = optimiser.get_optimal_portfolio()

        # merge optimal and current portfolio into one df
        df = pd.merge(
            current_portfolio[["symbol", "units"]],
            optimal_portfolio[["symbol", "units", "name", "previousClose"]],
            how="outer",
            on="symbol",
            suffixes=("_current", "_optimal")
        ).fillna(0)

        # remove any inf values
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # calculate units column
        df["delta_units"] = df['units_optimal'] - df['units_current']

        # drop any rows where units is zero
        df = df.drop(df[df['delta_units']==0].index)

        # sort by absolute difference
        df = df.sort_values('delta_units', key=np.abs, ascending=False)

        # loop through transactions until either delta is met or n > N
        n = 3 # TO DO
        value = 0
        transactions = pd.DataFrame(columns=df.columns.to_list()) # create new df for transactions
        for index, row in df.iterrows():
            if abs(value) > abs(target) or (n > 0 and len(transactions) >= n):
                break

            if np.sign(row['delta_units']) != np.sign(target):
                # skip transactions if sign does not match
                continue
            
            # if sign of transaction matches target, add to recommended transactions
            v = row['delta_units'] * row['previousClose']
            # check if transaction will reach target
            if abs(value + v) <= abs(target):
                transactions.loc[index,:] = row
                value += v

            else:
                # add partial transaction
                partial_row = row.copy()
                # round units up / down to nearest integer
                partial_row['units'] = np.copysign(np.ceil((np.abs(target - value)) / row['previousClose']), v)
                transactions.loc[index,:] = partial_row
                break

        # get adjusted utility before and after recommended transactions
        initial_adj_utility, final_adj_utility = optimiser.get_utility(current_portfolio), optimiser.get_utility(optimal_portfolio)

        return {
            "body": {
                "transactions": transactions.to_json(),
                "initial_adj_utility": initial_adj_utility,
                "final_adj_utility": final_adj_utility
            },
            "statusCode": 200
        }
    
    except KeyError as key:
        # indicates event body is missing a parameter
        return {
            "message": f"{key} is required",
            "statusCode": 400
        }
    
    except Exception as e:
        # any other error
        print(e)
        return { "statusCode": 500 }