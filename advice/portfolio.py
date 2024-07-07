from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from schemas import User
from helpers import get_portfolio_value, convert_holdings_to_dataframe
from .optimiser import Optimiser
from .params import OBJECTIVE_MAP

def get_transactions_from_optimal_portfolio(
    current_portfolio: pd.DataFrame|list[dict],
    optimal_portfolio: pd.DataFrame,
    amount: float = 0, # amount user is looking to raise from transactions
    max_rows: int = 5
):
    if type(current_portfolio) == "list":
        current_portfolio = pd.DataFrame.from_records(current_portfolio)
    # merge optimal and current portfolio
    # include price, beta, and priceTarget fields from optimal portfolio
    df = pd.merge(
        current_portfolio[["stockId", "units"]],
        optimal_portfolio[["stockId", "symbol", "units", "name", "previousClose", "beta", "priceTarget"]],
        how="outer",
        on="stockId",
        suffixes=("_current", "_optimal")
    ).fillna(0)

    # remove any inf values
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # calculate units column
    df["delta_units"] = df["units_optimal"] - df["units_current"]
    # calculate value column
    df["delta_value"] = df["delta_units"] * df["previousClose"]

    # drop any rows where value is zero
    df.drop(df[df["delta_value"]==0].index, inplace=True)

    # sort by absolute difference in value
    df.sort_values("delta_value", key=np.abs, ascending=False, inplace=True)

    if amount == 0:
        # indicates a portfolio review
        transactions = df.head(max_rows)
    else:
        temp = df[np.sign(df["delta_value"]) == np.sign(amount)].head(max_rows)
        temp_sum = temp["delta_value"].sum()
        if abs(temp_sum) > abs(amount) * 1.05:
            # sum is more than 5% above amount, must scale down
            # iterate through rows of temp until value sums to 'amount'
            value = 0
            transactions = pd.DataFrame(columns=df.columns.to_list())
            for index, row in temp.iterrows():
                # add row to transactions
                transactions.loc[index,:] = row
                # check if transaction must be scaled down to meet amount
                scaling_factor = min(abs(amount - value) / abs(row["delta_value"]), 1)
                if scaling_factor < 1:
                    transactions.loc[index, "delta_units"] = np.floor(row["delta_units"] * scaling_factor)
                    transactions.loc[index, "delta_value"] = row["delta_units"] * row["previousClose"]
                    break
                # increment value
                value += row["delta_value"]

        elif abs(temp_sum) < abs(amount) * 0.95:
            # sum is less than 5% below amount, must scale up
            # apply scaling factor to each transaction
            transactions = temp
            scaling_factor = abs(amount) / abs(temp_sum)
            print(scaling_factor)
            transactions["delta_units"] = np.round(transactions["delta_units"] * scaling_factor).astype(int)
            transactions["delta_value"] = transactions["delta_value"] * scaling_factor
        else:
            # sum is within 5% of amount, no scaling required
            transactions = temp

    # update optimal_portfolio
    for _, row in transactions.iterrows():
        index = optimal_portfolio[optimal_portfolio["stockId"] == row["stockId"]].index[0]
        optimal_portfolio.loc[index, "units"] = row["units_current"] + row["delta_units"]

    # drop rows from optimal_portfolio not in current_portfolio or transactions
    keep = pd.concat([current_portfolio["stockId"], transactions["stockId"]]).unique()
    optimal_portfolio = optimal_portfolio[optimal_portfolio["stockId"].isin(keep)]

    # drop any zero unit rows
    transactions.drop(transactions[transactions["delta_units"] == 0].index, inplace=True)
    # rename columns
    transactions.rename(columns={"delta_units": "units", "previousClose": "price"}, inplace=True)
    # extract required columns
    transactions = transactions[["stockId", "symbol", "name", "units", "price", "beta", "priceTarget"]]
    # convert to records before returning
    return transactions.to_dict(orient='records')

def get_recom_transactions(user: User, amount: float = 0):
    # want to select a random sample of stocks from recent advice records to exclude
    # this will give the impression of generating fresh recommendations
    exclude = []
    if (len(user.advice) > 0):
        now = datetime.now()
        symbols_to_exclude = [record["symbol"] for record in user.advice if now - record.createdAt < timedelta(days=1)]
        exclude = np.random.choice(symbols_to_exclude, np.random.randint(1, min(len(symbols_to_exclude), 5)), replace=False)
        
    current_value = get_portfolio_value(user.holdings)
    
    # initialise optimiser
    optimiser = Optimiser(user.holdings, user.profile, current_value + amount)
    # get optimal portfolio
    optimal_portfolio = optimiser.get_optimal_portfolio(exclude=exclude)
    # get transactions
    current_portfolio = convert_holdings_to_dataframe(user.holdings)
    transactions = get_transactions_from_optimal_portfolio(current_portfolio, optimal_portfolio, amount)

    # get adjusted utility before and after recommended transactions
    initial_adj_utility, final_adj_utility = optimiser.get_utility(current_portfolio), optimiser.get_utility(optimal_portfolio)
    # message to help LLM understand function output
    message = (
        "These transactions are recommended for the user based on their objective - {}. ".format(OBJECTIVE_MAP[user.profile.objective if user.profile else "RETIREMENT"]["description"]) +
        "Transactions are recommended by comparing the user's current portfolio to an optimal portfolio. The utility function is a Treynor ratio that is adjusted for the user's investing preferences."
    )

    return {
        "message": message,
        "transactions": transactions,
        "initial_adj_utility": initial_adj_utility,
        "final_adj_utility": final_adj_utility
    }