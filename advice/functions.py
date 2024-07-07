import pandas as pd
import numpy as np

from .optimiser import Optimiser
from .params import OBJECTIVE_MAP
from helpers import get_portfolio_value, get_sector_allocation

def should_buy_or_sell_stock(
    stock,
    current_portfolio,
    profile,
    amount # proposed amount to buy (positive) or sell (negative)
):
    # extract objective and set default as retirement
    objective = profile.objective if profile else "RETIREMENT"
    # get proposed number of units by dividing by previousClose
    proposed_units = round(amount / stock["previousClose"])
    # check if existing holding
    existing_holding = current_portfolio[current_portfolio["symbol"] == stock["symbol"]]
    is_existing = not existing_holding.empty
    # edge case where user wishes to sell stock not in portfolio
    if proposed_units < 0 and not is_existing:
        # TO DO
        return {
            "result": False,
            "message": "User does not hold {}".format(stock["symbol"])
        }

    # check whether proposed transaction is within sector allocations
    is_recommended_by_sector_allocation = True
    sector_allocation_message = "The proposed transaction is within the user's recommended sector allocation based on their objective and preferences. "
    if not stock["sector"] or objective == "TRADING":
        pass
    else:
        target_allocation = OBJECTIVE_MAP[objective]["sector_allocations"].get(stock["sector"])
        current_allocation = get_sector_allocation(current_portfolio, stock["sector"])
        proposed_allocation = current_allocation + amount
        if proposed_allocation > target_allocation:
            # proposed transaction is within sector allocations
            is_recommended_by_sector_allocation = False
            sector_allocation_message = (
                "The proposed transaction is outside the user's recommended sector allocation based on their objective and preferences. " +
                "User may consider doing something in another sector. "
            )

    # check whether stock is within recommendations for risk as measured by beta
    is_recommended_by_risk = True
    risk_message = "The risk (Beta) of the stock appears to be inline with the user's investment objective."
    if objective == "TRADING":
        # trading objective should have no requirements for beta
        pass
    elif not stock["beta"]:
        risk_message = "Risk (Beta) information is not available for this stock. "
    else:
        target_beta = OBJECTIVE_MAP[objective]["target_beta"]
        # check whether stock beta is within reasonable distance from target beta
        difference = stock["beta"] - target_beta
        if abs(difference) > 0.50:
            is_recommended_by_risk = proposed_units < 0 # True if user wishes to sell
            risk_message = (
                "Given the user's investment objective, the risk of (Beta) of the stock appears to be significantly {} than recommended. ".format("greater" if difference > 0 else "lower")
            )

    # check whether stock is within recommendations for income
    is_recommended_by_income = True
    income_message = "The dividend yield of the stock appears to be inline with the user's investment objective. "
    if objective == "TRADING":
        pass
    elif not stock["dividendYield"]:
        is_recommended_by_income = "Dividend information is not available for this stock. "
    else:
        target_yield = OBJECTIVE_MAP[objective]["target_yield"]
        # check whether stock yield is within reasonable distance from target
        difference = stock["dividendYield"] - target_yield
        if abs(difference) > 0.50:
            is_recommended_by_income = proposed_units < 0 # True if user wishes to sell
            risk_message = (
                "Given the user's investment objective, the dividend yield of the stock appears to be {} than recommended. ".format("greater" if difference > 0 else "lower")
            )

    # check whether proposed transaction is analyst recommended
    is_recommended_by_analyst = True
    analyst_recommendation_message = ""
    if not stock["priceTarget"]:
        is_recommended_by_analyst = False
        analyst_recommendation_message = "Analyst price target is not available for this stock, so we cannot provide a recommendation."
    else:
        is_recommended_by_analyst = stock["priceTarget"] < stock["previousClose"] if amount < 0 else stock["priceTarget"] > stock["previousClose"]
        analyst_recommendation_message = (
            "Analyst's have a price target of {} ".format(stock["priceTarget"]) +
            "which is {}% {} the previous close. ".format(
                100 * round((stock["priceTarget"] / stock["previousClose"]) - 1, 4),
                "above" if stock["priceTarget"] > stock["previousClose"] else "below"
            )
        )

    # finally, check whether portfolio utility is increased by transaction
    # initialise optimiser
    optimiser = Optimiser(current_portfolio, profile)

    # get initial adjusted utility
    initial_adj_utility = optimiser.get_utility(current_portfolio)

    # get adjusted utility after proposed transaction
    proposed_portfolio = current_portfolio.copy()
    if is_existing:
        # update existing row
        index = existing_holding.index[0]
        proposed_portfolio.loc[index, "units"] = max(existing_holding["units"].iloc[0] + proposed_units, 0)
    else:
        # insert new row
        proposed_portfolio.loc[-1] = { "stockId": stock["id"], "units": proposed_units }

    final_adj_utility = optimiser.get_utility(proposed_portfolio)

    is_utility_positive = bool(final_adj_utility > initial_adj_utility)
    utility_message = (
        "The proposed transaction {} the 'adjusted' utility of their portfolio as measured by the Treynor ratio.".format("increases" if is_utility_positive else "decreases")
    )

    # append messages together
    is_recommended = is_recommended_by_sector_allocation and is_recommended_by_risk and is_recommended_by_income and is_recommended_by_analyst and is_utility_positive
    message = (
        "The proposed transaction is {} for the user. This conclusion was made by assessing the following factors jointly.".format("recommended" if is_recommended else "not recommended") +
        "\n\nAnalyst recommendation:\n\n" + analyst_recommendation_message +
        "\n\nSector allocation:\n\n" + sector_allocation_message +
        "\n\nIncome:\n\n" + income_message +
        "\n\nRisk assessment:\n\n" + risk_message +
        "\n\nPortfolio utility:\n\n" + utility_message
    )

    return {
        "proposed_transaction": f"{'Buy' if amount > 0 else 'Sell'} ${amount:,.2f} in {stock['symbol']}",
        "user_objective": OBJECTIVE_MAP[objective]["description"],
        "is_recommended": is_recommended,
        "message": message,
        "is_recommended_by_sector_allocation": is_recommended_by_sector_allocation,
        "is_recommended_by_risk": is_recommended_by_risk,
        "is_recommended_by_income": is_recommended_by_income,
        "is_analyst_recommended": is_recommended_by_analyst,
        "is_utility_positive": is_utility_positive,
        "initial_adj_utility": str(initial_adj_utility),
        "final_adj_utility": str(final_adj_utility),
        "stockData": stock, # return stock data
    }

def get_transactions_from_optimal_portfolio(
    current_portfolio: pd.DataFrame,
    optimal_portfolio: pd.DataFrame,
    amount: float = 0, # amount user is looking to raise from transactions
    max_rows: int = 5
):
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
        # return transactions where delta_value is greater than 5% of the portfolio value
        # current_value = get_portfolio_value(current_portfolio)
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

def get_recom_transactions(current_portfolio, profile, prev_advice, amount: float = 0):
    # want to select a random sample of stocks from prev_advice records to exclude
    # this will give the appearance of generating a new series of recommendations
    prev_symbols = [transaction["symbol"] for record in prev_advice for transaction in record.transactions]
    if len(prev_symbols) > 0:
        exclude = np.random.choice(prev_symbols, np.random.randint(1, min(len(prev_symbols), 5)), replace=False)
    else:
        exclude = []
        
    current_value = get_portfolio_value(current_portfolio)
    
    # initialise optimiser
    optimiser = Optimiser(current_portfolio, profile, current_value + amount)
    # get optimal portfolio
    optimal_portfolio = optimiser.get_optimal_portfolio(exclude=exclude)

    # merge optimal and current portfolio
    # include price, beta, and priceTarget fields from optimal portfolio
    transactions = get_transactions_from_optimal_portfolio(current_portfolio, optimal_portfolio, amount)

    # get adjusted utility before and after recommended transactions
    initial_adj_utility, final_adj_utility = optimiser.get_utility(current_portfolio), optimiser.get_utility(optimal_portfolio)
    # message to help LLM understand function output
    message = (
        "These transactions are recommended for the user based on their objective - {}. ".format(OBJECTIVE_MAP[profile.objective if profile else "RETIREMENT"]["description"]) +
        "Transactions are recommended by comparing the user's current portfolio to an optimal portfolio. The utility function is a Treynor ratio that is adjusted for the user's investing preferences."
    )

    return {
        "message": message,
        "transactions": transactions,
        "initial_adj_utility": initial_adj_utility,
        "final_adj_utility": final_adj_utility
    }