import pandas as pd
import numpy as np

from schemas import User
from helpers import get_portfolio_value, get_sector_allocation
from .optimiser import Optimiser
from .params import OBJECTIVE_MAP

def get_stock_recommendation(
    stock: dict,
    user: User,
    amount: float # proposed amount to buy (positive) or sell (negative)
):
    """
    The following are considered when making recommendation.

        1. Overall portfolio risk (as measured by Beta)
        2. Target sector allocations.
        3. Income of the portfolio.
        4. Analyst price targets.
        5. Utility of portfolio before and after proposed transaction.
    """
    # extract objective and set default as retirement
    objective = user.profile.objective if user.profile else "RETIREMENT"
    # get proposed number of units by dividing by previousClose
    proposed_units = round(amount / stock["previousClose"])
    # check if existing holding
    existing_holding = [holding for holding in user.holdings if holding["stockId"] == stock["id"]][0] if "id" in stock else None
    # edge case where user wishes to sell stock not in portfolio
    if proposed_units < 0 and existing_holding is None:
        raise Exception("User does not hold {}".format(stock["symbol"]))

    # check whether proposed transaction is within sector allocations
    is_recommended_by_sector_allocation = True
    sector_allocation_message = "The proposed transaction is within the user's recommended sector allocation based on their objective and preferences. "
    if not stock["sector"] or objective == "TRADING":
        pass
    else:
        target_allocation = OBJECTIVE_MAP[objective]["sector_allocations"].get(stock["sector"])
        current_allocation = get_sector_allocation(user.holdings, stock["sector"])
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
    optimiser = Optimiser(user.holdings, user.profile)

    # get initial adjusted utility
    initial_adj_utility = optimiser.get_utility(user.holdings)

    # get adjusted utility after proposed transaction
    proposed_portfolio = user.holdings.copy()
    if existing_holding is None:
        # insert proposed holding
        proposed_portfolio.append({
            "stockId": stock["id"],
            "units": proposed_units
        })
    else:
        # update existing row
        index = user.holdings.index(existing_holding)
        proposed_portfolio[index]["units"] = max(existing_holding["units"] + proposed_units, 0)

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

