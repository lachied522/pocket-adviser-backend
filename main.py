import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from sqlalchemy.orm import Session

from schemas import GetRecommendationsRequest, GetAdviceByStockRequest
from database import SessionLocal
from cron import schedule_jobs

from optimiser import Optimiser
from params import OBJECTIVE_MAP
from helpers import get_holdings_and_profile, get_stock_by_symbol, get_portfolio_value, get_sector_allocation

import traceback

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TO DO
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        print('db closing')
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App running")

    if os.getenv("environment") == "PRODUCTION":
        scheduler = AsyncIOScheduler()
        # add jobs to scheduler
        schedule_jobs(scheduler)
        # start scheduler
        scheduler.start()
        print("Scheduler started")

    yield
    print("App Shutdown")

@app.get("/")
def root():
    return json.dumps("Hello World")

@app.post("/get-advice-by-stock")
def get_advice_by_stock(
    body: GetAdviceByStockRequest,
    response: Response,
    userId: str|None = None,
    db: Session = Depends(get_db)
):
    """
    Provides information on whether a transaction of amount (amount) in stock (symbol) is recommended \
    based on current portfolio and investment profile. Negative amount for sells.

    The following are considered when making recommendation.

    1. Overall portfolio risk (as measured by Beta)
    2. Target sector allocations.
    3. Income of the portfolio.
    4. Analyst price targets.
    5. Utility of portfolio before and after proposed transaction.
    """
    try:
        # check that symbol is valid
        stock = get_stock_by_symbol(body.symbol)
        if not stock:
            return {
                "message": f"Symbol {body.symbol} was not found",
                "statusCode": 400
            }

        # fetch data
        current_portfolio, profile = get_holdings_and_profile(userId, db)
        # extract objective and set default as retirement
        objective = profile.objective if profile else "RETIREMENT"
        # get proposed number of units by dividing by previousClose
        proposed_units = round(body.amount / stock["previousClose"])
        # check if existing holding
        existing_holding = current_portfolio[current_portfolio["stockId"] == stock["id"]]
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
            proposed_allocation = current_allocation + body.amount
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
                    "Given the user's investment objective, the risk of (Beta) of the stock is significantly {f} than recommended. ".format("greater" if difference > 0 else "lower")
                )

        # check whether proposed transaction is analyst recommended
        is_recommended_by_analyst = True
        analyst_recommendation_message = ""
        if not stock["priceTarget"]:
            is_recommended_by_analyst = False
            analyst_recommendation_message = "Analyst price target is not available for this stock, so we cannot provide a recommendation."
        else:
            is_recommended_by_analyst = stock["priceTarget"] < stock["previousClose"] if body.amount < 0 else stock["priceTarget"] > stock["previousClose"]
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
        initial_adj_utility = str(optimiser.get_utility(current_portfolio))

        # get adjusted utility after proposed transaction
        proposed_portfolio = current_portfolio.copy()
        if is_existing:
            # update existing row
            index = existing_holding.index[0]
            proposed_portfolio.loc[index, "units"] = max(existing_holding["units"].iloc[0] + proposed_units, 0)
        else:
            # insert new row
            proposed_portfolio.loc[-1] = { "stockId": stock["id"], "units": proposed_units }

        final_adj_utility = str(optimiser.get_utility(proposed_portfolio))

        is_utility_positive = final_adj_utility > initial_adj_utility
        utility_message = (
            "The proposed transaction {} increase the 'adjusted' utility of their portfolio as measured by the Treynor ratio.".format("does" if is_utility_positive else "does not")
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
            "proposed_transaction": f"{'Buy' if body.amount > 0 else 'Sell'} ${body.amount} in {body.symbol}",
            "user_objective": OBJECTIVE_MAP[objective]["description"],
            "is_recommended": is_recommended,
            "message": message,
            "is_recommended_by_sector_allocation": is_recommended_by_sector_allocation,
            "is_recommended_by_risk": is_recommended_by_risk,
            "is_recommended_by_income": is_recommended_by_income,
            "is_analyst_recommended": is_recommended_by_analyst,
            "is_utility_positive": is_utility_positive,
            "initial_adj_utility": initial_adj_utility,
            "final_adj_utility": final_adj_utility,
            "stockData": stock, # return stock data
        }

    except Exception as e:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/get-recommendations")
def get_recommendations(
    body: GetRecommendationsRequest,
    response: Response,
    userId: str|None = None,
    db: Session = Depends(get_db),
):
    """
    Provides a list of recommended transactions based on user's objective and preferences.
    """
    try:
        if body.action == "deposit":
            delta_value = abs(body.target)
        elif body.action == "withdraw":
            delta_value = -abs(body.target)
        else:
            delta_value = 0

        current_portfolio, profile = get_holdings_and_profile(userId, db)
        # get current portfolio value
        current_value = get_portfolio_value(current_portfolio)
        # handle edge case where implied portfolio value is zero or less
        if current_value + delta_value <= 0:
            if body.action == "review":
                # user is asking for a portfolio review where implied value is zero
                # we will use an implied value of $1,000
                delta_value = 1000
            else:
                # TO DO
                return {
                    "message": "The implied value of the user's portfolio is less than or equal to zero.",
                    "transactions": [],
                }

        # initialise optimiser
        optimiser = Optimiser(current_portfolio, profile, current_value+delta_value)
        # get optimal portfolio
        optimal_portfolio = optimiser.get_optimal_portfolio()

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
        df = df.drop(df[df["delta_value"]==0].index)

        # sort by absolute difference in value
        df = df.sort_values("delta_value", key=np.abs, ascending=False)

        n = 5 # TO DO
        if n > 0:
            # get first n transactions
            df = df[np.sign(df["delta_units"]) == np.sign(delta_value)].head(n)
            # scale n transactions up (or down) such that the total is equal to delta_value
            scaling_factor = abs(delta_value) / abs(df["delta_value"].sum())
            # recalculate units column and round to nearest int
            df["delta_units"] = (scaling_factor * df["delta_units"]).astype(int)
            # update units of optimal portfolio
            for _, row in df.iterrows():
                index = optimal_portfolio[optimal_portfolio["stockId"] == row["stockId"]].index[0]
                optimal_portfolio.loc[index, "units"] = row["units_current"] + row["delta_units"]
            # drop rows from optimal portfolio not in current_portfolio or df
            keep = pd.concat([current_portfolio["stockId"], df["stockId"]]).unique()
            optimal_portfolio = optimal_portfolio[optimal_portfolio["stockId"].isin(keep)]

        # rename columns
        df = df.rename(columns={"delta_units": "units", "previousClose": "price"})
        # extract required columns
        transactions = df[["stockId", "symbol", "name", "units", "price", "beta", "priceTarget"]]

        # get adjusted utility before and after recommended transactions
        # TO DO: calculate actual final utility
        initial_adj_utility, final_adj_utility = optimiser.get_utility(current_portfolio), optimiser.get_utility(optimal_portfolio)
        message = (
            "These transactions are recommended for the user based on their objective - {}. ".format(OBJECTIVE_MAP[profile.objective if profile else "RETIREMENT"]["description"]) +
            "Transactions are recommended by comparing the user's current portfolio to an optimal portfolio. The utility function is a Treynor ratio that is adjusted for the user's investing preferences."
        )

        return {
            "message": message,
            "transactions": [row.to_dict() for _, row in transactions.iterrows()],
            "initial_adj_utility": initial_adj_utility,
            "final_adj_utility": final_adj_utility
        }

    except Exception as e:
        # any other error
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
