import json
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, WebSocket, Header, Response, Depends, HTTPException, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from sqlalchemy.orm import Session

from schemas import GetRecommendationsRequest, GetAdviceByStockRequest
from database import SessionLocal
from cron import schedule_jobs

from optimiser import Optimser
from params import OBJECTIVE_MAP
from helpers import get_holdings_and_profile, get_stock_by_symbol, get_sector_allocation

import traceback

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

@app.post("/get-advice-by-stock/{userId}")
def get_advice_by_stock(
    userId: str,
    body: GetAdviceByStockRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Provides information on whether a transaction of amount (amount) in stock (symbol) is recommended \
    based on current portfolio and investment profile. Negative amount for sells.

    The following are considered when making recommendation.

    1. Target sector allocations.
    2. Utility of portfolio before and after proposed transaction.
    """
    try:
        # check that symbol is valid
        # TO DO: accomodate non-NASDAQ symbols, e.g. "BHP.AX"
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
        optimiser = Optimser(current_portfolio, profile)

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
            "is_recommended": is_recommended,
            "message": message,
            "is_recommended_by_sector_allocation": is_recommended_by_sector_allocation,
            "is_recommended_by_risk": is_recommended_by_risk,
            "is_recommended_by_income": is_recommended_by_income,
            "is_analyst_recommended": is_recommended_by_analyst,
            "is_utility_positive": is_utility_positive,
            "initial_adj_utility": initial_adj_utility,
            "final_adj_utility": final_adj_utility,
            **stock, # return stock data
        }

    except Exception as e:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/get-recommendations/{userId}")
def get_recommendations(
    userId: str,
    body: GetRecommendationsRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Provides information on whether a transaction of amount (amount) in stock (symbol) is recommended \
    based on current portfolio and investment profile. Negative amount for sells.

    The following are considered when making recommendation.

    1. Target sector allocations.
    2. Utility of portfolio before and after proposed transaction.
    """
    try:
        delta_value = body.target
        current_portfolio, profile = get_holdings_and_profile(userId, db)

        # initialise optimiser
        optimiser = Optimser(current_portfolio, profile, delta_value)
        # get optimal portfolio
        optimal_portfolio = optimiser.get_optimal_portfolio()

        # merge optimal and current portfolio into one df
        df = pd.merge(
            current_portfolio[["stockId", "units"]],
            optimal_portfolio[["stockId", "symbol", "units", "name", "previousClose"]],
            how="outer",
            on="stockId",
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
            if abs(value) > abs(delta_value) or (n > 0 and len(transactions) >= n):
                break

            if np.sign(row['delta_units']) != np.sign(delta_value):
                # skip transactions if sign does not match
                continue

            # if sign of transaction matches target, add to recommended transactions
            v = row['delta_units'] * row['previousClose']
            # check if transaction will reach target
            if abs(value + v) <= abs(delta_value):
                transactions.loc[index,:] = row
                value += v

            else:
                # add partial transaction
                partial_row = row.copy()
                # round units up / down to nearest integer
                partial_row['units'] = np.copysign(np.ceil((np.abs(delta_value - value)) / row['previousClose']), v)
                transactions.loc[index,:] = partial_row
                break

        # get adjusted utility before and after recommended transactions
        initial_adj_utility, final_adj_utility = optimiser.get_utility(current_portfolio), optimiser.get_utility(optimal_portfolio)

        return {
            "transactions": transactions.to_json(),
            "initial_adj_utility": initial_adj_utility,
            "final_adj_utility": final_adj_utility
        }

    except Exception as e:
        # any other error
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
