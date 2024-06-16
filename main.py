import json
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, WebSocket, Header, Response, Depends, HTTPException, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

import schemas, crud, helpers
from database import SessionLocal
from optimiser import Optimser

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

@app.get("/")
def root():
    return json.dumps("Hello World")

@app.post("/get-advice-by-stock/{userId}")
def get_advice_by_stock(
    userId: str,
    body: schemas.GetAdviceByStockRequest,
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
        stock = crud.get_stock_by_symbol(body.symbol, db)
        if not stock:
            return {
                "message": f"{body.symbol} was not found",
                "statusCode": 400
            }

        # fetch data
        current_portfolio, profile = helpers.get_holdings_and_profile(userId, db)

        # initialise optimiser
        optimiser = Optimser(current_portfolio)

        # get initial adjusted utility
        initial_adj_utility = optimiser.get_utility(current_portfolio)
        
        # get adjusted utility after proposed transaction
        units = round(body.amount / stock["previousClose"])
        final_portfolio = current_portfolio.copy()
        final_portfolio.set_index('stockId', inplace=True)
        if stock.id in current_portfolio.index:
            # update existing row
            final_portfolio.loc[stock.id, 'units'] = max(current_portfolio.loc[stock.id]['units'] + units, 0)
        else:
            # insert new row
            final_portfolio.loc[stock.id] = { "units": units }

        final_adj_utility = optimiser.get_utility(final_portfolio)
           
        return json.dumps({
            "initial_adj_utility": initial_adj_utility,
            "final_adj_utility": final_adj_utility
        })
    
    except Exception as e:
        # any other error
        print(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/get-recommendations")
def get_recommendations(
    body: schemas.GetRecommendationsRequest,
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
        current_portfolio = helpers.get_portfolio(db)
        
        # initialise optimiser
        optimiser = Optimser(current_portfolio, delta_value)
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

        return json.dumps({
            "transactions": transactions.to_json(),
            "initial_adj_utility": initial_adj_utility,
            "final_adj_utility": final_adj_utility
        })

    except Exception as e:
        # any other error
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
