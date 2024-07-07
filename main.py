import os
import traceback

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from advice.functions import get_recom_transactions, should_buy_or_sell_stock
from cron import schedule_jobs
from crud import insert_advice_record
from database import SessionLocal
from helpers import get_portfolio_value, get_stock_by_symbol, get_user_data
from schemas import GetAdviceByStockRequest, GetRecommendationsRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App running")

    if os.getenv("ENVIRONMENT") == "production":
        scheduler = AsyncIOScheduler()
        # add jobs to scheduler
        schedule_jobs(scheduler)
        # start scheduler
        scheduler.start()
        print("Scheduler started")

    yield
    print("App Shutdown")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.pocketadviser.com.au", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/get-advice-by-stock")
async def get_advice_by_stock(
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
        stock = await get_stock_by_symbol(body.symbol)
        if not stock:
            return {
                "message": f"Symbol {body.symbol} was not found",
                "statusCode": 400
            }

        # fetch data
        portfolio, profile, _ = get_user_data(userId, db)
        return should_buy_or_sell_stock(stock, portfolio, profile, body.amount)

    except Exception as e:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/get-advice")
def get_advice(
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
            amount = abs(body.amount)
        elif body.action == "withdraw":
            amount = -abs(body.amount)
        else:
            amount = 0

        current_portfolio, profile, prev_advice = get_user_data(userId, db)
        # get current portfolio value
        current_value = get_portfolio_value(current_portfolio)
        # handle edge case where implied portfolio value is zero or less
        if current_value + amount <= 0:
            if body.action == "review":
                # user is asking for a portfolio review where implied value is zero
                # we will use an implied value of $1,000
                amount = 1000
            else:
                # TO DO
                response.status_code = 400
                return {
                    "message": "The implied value of the user's portfolio is less than or equal to zero.",
                    "transactions": [],
                }

        res = get_recom_transactions(current_portfolio, profile, prev_advice)

        # insert advice record
        if userId is not None:
            insert_advice_record({
                "userId": userId,
                "action": body.action.upper(),
                "amount": body.amount,
                "transactions": res["transactions"],
            }, db)

            db.commit()

        return res

    except Exception as e:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
