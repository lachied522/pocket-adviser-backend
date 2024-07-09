import os
import traceback

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from crud import get_user_record, insert_advice_record
from database import SessionLocal

from helpers import get_portfolio_from_user, get_profile_from_user, get_portfolio_value, get_stock_by_symbol
from data.cron import schedule_jobs as schedule_data_jobs
from mailing.cron import schedule_jobs as schedule_mail_jobs
from advice.single_stock import get_stock_recommendation
from advice.portfolio import get_recom_transactions
from advice.params import OBJECTIVE_MAP

from schemas import GetAdviceByStockRequest, GetRecommendationsRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App running")

    if os.getenv("ENVIRONMENT") == "production":
        scheduler = AsyncIOScheduler()
        # add jobs to scheduler
        schedule_data_jobs(scheduler)
        schedule_mail_jobs(scheduler)
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
    """
    try:
        # check that symbol is valid
        stock = await get_stock_by_symbol(body.symbol)
        if not stock:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "message": "Stock not found"
            }

        # fetch user
        user = get_user_record(userId, db)
        if not user:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "message": "User not found"
            }

        return get_stock_recommendation(stock, user, body.amount)

    except Exception as e:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "message": e
        }

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

        user = get_user_record(userId, db)
        # get current portfolio value
        portfolio = get_portfolio_from_user(user)
        current_value = get_portfolio_value(portfolio)
        # handle edge case where implied portfolio value is zero or less
        if current_value + amount <= 0:
            if body.action == "review":
                # user is asking for a portfolio review where implied value is zero
                # we will use an implied value of $1,000
                amount = 1000
            else:
                # TO DO
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": "The implied value of the user's portfolio is less than or equal to zero.",
                }

        res = get_recom_transactions(user, amount)

        # insert advice record
        if userId is not None:
            insert_advice_record({
                "userId": userId,
                "action": body.action.upper(),
                "amount": body.amount,
                "transactions": res["transactions"],
            }, db)

            db.commit()

        # add message to help LLM understand function output
        profile = get_profile_from_user(user)
        message = (
            "These transactions are recommended for the user based on their objective - {}. ".format(OBJECTIVE_MAP[profile.objective]["description"]) +
            "Transactions are recommended by comparing the user's current portfolio to an optimal portfolio. The utility function is a Treynor ratio that is adjusted for the user's investing preferences."
        )

        return {
            "message": message,
            **res,
        }

    except Exception as e:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "message": e
        }
