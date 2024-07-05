import asyncio

from sqlalchemy import delete, and_
from sqlalchemy.dialects.postgresql import insert

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from data.helpers import get_aggregated_stock_data
from data.fmp import ApiClient
from universe import Universe
from models import Stock
from database import SessionLocal

async def refresh_stock_data_by_exchange(exchange: str) -> None:
    if not (exchange == "ASX" or exchange == "NASDAQ"):
        raise Exception("Exchange must be ASX or NASDAQ")

    try:
        print("Updating data for exchange: ", exchange)
        # initialise db connection
        db = SessionLocal()
        # fetch all symbols for exchange
        stocks = await ApiClient().get_all_stocks_by_exchange(exchange)
        # keep track of updated symbols
        updated = []
        errored = []
        max_calls = 50
        delay_per_call = 60 / max_calls
        # initialise a min_wait task
        min_wait = asyncio.create_task(asyncio.sleep(0))
        for quote in stocks:
            try:
                min_cap = 50_000_000_000 if exchange == "NASDAQ" else 5_000_000_000
                if quote['marketCap'] and quote['marketCap'] < min_cap:
                    continue

                # wait for min_wait
                await min_wait
                # fetch data
                data = await get_aggregated_stock_data(quote['symbol'], exchange, quote)
                # create a new min_wait task
                min_wait = asyncio.create_task(asyncio.sleep(delay_per_call))
                if data:
                    # upsert data
                    stmt = insert(Stock).values(data)
                    on_conflict_stmt = stmt.on_conflict_do_update(
                        index_elements=['symbol'],
                        set_=data
                    )
                    db.execute(on_conflict_stmt)
                    # append symbol to updated array
                    updated.append(quote['symbol'])
                    print(f"Data updated for {quote['symbol']}", end="\r")

            except Exception as e:
                errored.append(quote['symbol'])
                print(f"Could not refresh data for {quote['symbol']}: {str(e)}")

        # delete all sybmols that were not updated
        delete_stmt = delete(Stock).where(
            and_(
                ~Stock.symbol.in_(updated),
                Stock.exchange == exchange
            )
        )
        db.execute(delete_stmt)
        # commit changes
        db.commit()
        # revalidate universe
        Universe().revalidate()
        print("Data updated for symbols", ",".join(updated))
        print("Update errored for symbols", ",".join(errored))
    except Exception as e:
        print(f"Error refreshing data for exchange {exchange}: {str(e)}")
    finally:
        # close db
        db.close()

def schedule_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        refresh_stock_data_by_exchange,
        args=["ASX"],
        trigger=CronTrigger(hour=15, minute=0, day_of_week='mon-fri'),
        id="refresh_asx",
        name="Refresh ASX data at 5pm AEST",
        max_instances=1
    )

    scheduler.add_job(
        refresh_stock_data_by_exchange,
        args=["NASDAQ"],
        trigger=CronTrigger(hour=8, minute=0, day_of_week='tue-sat'),
        id="refresh_nasdaq",
        name="Refresh NASDAQ data at 9am AEST",
        max_instances=1
    )