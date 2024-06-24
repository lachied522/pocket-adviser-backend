import asyncio

from sqlalchemy import delete, and_
from sqlalchemy.dialects.postgresql import insert

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from data import StockDataClient
from universe import Universe
from models import Stock
from database import SessionLocal

client = StockDataClient()

async def get_aggregated_stock_data(symbol: str, exchange: str = 'NASDAQ', _quote: dict = None):
    # Aggregate data from required endpoints and format into a database-friendly record

    # Add '.AX' if exchange is ASX
    if exchange == 'ASX' and not symbol.endswith('.AX'):
        symbol += '.AX'

    # Ensure symbol is capitalized
    symbol = symbol.upper()

    # Creating asynchronous tasks for API requests
    tasks = [
        asyncio.create_task(client.get_quote(symbol)) if not _quote else asyncio.create_task(asyncio.sleep(0, result=_quote)),
        asyncio.create_task(client.get_company_profile(symbol)),
        asyncio.create_task(client.get_price_target(symbol)),
        asyncio.create_task(client.get_growth_rates(symbol)),
        asyncio.create_task(client.get_ratios(symbol)),
    ]

    # Wait for all tasks to complete
    quote, profile, consensus, growth, ratios = await asyncio.gather(*tasks)

    if not (quote and profile):
        # stock not found
        return None

    if profile.get("isFund"):
        # exclude funds
        return None

    return {
        'symbol': symbol,
        'previousClose': quote.get('price'),
        'changesPercentage': quote.get('changesPercentage'),
        'marketCap': quote.get('marketCap'),
        'exchange': quote.get('exchange'),
        'name': profile.get('companyName'),
        'description': profile.get('description'),
        'currency': profile.get('currency'),
        'country': profile.get('country'),
        'isEtf': profile.get('isEtf'),
        'sector': profile.get('sector'),
        'beta': profile.get('beta'),
        'pe': quote.get('pe'),
        'dividendAmount': profile.get('lastDiv'),
        'dividendYield': ratios.get('dividendYield') if ratios else None,
        'epsGrowth': growth.get('growthEps') if growth else None,
        'priceTarget': consensus.get('targetConsensus') if consensus else None,
    }

async def refresh_stock_data_by_exchange(exchange: str) -> None:
    if not (exchange == "ASX" or exchange == "NASDAQ"):
        raise Exception("Exchange must be ASX or NASDAQ")
    
    try:
        db = SessionLocal()
        # fetch all symbols for exchange
        stocks = await client.get_all_stocks_by_exchange(exchange)
        # keep track of updated symbols
        updated = []
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
                    print(f"Data updated for {quote['symbol']}")

            except Exception as e:
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
        print(f"Data updated for exchange {exchange}")
    except Exception as e:
        print(f"Error refreshing data for exchange {exchange}: {str(e)}")
    finally:
        # close db
        db.close()
    
def schedule_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        lambda: refresh_stock_data_by_exchange("ASX"),
        trigger=CronTrigger(hour=15, minute=0, day_of_week='mon-fri'),
        id="refresh_asx",
        name="Refresh ASX data at 5pm AEST",
        replace_existing=True
    )

    scheduler.add_job(
        lambda: refresh_stock_data_by_exchange("NASDAQ"),
        trigger=CronTrigger(hour=9, minute=0, day_of_week='tue-sat'),
        id="refresh_nasdaq",
        name="Refresh NASDAQ data at 9am AEST",
        replace_existing=True
    )