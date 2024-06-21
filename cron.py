import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from data import StockDataClient
from database import SessionLocal
from crud import upsert_stock

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
        # Stock not found
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
        'image': profile.get('image'),
        'sector': profile.get('sector'),
        'beta': profile.get('beta'),
        'pe': quote.get('pe'),
        'dividendAmount': profile.get('lastDiv'),
        'dividendYield': ratios.get('dividendYield') if ratios else None,
        'epsGrowth': growth.get('growthEps') if growth else None,
        'priceTarget': consensus.get('targetConsensus') if consensus else None,
    }


async def refresh_all_stock_data(exchange: str) -> None:
    if not (exchange == "ASX" or exchange == "NASDAQ"):
        raise Exception("Exchange must be ASX or NASDAQ")
    
    try:
        db = SessionLocal()
        stocks = await client.get_all_stocks_by_exchange(exchange)
        max_calls = 50
        delay_per_call = 60 / max_calls

        for quote in stocks:
            min_cap = 10_000_000_000 if exchange == "NASDAQ" else 1_000_000_000
            if quote['marketCap'] < min_cap:
                # TO DO: Handle stocks that are not updated here
                continue

            try:
                min_wait = asyncio.create_task(asyncio.sleep(delay_per_call))
                data = await get_aggregated_stock_data(quote['symbol'], exchange, quote)
                if data:
                    upsert_stock(data, db)  # Assuming there's an upsert_stock function
                    print(f"Data updated for {quote['symbol']}")
                
                await min_wait
            except Exception as e:
                print(f"Could not refresh data for {quote['symbol']}: {str(e)}")

        print(f"Data updated for exchange {exchange}")
    except Exception as e:
        print(f"Error refreshing data for exchange {exchange}: {str(e)}")
    finally:
        # close db
        db.close()
    
def schedule_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        lambda: refresh_all_stock_data("ASX"), 
        trigger=CronTrigger(hour=15, minute=0, day_of_week='mon-fri'),
        id="refresh_asx",
        name="Refresh ASX data at 5pm AEST",
        replace_existing=True
    )

    scheduler.add_job(
        lambda: refresh_all_stock_data("NASDAQ"), 
        trigger=CronTrigger(hour=9, minute=0, day_of_week='tue-sat'),
        id="refresh_nasdaq",
        name="Refresh NASDAQ data at 9am AEST",
        replace_existing=True
    )