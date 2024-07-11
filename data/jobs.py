import traceback

import asyncio

from sqlalchemy import delete, and_
from sqlalchemy.dialects.postgresql import insert

from data.helpers import get_aggregated_stock_data
from data.financial_modelling_prep import ApiClient
from universe import Universe
from models import Stock
from database import SessionLocal

EXCHANGES = {
    "ASX": {
        "min_cap": 5_000_000_000
    },
    "NASDAQ": {
        "min_cap": 50_000_000_000
    },
    "NYSE": {
        "min_cap": 100_000_000_000
    }
}

def commit_changes(upsert_stmts, delete_stmt):
    # initialise db connection
    db = SessionLocal()
    attempts = 0
    while attempts < 3:
        try:
            attempts += 1
            # execute upsert statement
            db.execute(upsert_stmts)
            # execute delete statement
            db.execute(delete_stmt)
            # commit changes
            db.commit()
            # close connection
            db.close()
            print("Changes commited")
            return
        except:
            db.rollback()
            print("Rolled back")

    print("Error executing statements after three attempts")

async def refresh_data_by_exchange(exchanges: str|list[str]) -> None:
    if isinstance(exchanges, str):
        exchanges = [exchanges]
    
    for exchange in exchanges:
        if not exchange in EXCHANGES:
            raise Exception("Exchange must be one of ", ", ".join(EXCHANGES.keys()))

        try:
            print("Updating data for exchange: ", exchange)
            # fetch all symbols for exchange
            stocks = await ApiClient().get_all_stocks_by_exchange(exchange)
            # some stocks have more than one listing on the same exchange, e.g. BAC
            # we will keep track of names that have been updated
            # the primary listing will always be first alphabetically
            updated = {
                "symbols": [],
                "names": [],
            }
            errored = [] # symbols of errored stocks
            to_upsert = []
            # financial modelling prep is rate limited to 300 calls/min
            # we will limit to 150 to allow room calls elsewhere
            max_calls = 150
            delay_per_call = 60 / max_calls
            # initialise a min_wait task
            min_wait = asyncio.create_task(asyncio.sleep(0))
            for quote in stocks:
                try:
                    if quote['marketCap'] is None or quote['marketCap'] < EXCHANGES[exchange]['min_cap']:
                        continue

                    # check that name does not already exist in updated list
                    if quote['name'] in updated['names']:
                        continue

                    # wait for min_wait
                    await min_wait
                    # fetch data
                    data = await get_aggregated_stock_data(quote['symbol'], exchange, quote)
                    # create a new min_wait task
                    min_wait = asyncio.create_task(asyncio.sleep(delay_per_call))
                    if data:
                        # append data to upsert array
                        to_upsert.append(data)
                        # append symbol to updated array
                        updated["symbols"].append(quote['symbol'])
                        updated["names"].append(quote['name'])
                        # print(f"Retreived data for {quote['symbol']}", end="\r")

                except Exception as e:
                    errored.append(quote['symbol'])
                    print(f"Could not refresh data for {quote['symbol']}: ", str(e))

            # upsert statement for updated symbols
            insert_stmts = insert(Stock).values(to_upsert)
            upsert_stmts = insert_stmts.on_conflict_do_update(
                index_elements=['symbol'],
                set_={c.key: c for c in insert_stmts.excluded}
            )

            # delete all symbols that were not updated
            delete_stmt = delete(Stock).where(
                and_(
                    ~Stock.symbol.in_(updated['symbols']),
                    Stock.exchange == exchange
                )
            )
            
            commit_changes(upsert_stmts, delete_stmt)
            # revalidate universe
            Universe().revalidate()
            print("Data updated for symbols", ",".join(updated['symbols']))
            if len(errored) > 0:
                print("Update errored for symbols", ",".join(errored))
        except Exception as e:
            traceback.print_exc()
            print(f"Error refreshing data for exchange {exchange}: {str(e)}")