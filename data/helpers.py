import asyncio

from data.fmp import ApiClient
from data.yahoo import get_stock_info # backup data source

async def get_aggregated_stock_data(symbol: str, exchange: str = 'NASDAQ', _quote: dict = None):
    """
    Aggregates data from required endpoints and format into a database-friendly record.
    """
    client = ApiClient()

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

    # exclude inactive stocks and etfs
    if not profile.get("isActivelyTrading") or profile.get("isEtf") or profile.get("isFund"):
        # exclude etfs
        return None

    # price target is essential for functionality
    # FMP does not have consensus info for ASX stocks
    # we will use our backup data source instead
    priceTarget = consensus.get('targetConsensus') if consensus else None
    if priceTarget is None:
        info = get_stock_info(symbol)
        priceTarget = info.get('priceTarget')

    return {
        'symbol': quote.get('symbol'),
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
        'epsGrowth': growth.get('growthEPS') if growth else None,
        'priceTarget': priceTarget,
    }