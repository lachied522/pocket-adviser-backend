"""
This module will be used as a backup source of stock data.
"""

import yfinance as yf

def get_stock_info(symbol: str):
    # Yahoo finance uses same symbol convention as FMP, i.e. '.AX' suffix
    stock = yf.Ticker(symbol)

    info = stock.info
    
    # some fields must be formatted
    epsGrowth = None
    if info.get('trailingEps') and info.get('forwardEps'):
        epsGrowth = info.get('forwardEps') / info.get('trailingEps') - 1

    country = "US"
    if info.get("country") == "Australia":
        country = "AU"

    return {
        'symbol': symbol,
        'previousClose': info.get('previousClose'),
        'changesPercentage': info.get('changesPercentage'),
        'marketCap': info.get('marketCap'),
        'exchange': info.get('exchange'),
        'name': info.get('longName'),
        'description': info.get('longBusinessSummary'),
        'currency': info.get('financialCurrency'),
        'isEtf': info.get('quoteType') != "EQUITY",
        'sector': info.get('sector'),
        'beta': info.get('beta'),
        'pe': info.get('trailingPE'),
        'dividendAmount': info.get('dividendRate'),
        'dividendYield': info.get('dividendYield'),
        'priceTarget': info.get('targetMeanPrice'),
        'country': country,
        'epsGrowth': epsGrowth,
    }