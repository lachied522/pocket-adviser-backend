import os
from typing import Optional, Union, Dict, List

import aiohttp

class ApiClient:
    API_KEY: str
    API_BASE_URL: str

    def __init__(self):
        API_KEY = os.getenv("FMP_API_KEY")
        if not API_KEY:
            raise Exception("Api key not defined")
        
        self.API_KEY = API_KEY
        self.API_BASE_URL = "https://financialmodelingprep.com/api"

    async def make_authenticated_api_request(self, endpoint: str, params: dict = None, version: Union[int, int] = 3):
        if params is None:
            params = {}
        params['apikey'] = self.API_KEY
        
        url = f"{self.API_BASE_URL}/v{version}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if not response.status == 200:
                    print(response.json())
                    raise Exception(f"Error making API request, status: {response.status}")
                return await response.json()

    async def get_all_stocks_by_exchange(self, exchange: str = 'NASDAQ') -> List[dict]:
        data = await self.make_authenticated_api_request(f"symbol/{exchange}")
        return data

    async def get_quote(self, symbol: str) -> Optional[dict]:
        data = await self.make_authenticated_api_request(f"quote/{symbol}")
        if not data:
            return None
        return data[0]

    async def get_company_profile(self, symbol: str) -> Optional[dict]:
        data = await self.make_authenticated_api_request(f"profile/{symbol}")
        if not data:
            return None
        return data[0]

    async def get_price_target(self, symbol: str) -> Optional[dict]:
        params = {'symbol': symbol}
        data = await self.make_authenticated_api_request('price-target-consensus', params, 4)
        if not data:
            return None
        return data[0]

    async def get_growth_rates(self, symbol: str, period: str = "annual", limit: int = 1) -> Optional[dict]:
        params = {'period': period, 'limit': str(limit)}
        data = await self.make_authenticated_api_request(f"income-statement-growth/{symbol}", params)
        if not data:
            return None
        return data[0]

    async def get_ratios(self, symbol: str, period: str = "annual", limit: int = 1) -> Optional[dict]:
        params = {'period': period, 'limit': str(limit)}
        data = await self.make_authenticated_api_request(f"ratios/{symbol}", params)
        if not data:
            return None
        return data[0]

    async def get_news_articles(self, symbols: list[str], page: int = 0, limit: int = 12) -> list:
        params = {
            'tickers': ','.join([s.upper() for s in symbols]),
            'page': str(page),
            'limit': str(limit)
        }
        data = await self.make_authenticated_api_request("stock_news", params)
        return data

    async def get_analyst_research(self, symbol: str, limit: int = 10) -> Optional[dict]:
        params = {
            'symbol': symbol
        }
        data = await self.make_authenticated_api_request("price-target", params, 4)
        if not data:
            return None
        return data

    async def get_performance(self, symbols: str|list[str]) -> Optional[dict]:
        # this endpoint can be batched
        if isinstance(symbols, list):
            symbols = ",".join(symbols)
        
        data = await self.make_authenticated_api_request(f"stock-price-change/{symbols}")

        if not data:
            return None
        return data

    async def get_historical_price(self, symbol: str, _from: str|None = None, _to: str|None = None) -> Optional[dict]:
        params = {}

        if _from is not None and _to is not None:
            params["from"] = _from
            params["to"] = _to

        data = await self.make_authenticated_api_request(f"historical-price-full/{symbol}", params)
        if not data:
            return None
        return data

    async def get_treasury_rates(self, _from: str, _to: str|None = None) -> Optional[dict]:
        params = {
            "from": _from
        }

        if _to is not None:
            params["to"] = _to

        data = await self.make_authenticated_api_request("treasury", params, 4)
        if not data:
            return None
        return data