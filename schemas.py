
from typing import Optional

from pydantic import BaseModel

class Stock(BaseModel):
    id: int
    symbol: str
    name: str
    previousClose: float
    priceTarget: float
    beta: Optional[float]
    dividendAmount: Optional[float]
    marketCap: float
    sector: str
    pe: Optional[float]
    epsGrowth: Optional[float]

class Holding(BaseModel):
    id: int
    symbol: str
    units: int

class PopulatedHolding(Stock, Holding):
    pass