
from typing import Optional, List, Dict

from pydantic import BaseModel

class Stock(BaseModel):
    id: int
    symbol: str
    name: str
    description: str
    image: Optional[str]
    exchange: str
    currency: str
    country: Optional[str]
    isEtf: bool
    previousClose: Optional[float]
    changesPercentage: Optional[float]
    priceTarget: Optional[float]
    beta: Optional[float]
    dividendAmount: Optional[float]
    dividendYield: Optional[float]
    marketCap: float
    sector: Optional[str]
    pe: Optional[float]
    epsGrowth: Optional[float]

class Holding(BaseModel):
    id: int
    units: int
    stockId: str
    userId: str
    
    class Config:
        from_attributes = True

class Profile(BaseModel):
    userId: str
    objective: str|None
    passive: float|None
    international: float|None
    preferences: Dict[str, str]

    class Config:
        from_attributes = True

class User(BaseModel):
    id: str

    holdings: List[Holding]
    profile: Optional[Profile]

    class Config:
        from_attributes = True

class PopulatedHolding(Stock, Holding):
    pass

class GetAdviceByStockRequest(BaseModel):
    symbol: str
    amount: float

class GetRecommendationsRequest(BaseModel):
    target: float # target amount to withdraw (negative) or deposit (positive)