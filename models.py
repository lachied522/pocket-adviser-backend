from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import backref, relationship

from database import Base

class Stock(Base):
    __tablename__ = 'Stock'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    name = Column(String)
    previousClose = Column(Float)
    priceTarget = Column(Float)
    beta = Column(Float, nullable=True)
    dividendAmount = Column(Float, nullable=True)
    marketCap = Column(Float)
    sector = Column(String)
    pe = Column(Float, nullable=True)
    epsGrowth = Column(Float, nullable=True)

class Holding(Base):
    __tablename__ = 'Holding'
    id = Column(Integer, primary_key=True)
    units = Column(Integer)
    stockId = Column(String, ForeignKey('Stock.id'))