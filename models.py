from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import backref, relationship
from sqlalchemy.types import JSON

from database import Base

class User(Base):
    __tablename__ = 'User'
    id = Column(String, primary_key=True)

    profile = relationship('Profile', backref=backref('user'), lazy='joined')
    holdings = relationship('Holding', backref=backref('user'), lazy='joined')

class Profile(Base):
    __tablename__ = 'Profile'
    userId = Column(String, ForeignKey('User.id'), primary_key=True)
    objective = Column(String, nullable=True)
    passive = Column(Float, nullable=True)
    international = Column(Float, nullable=True)
    preferences = Column(JSON)

class Stock(Base):
    __tablename__ = 'Stock'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    name = Column(String)
    exchange = Column(String)
    country = Column(String)
    isEtf = Column(Boolean)
    previousClose = Column(Float, nullable=True)
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
    userId = Column(String, ForeignKey('User.id'))
