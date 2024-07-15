from sqlalchemy import Column, Integer, String, Float, Boolean, ARRAY, ForeignKey
from sqlalchemy.orm import backref, relationship
from sqlalchemy.types import JSON, DateTime

from database import Base

class User(Base):
    __tablename__ = 'User'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    accountType = Column(String)
    mailFrequency = Column(String)

    profile = relationship('Profile', backref=backref('user'), lazy='joined')
    holdings = relationship('Holding', backref=backref('user'), lazy='joined')
    advice = relationship('Advice', backref=backref('user'), lazy='joined')

class Profile(Base):
    __tablename__ = 'Profile'
    userId = Column(String, ForeignKey('User.id'), primary_key=True)
    objective = Column(String, nullable=True)
    passive = Column(Float, nullable=True)
    international = Column(Float, nullable=True)
    preferences = Column(JSON)

class Advice(Base):
    __tablename__ = 'Advice'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String)
    amount = Column(Float)
    transactions = Column(ARRAY(JSON))
    initialAdjUtility = Column(Float, nullable=True)
    finalAdjUtility = Column(Float, nullable=True)
    createdAt = Column(DateTime, nullable=True)
    userId = Column(String, ForeignKey('User.id'))

class Stock(Base):
    __tablename__ = 'Stock'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    name = Column(String)
    description = Column(String)
    exchange = Column(String)
    currency = Column(String)
    country = Column(String, nullable=True)
    isEtf = Column(Boolean)
    previousClose = Column(Float, nullable=True)
    changesPercentage = Column(Float, nullable=True)
    priceTarget = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    dividendAmount = Column(Float, nullable=True)
    dividendYield = Column(Float, nullable=True)
    marketCap = Column(Float)
    sector = Column(String, nullable=True)
    pe = Column(Float, nullable=True)
    epsGrowth = Column(Float, nullable=True)

class Holding(Base):
    __tablename__ = 'Holding'
    id = Column(Integer, primary_key=True)
    units = Column(Integer)
    stockId = Column(String, ForeignKey('Stock.id'))
    userId = Column(String, ForeignKey('User.id'))
