import pandas as pd
from sqlalchemy.orm import Session

from database import SessionManager
import models

def get_all_stocks(manager: SessionManager = SessionManager()):
    # fetch all stock records
    return manager.session.query(models.Stock).all()
     
def get_stock_by_symbol(symbol: str, manager: SessionManager = SessionManager()):
    # fetch stock where symbol matches param
    return manager.session.query(models.Stock).filter(models.Stock.symbol == symbol.lower()).first()

def get_holdings(manager: SessionManager = SessionManager()):
    # fetch all holding records
    return manager.session.query(models.Holding).all()