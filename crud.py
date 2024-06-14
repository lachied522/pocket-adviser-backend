import pandas as pd
from sqlalchemy.orm import Session

from database import SessionLocal
import models

def get_all_stocks(db: Session = SessionLocal()):
    # fetch all stock records
    data = db.query(models.Stock).all()
    # close db before returning
    db.close()

    return data
     
def get_stock_by_symbol(symbol: str, db: Session = SessionLocal()):
    # fetch stock where symbol matches param
    data = db.query(models.Stock).filter(models.Stock.symbol == symbol.lower()).first()
    # close db before returning
    db.close()
    
    return data

def get_holdings(db: Session = SessionLocal()):
    # fetch all holding records
    data = db.query(models.Holding).all()
    # close db before returning
    db.close()
    
    return data