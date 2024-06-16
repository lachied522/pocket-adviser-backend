from sqlalchemy.orm import Session

import models

def get_all_stocks(db: Session):
    # fetch all stock records
    return db.query(models.Stock).all()
     
def get_stock_by_symbol(symbol: str, db: Session):
    # fetch stock where symbol matches param
    return db.query(models.Stock).filter(models.Stock.symbol == symbol.upper()).first()

def get_holdings_by_user_id(userId: str, db: Session):
    # fetch all holding records
    return db.query(models.Holding).filter(models.Holding.userId == userId).all()

def get_profile_by_user_id(userId: str, db: Session):
    # fetch all holding records
    return db.query(models.Profile).filter(models.Profile.userId == userId).first()