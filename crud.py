from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

import models
from schemas import Stock

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

def upsert_stock(data: dict, db: Session, should_commit: bool = True):
    stmt = insert(models.Stock).values(**data)
    # Define the do_update clause for the existing rows
    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=['symbol'],
        set_=data
    )
    # Execute the statement
    db.execute(on_conflict_stmt)

    if should_commit:
        db.commit()

def delete_stock_by_symbol(symbol: str, db: Session, should_commit: bool = True):
    stmt = delete(models.Stock).where(models.Stock.symbol == symbol)
    db.execute(stmt)

    if should_commit:
        db.commit()