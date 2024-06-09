from sqlalchemy import select
from sqlalchemy.orm import Session

from database import SessionLocal
import models, schemas

def get_stocks():
    # create new db connection
    db = SessionLocal()
    # fetch all stock records
    data = db.query(models.Stock).all()
    # close db connection before returning
    db.close()
    return data