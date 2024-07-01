import numpy as np
import pandas as pd

from sqlalchemy.orm import Session

from database import SessionLocal
from crud import get_all_stocks

class Universe:
    data: pd.DataFrame = pd.DataFrame()
    r_f: float | None = None # risk-free rate
    MRP: float | None = None # market risk premium

    def __new__(cls):
        # check if instance already exists and return
        if not hasattr(cls, 'instance'):
            cls.instance = super(Universe, cls).__new__(cls)
        return cls.instance

    def revalidate(self):
        db: Session = SessionLocal()
        data = get_all_stocks(db)
        # close db
        db.close()
        # convert to dataframe
        df = pd.DataFrame.from_records([m.__dict__ for m in data], index='id').drop('_sa_instance_state', axis=1)
        # add expected return column
        df["expReturn"] = df.apply(lambda x: x["priceTarget"] / x["previousClose"] - 1, axis=1).fillna(0)
        # update state
        self.data = df
        
    def get(self):
        if self.data.empty:
            self.revalidate()
            
        return self.data

    def get_stock_by_id(self, _id: int):
        df = self.get()
        if _id not in df.index.values:
            return None

        # replace NaN with None to ensure json serialisable
        _stock = df.loc[_id].replace(np.nan, None)
        return {
            "id": _id, # avoid np.int64 to ensure json serialisable
            **_stock.to_dict()
        }

    def get_stock_by_symbol(self, symbol: str):
        df = self.get()
        # ensure symbol is uppercase
        symbol = symbol.upper()
        if symbol not in df["symbol"].values:
            # try appending '.AX' to symbol
            if f"{symbol}.AX" in df["symbol"].values:
                symbol = f"{symbol}.AX"
            else:
                return None

        _id = df[df["symbol"] == symbol].index[0]
        # replace NaN with None to ensure json serialisable
        _stock = df.loc[_id].replace(np.nan, None)
        return {
            "id": int(_id),
            **_stock.to_dict()
        }