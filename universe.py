import pandas as pd

from crud import get_stocks


def get_universe(as_dataframe: bool = True):
    data = get_stocks()
    
    if as_dataframe:
        df = pd.DataFrame.from_records([m.__dict__ for m in data])
        # drop unnecessary columns
        return df.drop(['_sa_instance_state', 'id'], axis=1)
    
    return data