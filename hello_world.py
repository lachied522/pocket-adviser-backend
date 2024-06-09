import pandas as pd

from universe import get_universe
from optimiser import Optimser

holdings = [
    {
        'symbol': 'aapl',
        'units': 10,
    },
    {
        'symbol': 'unh',
        'units': 10,
    }
]

universe = get_universe()

optimiser = Optimser(pd.DataFrame.from_records(holdings), universe)

optimiser.get_optimal_portfolio()