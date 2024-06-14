from typing import List, Dict

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from helpers import get_universe, merge_portfolio_with_universe

OBJECTIVE_MAP = {
    "retirement": {
        "target_beta": 1.25,
        "beta_quantiles": [0.25, 1.0], # quantile of beta for stocks
        "target_div_yield": 0.02,
        "target_sector_allocation": {
            "basic-materials": 0.0191,
            "financial-services": 0.0747,
            "healthcare": 0.1740,
            "energy": 0.0698,
            "consumer-cyclical": 0.1046,
            "communication-services": 0.0780,
            "industrials": 0.0511,
            "consumer-defensive": 0.0646,
            "real-estate": 0.0084,
            "technology": 0.3485,
            "utilities": 0.005
            # https://www.ishares.com/us/products/239725/ishares-sp-500-growth-etf
        }
    },
    "income": {
        "target_beta": 0.75,
        "beta_quantiles": [0, 0.75],
        "target_div_yield": 0.05,
        "target_sector_allocation": {
            "basic-materials": 0.0754,
            "financial-services": 0.2565,
            "healthcare": 0.0097,
            "energy": 0.0505,
            "consumer-cyclical": 0.0635,
            "communication-services": 0.0530,
            "industrials": 0.1240,
            "consumer-defensive": 0.0611,
            "real-estate": 0.1543,
            "technology": 0.0395,
            "utilities": 0.1124
            # https://www.ssga.com/au/en_gb/intermediary/etfs/funds/spdr-sp-global-dividend-fund-wdiv
        }
    },
    "preservation": {
        "target_beta": 0.50,
        "beta_quantiles": [0, 0.75],
        "target_div_yield": 0.03,
        "target_sector_allocation": {
            "basic-materials": 0.0032,
            "financial-services": 0.1160,
            "healthcare": 0.1430,
            "energy": 0.025,
            "consumer-cyclical": 0.130,
            "communication-services": 0.093,
            "industrials": 0.110,
            "consumer-defensive": 0.122,
            "real-estate": 0.009,
            "technology": 0.140,
            "utilities": 0.08
            # https://www.vanguard.com.au/personal/invest-with-us/etf?portId=8201
        }
    },
    "first-home": {
        "target_beta": 1.1,
        "beta_quantiles": [0.25, 0.9],
        "target_div_yield": 0.01,
        "target_sector_allocation": {
            "basic-materials": 0.0242,
            "financial-services": 0.1296,
            "healthcare": 0.1337,
            "energy": 0.0457,
            "consumer-cyclical": 0.1067,
            "communication-services": 0.0877,
            "industrials": 0.0826,
            "consumer-defensive": 0.0666,
            "real-estate": 0.0240,
            "technology": 0.2708,
            "utilities": 0.0257
            # https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf
        }
    },
    "children": {
        "target_beta": 1.25,
        "beta_quantiles": [0.25, 0.9],
        "target_div_yield": 0.02,
        "target_sector_allocation": {
            "basic-materials": 0.0157,
            "financial-services": 0.1000,
            "healthcare": 0.1284,
            "energy": 0.0593,
            "consumer-cyclical": 0.1081,
            "communication-services": 0.0868,
            "industrials": 0.0447,
            "consumer-defensive": 0.0950,
            "real-estate": 0.0038,
            "technology": 0.3494,
            "utilities": 0.0062
            # https://www.ishares.com/us/products/239737/ishares-global-100-etf
        }
    },
    "trading": {
        "target_beta": 1.5,
        "beta_quantiles": [0.0, 1.0],
        "target_div_yield": None, # no target div yield for trading
        "target_sector_allocation": None # no target allocations for trading
    }
}


class Optimser:
    portfolio: pd.DataFrame # symbol, units, cost, for each stock held by user
    universe: pd.DataFrame # stock data for all available stocks in universe
    wp: pd.DataFrame # combined universe and portfolio were units is zero for stocks not held
    target: float # target portfolio value
    delta_value: float # target change in portfolio value
    objective: str = 'retirement' # user objective - is key of OBJECTIVE_MAP
    preferences: Dict[str, str] = {} # user preferences of form { [sector]: 'like'|'dislike' }
    error: float = 0.03 # margin for error when using target weights or amounts
    bias: float = 0.01 # bias placed on stocks based on preferences
    threshold: float = 0.05 # minimum weight for stock to be included in optimal portfolio
    a0: pd.Series
    formula: str = 'treynor' # formula used for utility function

    def __init__(
        self,
        portfolio: list | pd.DataFrame,
        delta_value: float = 0, # target change in portfolio value
        objective: str = 'retirement',
        preferences: Dict[str, str] = {},
        error: float = 0.03,
        bias: float = 0.01,
        threshold: float = 0.05,
        formula: str = 'treynor'
    ):
        self.portfolio = portfolio if type(portfolio) == pd.DataFrame else pd.DataFrame.from_records(portfolio)
        self.universe = get_universe()
        # merge portfolio and universe to get working portfolio
        wp = merge_portfolio_with_universe(self.universe, portfolio)
        self.wp = wp
        # define inital portfolio
        self.a0 = wp['units'] * wp['previousClose']
        # set target value for portfolio as current portfolio value plus delta
        self.delta_value = delta_value
        self.target = (wp['units'] * wp['previousClose']).sum() + delta_value
        # set objective and preferences
        self.objective = objective
        self.preferences = preferences

        self.error = error
        self.bias = bias
        self.threshold = threshold

        self.formula = formula

    def inv_utility_function(
            self,
            a: np.ndarray | pd.Series,
            df: pd.DataFrame,
            additional_factors: List[np.ndarray] = []
        ):
        """
        Inverse utility function for optimisation.
        """
        r_f, MRP = 0.05, 0.08 # TO DO
        match self.formula:
            case 'treynor':
                # Treynor ratio = (r_p - r_f) / Beta_p
                # see https://www.investopedia.com/terms/t/treynorratio.asp
                r_p = np.dot(a, df["exp_return"] + np.array(additional_factors).sum(axis=0))
                Beta_p = np.dot(a, df["beta"])
                U = (r_p - r_f) / Beta_p

            case 'sharpe':
                # Sharpe ratio = (r_p - r_f) / Sigma_p, see https://www.investopedia.com/terms/t/treynorratio.asp
                # requires stock volatilities
                raise NotImplementedError()

        # penalty is applied to discourage weights between 0 and 5% of the portfolio and deviations from current values
        # multiplication by 2 represents consideration for fees incurred both by a 'buy' and 'sell' transaction
        # this doesn't appear to work very well as weights close 0 and 5% are generated frequently
        # and deviations from current values are common

        penalty = 2 * (np.dot(a, np.logical_and(0 < a, a < self.target * self.threshold).astype(int)) + np.dot(a, a - self.a0 != 0)) / self.target
        return -(U-penalty)

    def get_constraints(self, df: pd.DataFrame):
        """
        Linear constraints of lb < A.x < ub. Returns tuple.
        """
        # weights must sum to one, within error
        lb, ub = self.target * (1-self.error), self.target * (1+self.error)
        sum_cons = [LinearConstraint(np.ones(len(df)), lb, ub)]

        # yield constraint
        yield_cons = []
        # target_div_yield = OBJECTIVE_MAP[self.objective]["target_div_yield"]
        # if target_div_yield is not None:
        #     div_yield = df["div_yield"].fillna(0)
        #     yield_cons = [LinearConstraint(div_yield, self.target*max(0, target_div_yield-0.01), self.target*(target_div_yield+0.01))] # error for dividends is 1%

        # active constraint
        # active_cons = []
        # target_active = self.target_active
        # if 0 < target_active and target_active < 1:
        #     active = np.array(wp['active']==True).astype(int)
        #     active_cons = [LinearConstraint(active, value*max(0, target_active-error), value*min(1, target_active+error))]

        # domestic constraint
        # domestic_cons = []

        # constrain weight of 'locked' holdings
        # locked_cons = []
        # # avoid edge case where value of 'locked' holdings is greater than target value
        # locked = wp[wp['locked']==True]
        # if value > np.sum(locked['units'].fillna(0) * locked['last_price']):
        #     for _, row in locked.iterrows():
        #         stock_vector = np.array(wp['symbol']==row['symbol'])
        #         # set inequality constraint with target value as upper limit
        #         locked_cons.append(LinearConstraint(stock_vector, row['last_price'] * row['units'], value))

        # sector constraints
        sector_cons = []
        sector_allocation_map = OBJECTIVE_MAP[self.objective]["target_sector_allocation"]
        if sector_allocation_map is not None:
            # avoid multi collinearity by removing one of the sectors
            target_sectors = list(sector_allocation_map.items())[:-1]

            # target sector allocations must be adjusted for locked holdings
            # if wp['locked'].any():
            #     locked = wp[wp['locked']==True]
            #     s = 1 # scaling factor for adjusting un-'locked' sectors by
            #     for i, (sector, target) in enumerate(target_sectors):
            #         # check if locked allocation is greater than target allocation
            #         locked_allocation = np.sum([s['units'] * s['last_price'] for _, s in locked[locked['sector']==sector].iterrows()])
            #         if locked_allocation >= value*sector_allocation_map[sector]:
            #             # target allocation is already met, remove sector from target allocations
            #             target_sectors.remove(target_sectors[i])
            #             s -= locked_allocation / value

            #     # adjust target allocations of remaining sectors
            #     sector_allocation_map = {k: v * s for k, v in sector_allocation_map.items()}

            # seem to have better results by defining constraint for each sector individually
            for sector, target in target_sectors:
                # array indicating which stocks belong to each sector
                sector_vector = np.array(df["sector"] == sector).astype(int)
                sector_cons.append(LinearConstraint(sector_vector, self.target*max(0, target-self.error), self.target*min(1, target+self.error)))

        return tuple(
            sum_cons +
            yield_cons +
            sector_cons
            # domestic_cons
            # locked_cons
        )

    def get_additional_factors(self, df: pd.DataFrame) -> list:
        additional_factors = []
        # featured stocks
        # additional_factors.append(self.bias * np.array(df['tags'].apply(lambda x: 'Featured' in x)))

        # user preferences
        if self.preferences is not None:
            for sector, preference in self.preferences.items():
                factor = self.bias * self.target * np.array(df["sector"] == sector).astype(int)

                if preference=="dislike":
                    # reverse direction of bias
                    factor *= -1

                additional_factors.append(factor)

        return additional_factors

    def apply_filers(self, df: pd.DataFrame):
        # TO DO
        return df

    def get_optimal_portfolio(self):
        # apply filters
        df = self.apply_filers(self.wp.copy())

        # get constraints
        cons = self.get_constraints(df)

        # get additional factors
        additional_factors = self.get_additional_factors(df)

        # define boundaries
        # lower bound equal to zero (no shorts)
        lb = np.zeros(len(df))
        # upper bound equal to total value of portfolio
        ub = self.target * np.ones(len(df))
        # set keep_feasible True to ensure iterations remain within bounds
        bnds = Bounds(lb, ub, keep_feasible=True)

        # first guess for minimiser is equal weight
        equal_weight = self.target * np.ones(len(df)) / len(df)

        # SLSQP appears to perform the best
        a = minimize(self.inv_utility_function, equal_weight, args=(df, additional_factors),
                    method='SLSQP', bounds=bnds, constraints=cons,
                    options={'maxiter': 100}).x

        # update units column of optimal portfolio and return
        df['units'] = np.round(a / df['previousClose'])
        # drop zero unit rows and return
        self.optimal_portfolio = df.drop(df[df['units'] < 1].index)
        return self.optimal_portfolio
    
    def get_utility(self, portfolio: pd.DataFrame):
        """
        Helper function for obtaining the adjusted utility of a portfolio.
        """
        # merge portfolio and universe to get working portfolio
        wp = merge_portfolio_with_universe(self.universe, portfolio)
        # extract amount
        a = wp["units"].fillna(0) * wp["previousClose"]
        return -self.inv_utility_function(a, self.wp, self.get_additional_factors(portfolio))