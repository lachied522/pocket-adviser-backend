from typing import List, Dict

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from params import OBJECTIVE_MAP
from helpers import get_universe, get_riskfree_rate, merge_portfolio_with_universe
from schemas import Profile

class Optimser:
    portfolio: pd.DataFrame # symbol, units, cost, for each stock held by user
    profile: Profile | None
    objective: str
    universe: pd.DataFrame # stock data for all available stocks in universe
    wp: pd.DataFrame # combined universe and portfolio were units is zero for stocks not held
    target: float # target portfolio value
    delta_value: float = 0 # target change in portfolio value
    error: float = 0.03 # margin for error when using target weights or amounts
    bias: float = 0.05 # bias placed on stocks based on preferences
    threshold: float = 0.05 # minimum weight for stock to be included in optimal portfolio
    a0: pd.Series
    formula: str = 'treynor' # formula used for utility function

    def __init__(
        self,
        portfolio: list | pd.DataFrame,
        profile: Profile | None,
        delta_value: float = 0, # target change in portfolio value
        error: float = 0.03,
        bias: float = 0.05,
        threshold: float = 0.05,
        formula: str = 'treynor'
    ):
        self.portfolio = portfolio if type(portfolio) == pd.DataFrame else pd.DataFrame.from_records(portfolio)
        self.objective = profile.objective if profile else "RETIREMENT"
        self.profile = profile
        # get universe
        self.universe = get_universe()
        # merge portfolio and universe to get working portfolio
        wp = merge_portfolio_with_universe(self.universe, portfolio)
        self.wp = wp
        # define inital portfolio
        self.a0 = wp['units'] * wp['previousClose']
        # set target value for portfolio as current portfolio value plus delta
        self.delta_value = delta_value
        self.target = (wp['units'] * wp['previousClose']).sum() + delta_value

        self.error = error
        self.bias = bias
        self.threshold = threshold

        self.formula = formula

    def inv_utility_function(
            self,
            a: np.ndarray | pd.Series,
            df: pd.DataFrame,
            additional_factors: List[np.ndarray] = [],
            include_penalty: bool = True
        ):
        """
        Inverse utility function for optimisation.
        """
        r_f = get_riskfree_rate() # TO DO
        match self.formula:
            case 'treynor':
                # Treynor ratio = (r_p - r_f) / Beta_p
                # see https://www.investopedia.com/terms/t/treynorratio.asp
                r_p = np.dot(a, df["expReturn"] + np.array(additional_factors).sum(axis=0))
                Beta_p = np.dot(a, df["beta"])
                # prevent divide by zero by adding small amount to beta
                U = (r_p - r_f) / (Beta_p if Beta_p != 0 else 0.01)

            case 'sharpe':
                # Sharpe ratio = (r_p - r_f) / Sigma_p, see https://www.investopedia.com/terms/t/treynorratio.asp
                # requires stock volatilities
                raise NotImplementedError()

        # penalty is applied to discourage weights between 0 and 5% of the portfolio and deviations from current values
        # multiplication by 2 represents consideration for fees incurred both by a 'buy' and 'sell' transaction
        penalty = 0
        if include_penalty:
            penalty = 2 * (np.dot(a, np.logical_and(0 < a, a < self.target * self.threshold).astype(int)) + np.dot(a, a - self.a0 != 0)) / self.target
        return -(U-penalty)
    
    def apply_filters(self, df: pd.DataFrame):
        """
        Filters working portfolio based on investment style.
        """
        # handle edge cases
        if self.profile.passive == 1:
            # portfolio is all ETFs, direct equities can be removed.
            return df.drop(df[df['isEtf']==True].index)

        # extract beta thresholds from profile
        beta_thresholds = OBJECTIVE_MAP[self.objective]["beta_quantiles"]
        match self.objective:
            # filter working portfolio based on objective
            # case 'RETIREMENT':
            #     # calculate growth rates
            #     wp['growth'] = wp.apply(lambda x: (x['forward_EPS'] / x['trailing_EPS']) - 1 if x['forward_EPS'] and x['trailing_EPS'] else None, axis=1)
            #     # calculate PEG ratios
            #     wp['PEG'] = wp.apply(lambda x: x['PE'] / x['growth'] if x['PE'] and x['growth'] else None, axis=1)

            #     for sector in wp['sector'].unique():
            #         # filter stocks on per sector basis to preserve sector allocation
            #         if sector is not None:
            #             sector_mask = wp["sector"]==sector
            #             # cut stocks that are in bottom 50% of PEG
            #             lower_PEG = wp[sector_mask]['PEG'].median()
            #             PEG_filter = wp['PEG'] < lower_PEG
            #             # cut stocks that are outside quantiles for beta
            #             lower_beta = wp[sector_mask]['beta'].quantile(beta_thresholds[0])
            #             upper_beta = wp[sector_mask]['beta'].quantile(beta_thresholds[1])
            #             beta_filter = (wp[sector_mask]['beta'] < lower_beta) | (wp[sector_mask]['beta'] > upper_beta)
                        
            #             combined_filter = sector_mask & (PEG_filter | beta_filter)

            #             wp = wp.drop(wp[combined_filter & ~wp['locked']].index)

            case _:
                # filter stocks on per sector basis to preserve sector allocation
                for sector in df["sector"].unique():
                    if sector is not None:
                        sector_mask = df["sector"]==sector
                        # cut stocks that are outside quantiles for beta
                        lower_beta = df[sector_mask]["beta"].quantile(beta_thresholds[0])
                        upper_beta = df[sector_mask]["beta"].quantile(beta_thresholds[1])
                        beta_filter = (df[sector_mask]["beta"] < lower_beta) | (df[sector_mask]["beta"] > upper_beta)
                        
                        df = df.drop(df[beta_filter].index)

        return df
    
    def get_sector_allocation(self):
        """
        Returns target sector allocation for optimisation based on objective and preferences if any.
        """
        targets = OBJECTIVE_MAP[self.objective]["sector_allocations"]

        if targets is None:
            # occurs when objective is TRADING
            return None

        if self.profile.preferences is not None:
            for key, value in self.profile.preferences.items():
                if key in targets:
                    if value == "like":
                        # bias up sector
                        targets[key] += 0.05
                    else:
                        # bias down sector
                        targets[key] = max(targets[key] - 0.05, 0)

            # ensure sector weights sum to 1
            s = sum(targets.values())
            targets = {k: v / s for k, v in targets.items()}
        
        return targets

    def get_constraints(self, df: pd.DataFrame):
        """
        Linear constraints of lb < A.x < ub. Returns tuple.
        """
        # weights must sum to one, within error
        lb, ub = self.target * (1-self.error), self.target * (1+self.error)
        sum_cons = [LinearConstraint(np.ones(len(df)), lb, ub)]

        # yield constraint
        yield_cons = []
        target_yield = OBJECTIVE_MAP[self.objective]["target_yield"]
        if target_yield is not None:
            div_yield = df['dividendYield'].fillna(0)
            yield_cons = [LinearConstraint(div_yield, self.target*max(0, target_yield-0.01), self.target*(target_yield+0.01))] # error for dividends is 1%

        # passive constraint
        passive_cons = []
        target_passive = self.profile.passive
        if 0 < target_passive and target_passive < 1:
            passive = np.array(df['isEtf']==True).astype(int)
            passive_cons = [LinearConstraint(passive, self.target*max(0, passive-self.error), self.target*min(1, passive+self.error))]

        # domestic constraint
        domestic_cons = [] # TO DO

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
        sector_allocation_map = self.get_sector_allocation()
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
            passive_cons +
            sector_cons +
            domestic_cons
            # locked_cons
        )

    def get_additional_factors(self, df: pd.DataFrame) -> list:
        additional_factors = []
        # featured stocks
        # additional_factors.append(self.bias * np.array(df['tags'].apply(lambda x: 'Featured' in x)))

        # user preferences
        if self.profile.preferences is not None:
            for sector, preference in self.profile.preferences.items():
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
        return -self.inv_utility_function(a, self.wp, self.get_additional_factors(wp), include_penalty=False)