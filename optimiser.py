from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from params import OBJECTIVE_MAP
from helpers import get_portfolio_value, get_riskfree_rate, merge_portfolio_with_universe
from schemas import Profile

DEFAULT_PROFILE = Profile(
    userId="",
    objective="RETIREMENT",
    international=30,
    passive=30,
    preferences={},
)

class Optimiser:
    portfolio: pd.DataFrame # symbol, units, cost, for each stock held by user
    profile: Profile
    objective: str
    target: float # target portfolio value
    size: int = 50 # target size of portfolio
    error: float = 0.03 # margin for error when using target weights or amounts
    bias: float = 0.05 # bias placed on stocks based on preferences
    threshold: float = 0.05 # minimum weight for stock to be included in optimal portfolio
    formula: str = 'treynor' # formula used for utility function

    def __init__(
        self,
        portfolio: list|pd.DataFrame,
        profile: Profile|None,
        target: float|None = 0,
        size: int = 50,
        error: float = 0.03,
        bias: float = 0.05,
        threshold: float = 0.05,
        formula: str = 'treynor'
    ):
        self.portfolio = portfolio if type(portfolio) == pd.DataFrame else pd.DataFrame.from_records(portfolio)
        self.profile = profile if profile else DEFAULT_PROFILE
        # set target value
        self.target = target if target else get_portfolio_value(portfolio)

        self.size = size
        self.error = error
        self.bias = bias
        self.threshold = threshold

        self.formula = formula

    def inv_utility_function(
            self,
            a: np.ndarray|pd.Series,
            df: pd.DataFrame,
            additional_factors: List[np.ndarray] = [],
            a0: np.ndarray|pd.Series|None = None,
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
                r_p = np.dot(a, df["expReturn"].fillna(0) + np.array(additional_factors).sum(axis=0))
                Beta_p = np.dot(a, df["beta"].fillna(1))
                # prevent divide by zero by adding small amount to beta
                U = (r_p - r_f) / (Beta_p if Beta_p != 0 else 0.01)

            case 'sharpe':
                # Sharpe ratio = (r_p - r_f) / Sigma_p, see https://www.investopedia.com/terms/t/treynorratio.asp
                # requires stock volatilities
                raise NotImplementedError()

        penalty = 0
        if include_penalty:
            # discourage weights between 0 and 5% of the portfolio
            penalty = np.dot(a, np.logical_and(0 < a, a < self.target * self.threshold).astype(int))
            if a0 is not None:
                penalty = np.dot(a, a - a0 != 0)
            # multiplication by 2 represents consideration for fees incurred both by a 'buy' and 'sell' transaction
            penalty = 2 * penalty / self.target
        return -(U-penalty)
    
    def apply_filters(self, df: pd.DataFrame):
        """
        Filters working portfolio based on investment style.
        """
        # handle edge cases
        if self.profile.passive == 1:
            # portfolio is all ETFs, direct equities can be removed.
            return df.drop(df[df['isEtf']==True].index)

        initial_size = len(df)

        # calculate PEG ratios
        # def getPeg(row):
        #     if row["pe"] and row["epsGrowth"]:
        #         if row["epsGrowth"] > 0:
        #             return row["pe"] / row["epsGrowth"]
        #     # return a large number
        #     return 1000
        # df["PEG"] = df.apply(getPeg, axis=1)

        # drop stocks until size of df is 50
        to_keep = np.array([])
        for isEtf, _ in df.groupby("isEtf"):
            if isEtf:
                # TO DO
                pass
            else:
                for sector, group in df.groupby("sector"):
                    if sector not in OBJECTIVE_MAP[self.profile.objective]["sector_allocations"]:
                        continue

                    # get target number of stocks for this sector based on target sector allocation
                    num = np.ceil(self.size * OBJECTIVE_MAP[self.profile.objective]["sector_allocations"][sector])
                    q = 0 # increase this until target size is met
                    sub = group.copy()
                    while len(sub) > num and q < 1:
                        # drop stocks that are outside quantiles for beta
                        lower_beta = group["beta"].quantile(OBJECTIVE_MAP[self.profile.objective]["beta_quantiles"][0])
                        upper_beta = group["beta"].quantile(OBJECTIVE_MAP[self.profile.objective]["beta_quantiles"][1])
                        # get quantile for expected return
                        lower_exp_return = group["expReturn"].quantile(q)                # drop stocks are in top 50% for PEG
                        # upper_peg = group["PEG"].median()

                        sub = group[
                            (group["units"] > 0) | # keep stocks that are already in the portfolio
                            (
                                (group["expReturn"] > lower_exp_return) &
                                # (df["PEG"] < upper_peg ) &
                                ((group["beta"] > lower_beta) & (group["beta"] < upper_beta))
                            )
                        ]
                        q += 0.1

                    to_keep = np.concatenate((to_keep, sub.index.values))

        df = df.loc[to_keep]

        print(f"Dropped {initial_size - len(df)} rows from df")
        # must updated a0 to be same shape as new df
        self.a0 = df["units"] * df["previousClose"]
        return df

    def get_sector_allocation(self):
        """
        Returns target sector allocation for optimisation based on objective and preferences if any.
        """
        targets = OBJECTIVE_MAP[self.profile.objective]["sector_allocations"]

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
        target_yield = OBJECTIVE_MAP[self.profile.objective]["target_yield"]
        if target_yield is not None:
            div_yield = df['dividendYield'].fillna(0)
            yield_cons = [LinearConstraint(div_yield, self.target*max(0, target_yield-0.01), self.target*(target_yield+0.01))] # error for dividends is 1%

        # passive constraint
        passive_cons = []
        # target_passive = self.profile.passive if self.profile else 0.3
        # if 0 < target_passive and target_passive < 1:
        #     passive = np.array(df['isEtf']==True).astype(int)
        #     passive_cons = [LinearConstraint(passive, self.target*max(0, passive-self.error), self.target*min(1, passive+self.error))]

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

    def get_optimal_portfolio(self):
        # get a working copy of the portfolio
        df = merge_portfolio_with_universe(self.portfolio)

        # apply filters
        df = self.apply_filters(df)

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

        # define inital amounts for penalty calculation
        a0 = df['units'].fillna(0) * df['previousClose']

        # first guess for minimiser is equal weight
        equal_weight = self.target * np.ones(len(df)) / len(df)

        # SLSQP appears to perform the best
        a = minimize(self.inv_utility_function, equal_weight, args=(df, additional_factors, a0),
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
        df = merge_portfolio_with_universe(portfolio)
        # extract amount
        a = df["units"].fillna(0) * df["previousClose"]
        return -self.inv_utility_function(a, df, self.get_additional_factors(df), include_penalty=False)