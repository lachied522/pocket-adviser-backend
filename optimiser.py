from typing import List

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

class Optimser:
    portfolio: pd.DataFrame # symbol, units, cost, for each stock held by user
    universe: pd.DataFrame # stock data for all available stocks in universe
    wp: pd.DataFrame # combined universe and portfolio were units is zero for stocks not held
    target: float # target portfolio value
    delta_value: float # target change in portfolio value
    threshold: float # minimum weight for stock to be included in optimal portfolio
    error: float
    
    def __init__(
        self,
        portfolio: pd.DataFrame,
        universe: pd.DataFrame,
        delta_value: float = 0, # target change in portfolio value
        threshold: float = 0.05,
        error: float = 0.03
    ):
        self.portfolio = portfolio
        self.universe = universe
        # merge portfolio and universe to get working portfolio
        wp = pd.merge(universe, portfolio[['symbol', 'units']], on='symbol', how='left')
        # fill 'units' column with zero
        wp['units'] = wp['units'].fillna(0)
        self.wp = wp
        # set target value for portfolio as current portfolio value plus delta
        self.delta_value = delta_value
        self.target = (wp['units'] * wp['previousClose']).sum() + delta_value

        self.threshold = threshold
        self.error = error
    
    def inv_utility_function(self, a, exp_returns, betas, r, a0, additional_factors: List[np.array] = []):
        """
        Inverse utility function for optimisation. Utility is modified Treynor Ratio with penalties.
        """
        U = (np.dot(a, exp_returns + np.array(additional_factors).sum(axis=0)) - r) / np.dot(a, betas)
        # penalty is applied to discourage weights between 0 and 5% of the portfolio and deviations from current values
        # multiplication by 2 represents consideration for fees incurred both by a 'buy' and 'sell' transaction
        # this doesn't appear to work very well as weights close 0 and 5% are generated frequently
        # and deviations from current values are common
        penalty = 2 * (np.dot(a, np.logical_and(0 < a, a < self.target * self.threshold).astype(int)) + np.dot(a, a - a0 != 0)) / self.target
        return -(U-penalty)
    
    def get_constraints(self, df: pd.DataFrame):
        """
        Linear constraints of lb < A.x < ub. Returns tuple.
        """
        # weights must sum to one, within error
        lb, ub = self.target * (1-self.error), self.target * (1+self.error)
        sum_cons = [LinearConstraint(np.ones(len(df)), lb, ub)]

        return tuple(
            sum_cons
        )

    def get_optimal_portfolio(self):
        r, MRP = 0.05, 0.08 # TO DO

        # define optimal portfolio
        op = self.wp.copy()

        # get constraints
        cons = self.get_constraints(op)

        # get additional factors
        additional_factors = []

        # define boundaries
        # lower bound equal to zero (no shorts)
        lb = np.zeros(len(op))
        # upper bound equal to total value of portfolio
        ub = self.target * np.ones(len(op))
        # set keep_feasible True to ensure iterations remain within bounds
        bnds = Bounds(lb, ub, keep_feasible=True)

        # first guess is equal weight
        equal_weight = self.target * np.ones(len(op)) / len(op)

        # utility function arguments
        a0 = op['units'] * op['previousClose']
        betas = op['beta']
        exp_returns = op.apply(lambda x: x['priceTarget'] / x['previousClose'] - 1, axis=1)

        # SLSQP appears to perform the best
        a = minimize(self.inv_utility_function, equal_weight, args=(exp_returns, betas, r, a0, additional_factors),
                    method='SLSQP', bounds=bnds, constraints=cons,
                    options={'maxiter': 100}).x

        # update units column of optimal portfolio and return
        op['units'] = np.round(a / op['previousClose'])
        # drop zero unit rows and return
        self.optimal_portfolio = op.drop(op[op['units'] < 1].index)
        return self.optimal_portfolio

    def get_transactions(
        self,
        n: int = -1 # maximum number of transactions to return
    ):
        """
        Return difference between current portfolio and optimal portfolio as recommended transactions.
        """
        if not hasattr(self, "optimal_portfolio"):
            self.get_optimal_portfolio()

        df = pd.merge(
            self.portfolio[["symbol", "units"]],
            self.optimal_portfolio[["symbol", "units", "name", "previousClose"]],
            how="outer",
            on="symbol",
            suffixes=("_current", "_optimal")
        ).fillna(0)

        # remove any inf values
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # calculate units column
        df["units"] = df['units_optimal'] - df['units_current']

        # drop any rows where units is zero
        df = df.drop(df[df['units']==0].index)

        # sort by absolute difference
        df = df.sort_values('units', key=np.abs, ascending=False)

        # loop through transactions until either target is met or n > N
        value = 0
        transactions = pd.DataFrame(columns=df.columns.to_list()) # create new DF
        for index, row in df.iterrows():
            if abs(value) > abs(self.target) or (n > 0 and len(transactions) >= n):
                break

            if np.sign(row['units']) == np.sign(self.delta_value):
                # if sign of transaction matches target, add to recommended transactions
                v = row['units'] * row['previousClose']
                # check if transaction will reach target
                if abs(value + v) <= abs(self.target):
                    transactions.loc[index,:] = row
                    value += v

                else:
                    # add partial transaction
                    partial_row = row.copy()
                    # round units up / down to nearest integer
                    partial_row['units'] = np.copysign(np.ceil((np.abs(self.target - value)) / row['previousClose']), v)
                    transactions.loc[index,:] = partial_row
                    break

        return transactions