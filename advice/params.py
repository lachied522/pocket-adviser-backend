"""
Parameters for optimisation model based user profile
"""

OBJECTIVE_MAP = {
    "RETIREMENT": {
        "name": "long-term savings",
        "description": "accumulate capital over the long-term",
        "target_beta": 1.25,
        "beta_quantiles": [0.25, 1.0], # quantile of beta for stocks
        "target_yield": 0.02,
        "sector_allocations": {
            "Basic Materials": 0.0191,
            "Financial Services": 0.0747,
            "Healthcare": 0.1740,
            "Energy": 0.0698,
            "Consumer Cyclical": 0.1046,
            "Communication Services": 0.0780,
            "Industrials": 0.0511,
            "Consumer Defensive": 0.0646,
            "Real Estate": 0.0084,
            "Technology": 0.3485,
            "Utilities": 0.005
            # https://www.ishares.com/us/products/239725/ishares-sp-500-growth-etf
        }
    },
    "INCOME": {
        "name": "passive income",
        "description": "earn passive income",
        "target_beta": 0.75,
        "beta_quantiles": [0.10, 0.75],
        "target_yield": 0.05,
        "sector_allocations": {
            "Basic Materials": 0.0754,
            "Financial Services": 0.2565,
            "Healthcare": 0.0097,
            "Energy": 0.0505,
            "Consumer Cyclical": 0.0635,
            "Communication Services": 0.0530,
            "Industrials": 0.1240,
            "Consumer Defensive": 0.0611,
            "Real Estate": 0.1543,
            "Technology": 0.0395,
            "Utilities": 0.1124
            # https://www.ssga.com/au/en_gb/intermediary/etfs/funds/spdr-sp-global-dividend-fund-wdiv
        }
    },
    "PRESERVATION": {
        "name": "capital preservation",
        "description": "protect against downside risk",
        "target_beta": 0.50,
        "beta_quantiles": [0, 0.75],
        "target_yield": 0.03,
        "sector_allocations": {
            "Basic Materials": 0.0032,
            "Financial Services": 0.1160,
            "Healthcare": 0.1430,
            "Energy": 0.025,
            "Consumer Cyclical": 0.130,
            "Communication Services": 0.093,
            "Industrials": 0.110,
            "Consumer Defensive": 0.122,
            "Real Estate": 0.009,
            "Technology": 0.140,
            "Utilities": 0.08
            # https://www.vanguard.com.au/personal/invest-with-us/etf?portId=8201
        }
    },
    "DEPOSIT": {
        "name": "upcoming expense",
        "description": "save for an upcoming expense",
        "target_beta": 1.1,
        "beta_quantiles": [0.25, 0.9],
        "target_yield": 0.01,
        "sector_allocations": {
            "Basic Materials": 0.0242,
            "Financial Services": 0.1296,
            "Healthcare": 0.1337,
            "Energy": 0.0457,
            "Consumer Cyclical": 0.1067,
            "Communication Services": 0.0877,
            "Industrials": 0.0826,
            "Consumer Defensive": 0.0666,
            "Real Estate": 0.0240,
            "Technology": 0.2708,
            "Utilities": 0.0257
            # https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf
        }
    },
    "CHILDREN": {
        "name": "children",
        "description": "provide for children in 10-20 year's time",
        "target_beta": 1.25,
        "beta_quantiles": [0.25, 0.9],
        "target_yield": 0.02,
        "sector_allocations": {
            "Basic Materials": 0.0157,
            "Financial Services": 0.1000,
            "Healthcare": 0.1284,
            "Energy": 0.0593,
            "Consumer Cyclical": 0.1081,
            "Communication Services": 0.0868,
            "Industrials": 0.0447,
            "Consumer Defensive": 0.0950,
            "Real Estate": 0.0038,
            "Technology": 0.3494,
            "Utilities": 0.0062
            # https://www.ishares.com/us/products/239737/ishares-global-100-etf
        }
    },
    "TRADING": {
        "name": "trading",
        "description": "profit in the short-term",
        "target_beta": 1.5,
        "beta_quantiles": [0.0, 1.0],
        "target_yield": None, # no target div yield for trading
        "sector_allocations": None # no target allocations for trading
    }
}