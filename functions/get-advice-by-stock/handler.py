from pydantic import BaseModel

from optimiser import Optimser
from crud import get_stock_by_symbol
from helpers import get_universe, get_portfolio

#
#     body: {
#         symbol: str
#         amount: int
#     }
# }

class EventBody(BaseModel):
    holdings: list
    target: int

def handler(event, context):
    """
    Provides information on whether a transaction of amount (amount) in stock (symbol) is recommended \
    based on current portfolio and investment profile. Negative amount for sells.
    
    The following are considered when making recommendation.
    
    1. Target sector allocations.
    2. Utility of portfolio before and after proposed transaction.
    """
    print("event", event)
    print("context", context)

    try:
        symbol = event["body"]["symbol"] # target funds to raise, defaults to zero
        amount = event["body"]["amount"] # array of user's holdings

        # check that symbol is valid
        stock = get_stock_by_symbol(symbol)
        if not stock:
            return {
                "message": f"{symbol} was not found",
                "statusCode": 400
            }

        # fetch data
        current_portfolio, universe = get_portfolio(), get_universe()

        # initialise optimiser
        optimiser = Optimser(current_portfolio, universe)

        # get initial adjusted utility
        initial_adj_utility = optimiser.get_utility(current_portfolio)
        
        # get adjusted utility after proposed transaction
        units = round(amount / stock["previousClose"])
        final_portfolio = current_portfolio.copy()
        final_portfolio.set_index('stockId', inplace=True)
        if stock.id in current_portfolio.index:
            # update existing row
            final_portfolio.loc[stock.id, 'units'] = max(current_portfolio.loc[stock.id]['units'] + units, 0)
        else:
            # insert new row
            final_portfolio.loc[stock.id] = { "units": units }

        final_adj_utility = optimiser.get_utility()
           
        return {
            "body": {
                "initial_adj_utility": initial_adj_utility,
                "final_adj_utility": final_adj_utility
            },
            "statusCode": 200
        }
    
    except KeyError as key:
        # indicates event body is missing a parameter
        return {
            "message": f"{key} is required",
            "statusCode": 400
        }
    
    except Exception as e:
        # any other error
        print(e)
        return { "statusCode": 500 }