Hello!

This is the backend for my Pocket Adviser app, which is live at https://pocketadviser.com.au. See https://github.com/lachied522/pocket-adviser for the frontend.

It uses pandas and scipy to calculate a user's optimal portfolio of stocks based on their objective and preferences (see advice/optimiser.py). This is used to generate a list of recommended transactions by comparing the optimal portfolio with the user's current portfolio (see advice/portfolio.py), as well as determine whether a single stock would make a good investment (advice/single_stock.py). 📈📉

It also handles long-running tasks, such as refreshing stock data in the database (see data/jobs.py).

It uses sqlalchemy for databasing, and is served with Fastapi.

A paid feature that was added later is daily newsletters with commentary on the general market and individual stocks, generated on a per-user basis (see mailing/jobs.py).
