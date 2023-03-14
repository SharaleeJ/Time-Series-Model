pip install yfinance --upgrade --no-cache-dir
pip install yahooquery


#Import Packages
import numpy as np
import pandas as pd 
import math
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf
from yfinance import Ticker
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error
from yahooquery import Ticker

# Mission Statement 
#foster[ing] an inclusive environment that leverages the diverse backgrounds and perspectives of all our employees, suppliers, customers, and partners to drive a sustainable global competitive advantage.

# Where it Started
#Oracle is a complete suite of integrated cloud applications and a cloud infrastructure platform started by Larry Ellison with Bob Miner & Ed Oates. Orginally named Software Development Laboratories (SDL), Orcale was started with a $2000 investment on June 16 1977 in Santa Clara, Californa and has grown into a multinational corporation with over a billion dollars in profit. 


ORCL = Ticker ('ORCL')
print(ORCL.income_statement())
print(ORCL.balance_sheet())
print(ORCL.cash_flow())
print(ORCL.grading_history)
print(ORCL.recommendation_trend)



ticker = yf.Ticker('ORCL')
#Parameters of Oracle Stock data we would like

prices = ticker.history(start='2019-12-01')[['Close']]
prices = prices.tz_localize(None)
prices.index = pd.DatetimeIndex(prices.index).to_period('B')
train_end_date = '2023-1-30'


prices_train = prices.loc[:train_end_date]
prices_train

#Actual Stock Price

prices_test = prices.loc[train_end_date:][1:]
prices_test.head()

plot_acf(prices_train.pct_change()[1:])
plt.show()

plot_pacf(prices_train.pct_change()[1:])
plt.show()

order = (1,1,1)


# Train and forecast
steps= 10 # length of the period you are trying to predict. 

# Train ARIMA model
model = ARIMA(prices_train.values,
    order=order).fit()

# Forecast the future price values
df_traditional = pd.DataFrame(
    model.forecast(steps=steps),
    index=prices_test[:steps].index
)

pd.concat((prices_train[-50:], prices_test[:steps], df_traditional), axis=1).plot()
plt.legend(['Training set', 'Actual price', 'Multi-step forecast'])
plt.grid()
plt.title('Multi-step forecast')
plt.ylabel('Oracle stock price')
plt.show()

# Initialize predictions array
predictions = []

# Loop through time periods
for step in range(steps): 
    
    # Adds the new prices to the previous ones
    prices_train_i = pd.concat((prices_train, prices_test[:step]), axis=0)
    
    # Train new model
    model_i = ARIMA(prices_train_i.values, order=order).fit()
    
    # Forecast next day price
    pred = model_i.forecast(steps=1)
    
    # Predictions array
    predictions.append(pred)

# Dataframe
df_rolling = pd.DataFrame(predictions, index=prices_test[:steps].index)
df_rolling.head()

#Actual Price
prices_test.head()


pd.concat((prices_train[-50:], prices_test[:steps], df_rolling), axis=1).plot()
plt.legend(['Training set', 'Actual price', 'Rolling forecast'])
plt.grid()
plt.title('Rolling forecast')
plt.ylabel('Oracle stock price')
plt.show()

pd.concat((prices_train[-100:], 
    prices_test[:steps], 
    df_rolling,
    df_traditional,
    ), axis=1).plot()
plt.legend(['Training set', 'Actual price', 'Rolling forecast', 'Multi-step forecast'])
plt.grid()
plt.title('Rolling forecast vs Multi-step forecast')
plt.ylabel('Oracle stock price')
plt.show()

def forecast_accuracy(df_rolling, prices_test):
    mape = np.mean(np.abs(df_rolling - prices_test)/np.abs(prices_test)) # Mean Absolute Percentage Error (MAPE)
    return(mape)
forecast_accuracy(pred, prices_train)

