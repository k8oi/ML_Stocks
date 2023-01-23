import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#setting up data
ticker = "nke"
stock = yf.Ticker(ticker)
stock_hist = stock.history(period="max")

#moving data to find out difference in prices between two days
stock_prev = stock_hist.copy()
stock_prev = stock_prev.shift(1)

###finding actual close
data = stock_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

##setup out target
data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

###join the data
predict = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(stock_prev[predict]).iloc[1:]

##create a model
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

#create a train and test set
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predict], train["Target"])

#error of predicitons
preds = model.predict(test[predict])
preds = pd.Series(preds, index=test.index)
ps = precision_score(test["Target"], preds)
print("BASIC TRAIN SUCCESS %: ", ps)

#combine predicitons and test values
combined = pd.concat({"Target": test["Target"],"Predicitions": preds}, axis=1)

################################
#backtesting
i = 1000
step = 750

predictions = []
prescore = []

def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)
    
predictions = backtest(data, model, predict)


####value counts
test_predictions = predictions["Predictions"].value_counts()
real_changes = predictions["Target"].value_counts()
ps = precision_score(predictions["Target"], predictions["Predictions"])
print("Estimates: ", test_predictions, "Real Data: ", real_changes, "Back Tested Prediction Estimator: ", ps)