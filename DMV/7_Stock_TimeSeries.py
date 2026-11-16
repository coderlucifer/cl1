"""
Assignment 7 : Time Series Data Analysis - Stock market prices (trend, seasonality, ARIMA forecast)
===================================================================================================
INSTRUCTIONS
 1. Install:  pip install pandas numpy matplotlib statsmodels
 2. Dataset : Stock_Prices.csv with columns Date, Adj_Close, Close, High, Low, Open, Volume (any stock
              from Kaggle / Yahoo Finance works). Option A (default): fetch it online from the
              datasets repo. Option B: load a local copy kept next to this script (comment out
              Option A and uncomment the Option B line).
 3. Run     : python 7_Stock_TimeSeries.py
 4. Steps   : load -> explore -> Date to datetime index -> line plot -> moving averages
              -> seasonality (day of week, month, decomposition) -> correlation with volume
              -> stationarity test (ADF) -> ARIMA forecast -> exponential smoothing forecast
              -> compare both models with RMSE / MAE
"""

# --- Step 0: Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Step 1: Import the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/Stock_Prices.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("Stock_Prices.csv")

# --- Step 2: Explore the dataset ---
# Size, column types, first rows, summary stats and missing cells.
print("Shape:", df.shape)
df.info()
print(df.head())
print(df.describe().round(2))
print("Missing values:", int(df.isnull().sum().sum()))

# --- Step 3: Make sure Date is a proper datetime and use it as the index ---
# Real dates, oldest first, as the index; gaps take the previous day's value.
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
df = df.ffill()
print("\nDate range:", df.index.min().date(), "to", df.index.max().date(), "| rows:", len(df))

# --- Step 4: Line chart of historical prices ---
# Two stacked charts sharing the x-axis: closing price on top, volume below.
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(df.index, df["Close"], color="navy")
ax[0].set(title="Closing price", ylabel="Price")
ax[1].bar(df.index, df["Volume"], color="gray", width=1.0)
ax[1].set(title="Volume", ylabel="Shares", xlabel="Date")
for a in ax:
    a.grid(True, ls=":", alpha=0.6)
plt.tight_layout(); plt.show()

# --- Step 5: Moving (rolling) averages to smooth noise and show the trend ---
# 20- and 50-day averages; the EMA gives recent days more weight.
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
# Line chart: price with its three moving averages.
plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Close", alpha=0.5)
plt.plot(df["MA20"], label="20-day MA", lw=2)
plt.plot(df["MA50"], label="50-day MA", lw=2)
plt.plot(df["EMA20"], label="20-day EMA", lw=1.5, ls="--")
plt.title("Moving averages"); plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.grid(True, ls=":", alpha=0.6)
plt.tight_layout(); plt.show()
# A short MA crossing above the long MA is a bullish signal (golden cross).
# Count days where MA20 crosses above MA50 (today above, yesterday not).
cross_up = ((df["MA20"] > df["MA50"]) & (df["MA20"].shift(1) <= df["MA50"].shift(1))).sum()
print("Number of 20/50-day upward crossovers:", int(cross_up))

# --- Step 6: Seasonality analysis (weekly, monthly, yearly patterns) ---
# Daily % return, averaged by weekday and by month.
df["Return"] = df["Close"].pct_change() * 100
by_day = df.groupby(df.index.dayofweek)["Return"].mean()
by_day.index = ["Mon", "Tue", "Wed", "Thu", "Fri"][:len(by_day)]
by_month = df.groupby(df.index.month)["Return"].mean()
print("\nAverage return (%) by weekday:\n", by_day.round(3))
print("\nAverage return (%) by month:\n", by_month.round(3))

# Split Close into trend + seasonal + residual; period=5 is one trading week.
dec = seasonal_decompose(df["Close"], model="additive", period=5)
# Bar charts of weekday and monthly patterns, plus the repeating weekly component.
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
by_day.plot(kind="bar", ax=ax[0], color="teal", title="Average return by weekday (%)", rot=0)
by_month.plot(kind="bar", ax=ax[1], color="orange", title="Average return by month (%)", rot=0)
dec.seasonal.iloc[:30].plot(ax=ax[2], title="Seasonal component (first 30 days)")
plt.tight_layout(); plt.show()
# Full decomposition: observed, trend, seasonal and residual panels.
fig = dec.plot(); fig.set_size_inches(10, 7); plt.tight_layout(); plt.show()

# --- Step 7: Correlation between price and other variables ---
# Pearson correlation between every pair of columns.
cols = ["Open", "High", "Low", "Close", "Volume", "Return"]
corr = df[cols].corr()
print("\nCorrelation matrix:\n", corr.round(3))
# Big moves (up or down) usually come with high volume, hence abs().
print("Close vs Volume:", round(df["Close"].corr(df["Volume"]), 3),
      "| |Return| vs Volume:", round(df["Return"].abs().corr(df["Volume"]), 3))
# Heatmap of the matrix: red = positive, blue = negative.
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(cols)), cols, rotation=45); plt.yticks(range(len(cols)), cols)
plt.title("Correlation heatmap"); plt.tight_layout(); plt.show()

# --- Step 8: Stationarity test (ARIMA needs a stationary series) ---
# ADF test: p-value < 0.05 means stationary. Differencing removes the trend.
p_raw = adfuller(df["Close"].dropna())[1]
p_diff = adfuller(df["Close"].diff().dropna())[1]
print(f"\nADF p-value of price: {p_raw:.4f} -> {'stationary' if p_raw < 0.05 else 'NOT stationary'}")
print(f"ADF p-value of 1st difference: {p_diff:.4f} -> {'stationary' if p_diff < 0.05 else 'NOT stationary'}")
# So we need d = 1 (difference once) in ARIMA(p, d, q).

# --- Step 9: Train / test split (keep the LAST part as test, never shuffle time series) ---
# Test on the last h days, train on the rest; business-daily frequency is needed by statsmodels.
h = 30
train, test = df["Close"].iloc[:-h], df["Close"].iloc[-h:]
train = train.asfreq("B").ffill()

# --- Step 10: ARIMA model ---
# ARIMA(p, d, q): p = past values used (AR), d = times differenced, q = past errors used (MA).
# Fit on the training data and forecast the test period.
arima = ARIMA(train, order=(5, 1, 0)).fit()
print("\n", arima.summary().tables[1])
arima_fc = arima.forecast(steps=h)
arima_fc.index = test.index

# --- Step 11: Exponential smoothing model (Holt's linear trend) ---
# Straight-line trend (trend="add"); recent days weigh more than old ones.
hw = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
hw_fc = hw.forecast(h)
hw_fc.index = test.index

# --- Step 12: Compare the models ---
def scores(actual, pred):
    # RMSE, MAE and MAPE of a forecast against the actual values.
    err = actual - pred
    return np.sqrt((err ** 2).mean()), err.abs().mean(), (err.abs() / actual).mean() * 100

res = pd.DataFrame({"ARIMA(5,1,0)": scores(test, arima_fc), "Exp. smoothing": scores(test, hw_fc)},
                   index=["RMSE", "MAE", "MAPE %"]).T.round(3)
print("\nForecast accuracy on the last", h, "days:\n", res)

# Line chart: last 90 training days, actual test prices and both forecasts.
plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-90:], label="Train (last 90 days)")
plt.plot(test, label="Actual", color="black", lw=2)
plt.plot(arima_fc, label="ARIMA forecast", color="red", ls="--")
plt.plot(hw_fc, label="Exp. smoothing forecast", color="green", ls=":")
plt.title("Stock price forecast vs actual"); plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.grid(True, ls=":", alpha=0.6)
plt.tight_layout(); plt.show()

# --- Step 13: Forecast the real future (refit on ALL data) ---
# Refit on every day and forecast ahead with a 95% confidence range.
final = ARIMA(df["Close"].asfreq("B").ffill(), order=(5, 1, 0)).fit()
future = final.get_forecast(steps=h)
ci = future.conf_int()
print("\nNext", h, "business days (first 5):\n", future.predicted_mean.head().round(2))
print("Day", h, "forecast: %.2f (95%% range %.2f to %.2f)" % (future.predicted_mean.iloc[-1], ci.iloc[-1, 0], ci.iloc[-1, 1]))

"""
=====================================================================
CODE EXPLANATION
=====================================================================
1. Imports: pandas / numpy for data, matplotlib for charts, and statsmodels for the time
   series tools (seasonal_decompose, adfuller, ARIMA, ExponentialSmoothing).
2. Load: read Stock_Prices.csv (Date, Adj_Close, Close, High, Low, Open, Volume) from the
   datasets repo with pd.read_csv(url) (Option A, needs internet). For Option B, comment out
   Option A and uncomment pd.read_csv("Stock_Prices.csv") to load a local copy.
3. Explore: shape, info(), head(), describe() and a missing-value count.
4. Datetime index: pd.to_datetime converts the Date text, set_index(Date) makes it the row
   label, sort_values puts it in time order and ffill fills gaps.
5. Line chart: closing price and volume against time.
6. Moving averages: rolling(20/50).mean() and ewm(span=20) smooth the noise. We also count
   the days when the 20-day line crosses above the 50-day line.
7. Seasonality: average daily return by weekday, average daily return by month and
   seasonal_decompose(period=5) which splits the series into trend, seasonal and residual.
8. Correlation: corr() between prices, volume and returns, shown as a heat-map.
9. Stationarity: the ADF test on the price and on its first difference. The price is not
   stationary, the difference is, so ARIMA uses d = 1.
10. Split: last 30 business days are the test set, the earlier days are the training set
    (time series are split by time, never shuffled).
11. ARIMA(5,1,0) is fitted on the training data and forecasts 30 days.
12. Exponential smoothing (Holt linear trend) forecasts the same 30 days.
13. Compare: RMSE, MAE and MAPE of both models against the real last 30 days, then plot.
14. Final forecast: ARIMA is refitted on all data to predict the next 30 business days with
    a 95% confidence range.

=====================================================================
EXPECTED OUTPUT (real run on the default file: real MSFT daily prices, 1986-2026)
=====================================================================
 - Shape (10238, 7). Price rises strongly over 40 years; MA20 hugs the price and MA50 is
   smoother. 114 upward 20/50-day crossovers.
 - Average return (%) by weekday: Mon 0.087, Tue 0.113, Wed 0.208, Thu 0.085, Fri 0.034.
 - Average return (%) by month: Jan 0.229, Feb -0.046, Mar 0.138, Apr 0.164, May 0.138,
   Jun 0.081, Jul 0.024, Aug 0.072, Sep 0.039, Oct 0.227, Nov 0.118, Dec 0.073.
   (Differences are small and noisy; they are not a trading rule.)
 - Correlation: Open/High/Low/Close = 1.0; Close vs Volume -0.37 (old years had higher volume
   and lower prices); |Return| vs Volume 0.494.
 - ADF on price: p = 0.999 (not stationary). ADF on first difference: p = 0.0000
   (stationary). This justifies d = 1.
 - Last-30-day test: ARIMA(5,1,0) RMSE 31.90 / MAE 29.24 / MAPE 5.53%;
   exponential smoothing RMSE 30.02 / MAE 27.50 / MAPE 5.20% (slightly better here; the winner
   on other data may differ). Day-30 future forecast 522.05 with 95% range 495.77 to 548.34.
 - ARIMA and exponential smoothing give almost flat/straight forecasts. This is normal:
   a stock behaves like a random walk, so the best forecast is "about today's price" plus drift.
   The confidence band widens the further ahead we look.
 - Exact numbers change with your own Stock_Prices.csv. Judge the models by RMSE/MAE (lower is better).

=====================================================================
DETAILED THEORY NOTES
=====================================================================
1. WHAT IS A TIME SERIES?
   - A sequence of values recorded at regular time intervals (daily stock close, hourly
     temperature, monthly sales). ORDER MATTERS: today depends on yesterday, so rows cannot be
     shuffled or randomly split like normal ML data.
   - Time series analysis = understand the pattern (descriptive). Forecasting = predict the future.

2. COMPONENTS OF A TIME SERIES
   - Trend: long-term upward or downward movement.
   - Seasonality: a pattern that repeats at a fixed period (weekday effect, December sales).
   - Cyclic: rises and falls with no fixed period (business cycles).
   - Noise / residual: random leftover that cannot be explained.
   - Additive model: value = trend + seasonal + residual (used here).
     Multiplicative model: value = trend x seasonal x residual (when swings grow with the level).

3. STOCK DATA COLUMNS
   - Open: first trade price of the day. Close: last trade price (most used).
   - High / Low: highest / lowest price of the day. Volume: number of shares traded.
   - Return = (today - yesterday) / yesterday. Analysts model returns because they are more
     stable than raw prices.

4. MOVING AVERAGE (ROLLING MEAN)
   - Average of the last n values; it moves forward one step at a time. It smooths short-term
     noise so the trend is visible. Larger window = smoother but slower to react.
   - Simple MA gives every day equal weight. Exponential MA (EMA) gives recent days more weight.
   - Golden cross: short MA crosses above long MA (bullish). Death cross: crosses below (bearish).

5. STATIONARITY
   - A series is stationary if its mean, variance and autocorrelation do not change over time.
     Stock prices are NOT stationary (they trend). Most classical models (ARIMA) need stationarity.
   - Differencing (today - yesterday) removes the trend and usually makes the series stationary.
   - ADF test (Augmented Dickey-Fuller): H0 = not stationary. If p-value < 0.05 reject H0, so the
     series is stationary.

6. ARIMA(p, d, q)
   - AR (AutoRegressive, p): predicts today from the last p values.
   - I (Integrated, d): number of times we difference the data to make it stationary.
   - MA (Moving Average, q): predicts today from the last q forecast errors.
   - ARIMA(5,1,0) = use the last 5 values, difference once, no MA part.
   - SARIMA adds seasonal terms (P, D, Q, m). ARIMAX adds outside variables (e.g. volume).
   - Choosing p and q: look at ACF / PACF plots or compare AIC (lower is better).

7. EXPONENTIAL SMOOTHING
   - Forecast = weighted average of past values with weights that decay exponentially.
   - Simple: level only. Holt: level + trend (used here). Holt-Winters: level + trend + seasonality.
   - Smoothing parameter alpha near 1 = trust the latest value, near 0 = trust the long history.

8. EVALUATION METRICS
   - MAE = average absolute error (same unit as price, easy to read).
   - RMSE = square root of the average squared error; punishes big errors more.
   - MAPE = average absolute error as a percentage of the actual value.
   - Lower is better for all three. Always evaluate on a hold-out period at the END of the data.

9. CORRELATION
   - Pearson r from -1 to +1. Open, High, Low and Close are almost perfectly correlated (r near 1).
     Price vs volume is usually weak, but absolute return vs volume is positive: big moves
     come with heavy trading.
   - Correlation is not causation.

10. LIMITS
   - Markets are close to a random walk, so long-range accurate forecasts are not possible.
   - Models use only past prices; news, earnings and policy changes are not captured.
   - Never use this for real trading decisions; it is a learning exercise.

=====================================================================
LIBRARIES AND FUNCTIONS USED
=====================================================================
 - numpy: sqrt for the RMSE metric.
 - pandas: read_csv, to_datetime, set_index, sort_values, ffill, asfreq,
           rolling().mean(), ewm().mean(), pct_change, diff, shift, groupby, corr, describe, info.
 - matplotlib.pyplot: subplots, plot, bar, imshow, colorbar, legend, grid, tight_layout, show.
 - statsmodels seasonal_decompose: split series into trend / seasonal / residual.
 - statsmodels adfuller: stationarity test, returns the p-value at index 1.
 - statsmodels ARIMA(...).fit(), forecast(), get_forecast(), conf_int(): the ARIMA model.
 - statsmodels ExponentialSmoothing(...).fit().forecast(): Holt smoothing model.

=====================================================================
DETAILED VIVA Q&A
=====================================================================
Q1. What is a time series and why is it different from normal data?
A1. It is data recorded in time order. Each value depends on earlier ones, so rows cannot be
    shuffled, and train/test splits must keep time order (train = past, test = future).

Q2. What are the components of a time series?
A2. Trend, seasonality, cyclic variation and random noise (residual).

Q3. What is stationarity and why does ARIMA need it?
A3. Stationary means constant mean, variance and autocorrelation over time. ARIMA's AR and MA
    parts assume stable statistical behaviour, otherwise the coefficients mean nothing.

Q4. How do you make a series stationary?
A4. Differencing (today - yesterday), taking the log first to stabilise variance, or removing the
    trend / seasonality by decomposition.

Q5. Explain the ADF test.
A5. Null hypothesis: the series has a unit root (not stationary). A p-value below 0.05 rejects it,
    so the series is stationary.

Q6. What do p, d and q mean in ARIMA?
A6. p = number of lag values used, d = number of differences taken, q = number of lagged
    forecast errors used.

Q7. How do you choose p and q?
A7. Read the PACF plot for p and the ACF plot for q, or fit several combinations and pick the one
    with the lowest AIC / BIC.

Q8. ARIMA vs exponential smoothing?
A8. ARIMA models the correlation between values and needs stationarity. Exponential smoothing
    models level/trend/seasonality with decaying weights. Smoothing is simpler; ARIMA is more flexible.

Q9. What is a moving average used for?
A9. Smoothing noise to see the trend, and giving buy/sell signals when a short average crosses a
    long one.

Q10. What is seasonality? Give a stock example.
A10. A pattern that repeats at fixed intervals, e.g. weekday effects (Monday returns) or
     year-end rallies.

Q11. Why are stock forecasts usually flat?
A11. Prices behave like a random walk: the best guess for tomorrow is today's price plus a small
     drift, so the forecast has no strong shape and the uncertainty band just widens.

Q12. Additive vs multiplicative decomposition?
A12. Additive: components are summed (constant swings). Multiplicative: components multiply
     (swings grow with the level).

=====================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=====================================================================
Q0. How do you switch between online and local data?
A0. Option A (active) reads the CSV from the datasets repo URL with pd.read_csv(url). For Option B, comment out
    Option A and uncomment pd.read_csv("Stock_Prices.csv"), which reads a local copy next to the script.

Q1. Why did we use set_index("Date") after to_datetime?
A1. A datetime index lets pandas understand time: dayofweek / month, resampling, asfreq and
    plotting against dates all depend on it, and statsmodels needs it for forecasting.

Q2. Why is the test set the LAST 30 days and not a random 20%?
A2. A random split would let the model see the future while training (data leakage). Real
    forecasting always predicts forward, so we test on the most recent unseen period.

Q3. What does rolling(window=20).mean() return for the first 19 rows?
A3. NaN, because 20 values are needed to compute the first average.

Q4. What does asfreq("B").ffill() do and why?
A4. It forces a regular business-day index and fills missing days (holidays) with the previous
    value. statsmodels needs a regular frequency to forecast properly.

Q5. Why period=5 in seasonal_decompose?
A5. Stocks trade 5 days a week, so the natural repeating cycle is one trading week.

Q6. Why do we difference once (d = 1) in ARIMA(5,1,0)?
A6. The ADF test showed the price is not stationary but its first difference is, so one
    difference is enough.

Q7. How do we judge which forecast is better?
A7. Compare RMSE, MAE and MAPE on the 30 held-out days; the lower value wins.

Q8. What does get_forecast().conf_int() give?
A8. A 95% lower and upper range for each forecast day; it widens with the horizon because
    uncertainty grows.

Q9. Why refit on all data in the last step?
A9. After choosing and testing the model we have no reason to hold data back; using every
    observation gives the best possible forecast of the real future.

Q10. What would you change to capture a yearly seasonal pattern?
A10. Use SARIMA (SARIMAX) with seasonal order (P, D, Q, 252) or Holt-Winters with
     seasonal_periods=252 and a few years of data.

END OF VIVA NOTES
"""
