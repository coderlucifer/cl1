"""
Assignment 2 : Interacting with Web APIs - Weather data from OpenWeatherMap
===========================================================================
INSTRUCTIONS
 1. Install:  pip install pandas numpy matplotlib requests
 2. API key : register free at https://openweathermap.org/api , copy the key, then set it
              (do NOT hard-code or share it):
                 Windows PowerShell :  $env:OPENWEATHER_API_KEY="your_key"
                 Linux / Mac        :  export OPENWEATHER_API_KEY=your_key
              New keys take ~1-2 hours to activate. Without a working key, or offline,
              the script falls back to SYNTHETIC forecast data so the rest still runs.
 3. Run     : python 2_Weather_API.py [CityName]        (default city: Pune)
 4. Steps   : call 5-day/3-hour forecast API -> extract temp, humidity, wind, rain -> clean
              -> daily aggregation -> statistics -> line/bar/scatter plots -> correlation heatmap
              -> simple map of city coordinates
"""
import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# --- Step 1: Configuration ---
# The API key comes from an environment variable, never from the code.
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
# City from the command line, default Pune.
CITY = sys.argv[1] if len(sys.argv) > 1 else "Pune"
# Forecast endpoint: one reading every 3 hours for 5 days.
URL = "https://api.openweathermap.org/data/2.5/forecast"

# --- Step 2: Call the API (with error handling + fallback) ---
data = {}  # stays empty if the call fails
try:
    # timeout=10 stops the script hanging if the server never answers.
    res = requests.get(URL, params={"q": CITY, "units": "metric", "appid": API_KEY}, timeout=10)
    # 200 = ok, 401 = bad key, 404 = city not found, 429 = too many calls.
    print("HTTP status:", res.status_code)
    if res.status_code == 200:
        data = res.json()
except requests.RequestException as e:
    # Offline or timeout: carry on with the fallback data.
    print("Request failed:", e)

# Use live data if it arrived, otherwise build synthetic data.
if data.get("list"):
    items = data["list"]
    lat, lon = data["city"]["coord"]["lat"], data["city"]["coord"]["lon"]
    print(f"Loaded LIVE data for {CITY}: {len(items)} readings")
else:
    print("Using SYNTHETIC sample data (set a valid OPENWEATHER_API_KEY for live data)")
    lat, lon = 18.52, 73.86  # approx. Pune
    start = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    rng = np.random.default_rng(1)  # fixed seed so the fake data repeats
    # 40 fake readings in the same nested shape as the real reply.
    # Temp and humidity use opposite sin waves so they are negatively related.
    # Rain appears only every 7th reading, as the real API only lists rain when it rains.
    items = [{"dt_txt": (start + datetime.timedelta(hours=3 * i)).strftime("%Y-%m-%d %H:%M:%S"),
              "main": {"temp": round(28 + 5 * np.sin(i / 4) + rng.normal(0, .5), 1),
                       "humidity": int(65 - 15 * np.sin(i / 4) + rng.normal(0, 3))},
              "wind": {"speed": round(abs(rng.normal(4, 1.2)), 1)},
              "rain": {"3h": 0.5 + float(rng.random()) if i % 7 == 0 else 0.0}} for i in range(40)]

# --- Step 3: Extract attributes into a DataFrame ---
# Flatten each nested reading into one row; rain is missing on dry days, so default to 0.
df = pd.DataFrame([{"datetime": it["dt_txt"], "temp": it["main"]["temp"],
                    "humidity": it["main"]["humidity"], "wind_speed": it["wind"]["speed"],
                    "rain": it.get("rain", {}).get("3h", 0.0)} for it in items])

# --- Step 4: Clean / pre-process ---
# Convert to real dates (bad text becomes NaT), then drop invalid or repeated times.
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).drop_duplicates("datetime")
# No rain value means no rain (0); other gaps are filled from neighbouring readings.
df["rain"] = df["rain"].fillna(0)
num = ["temp", "humidity", "wind_speed"]
df[num] = df[num].interpolate().bfill()
# Keep just the calendar day, used later to group readings into days.
df["date"] = df["datetime"].dt.date
print("\nFirst rows:\n", df.head())

# --- Step 5: Modelling / statistics + aggregation ---
# Basic temperature stats, then the slope of a straight-line trend.
print("\nAverage temp : %.1f C | Max %.1f | Min %.1f" % (df.temp.mean(), df.temp.max(), df.temp.min()))
slope = np.polyfit(np.arange(len(df)), df["temp"], 1)[0]
print("Temperature trend: %+.3f C per 3-hour step" % slope)
# Collapse the readings into one summary row per day.
daily = df.groupby("date").agg(avg_temp=("temp", "mean"), max_temp=("temp", "max"),
                               min_temp=("temp", "min"), avg_humidity=("humidity", "mean"),
                               avg_wind=("wind_speed", "mean"), total_rain=("rain", "sum")).round(2)
print("\n--- Daily Weather Summary ---\n", daily)
# How strongly each pair of weather variables moves together.
print("\nCorrelation matrix:\n", df[["temp", "humidity", "wind_speed", "rain"]].corr().round(2))

# --- Step 6: Visualise ---
# A 2x3 grid of charts.
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
# Temperature over time.
ax[0, 0].plot(df["datetime"], df["temp"], marker="o", color="tomato")
ax[0, 0].set(title=f"Temperature - {CITY}", ylabel="deg C"); ax[0, 0].tick_params(axis="x", rotation=30)
# Daily rainfall bars.
daily["total_rain"].plot(kind="bar", ax=ax[0, 1], color="steelblue", title="Daily rainfall (mm)")
# Wind speed over time.
ax[0, 2].plot(df["datetime"], df["wind_speed"], color="green")
ax[0, 2].set(title="Wind speed (m/s)"); ax[0, 2].tick_params(axis="x", rotation=30)
# Temp vs humidity, with dot colour showing wind speed.
ax[1, 0].scatter(df["temp"], df["humidity"], c=df["wind_speed"], cmap="viridis", edgecolor="k")
ax[1, 0].set(title="Temp vs Humidity", xlabel="Temp", ylabel="Humidity")
# Correlation heatmap on a fixed -1..+1 scale, with variable names on the axes.
c = df[["temp", "humidity", "wind_speed", "rain"]].corr()
im = ax[1, 1].imshow(c, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_xticks(range(4)); ax[1, 1].set_xticklabels(c.columns, rotation=30)
ax[1, 1].set_yticks(range(4)); ax[1, 1].set_yticklabels(c.columns); ax[1, 1].set_title("Correlation heatmap")
# Basic map: one bubble at lon/lat, coloured by mean temperature.
ax[1, 2].scatter([lon], [lat], s=400, c=[df.temp.mean()], cmap="coolwarm", edgecolor="k")
ax[1, 2].annotate(CITY, (lon, lat), textcoords="offset points", xytext=(10, 10))
ax[1, 2].set(title="Location (bubble = mean temp)", xlabel="Longitude", ylabel="Latitude")
plt.tight_layout(); plt.show()

"""
=======================================================================================
CODE EXPLANATION
=======================================================================================
1. Step 1 (configuration): Read the API key from the environment variable
   OPENWEATHER_API_KEY using os.getenv, the city from sys.argv, and set the endpoint URL.
2. Step 2 (API call): requests.get() sends an HTTP GET with the city, units=metric and the
   key. We print the status code, parse JSON with res.json() if it is 200, and catch
   requests.RequestException (offline, timeout). If no data came back we build 40 synthetic
   3-hourly readings (numpy sin + random noise) with the same nested structure as the API.
3. Step 3 (extract): A list comprehension flattens each nested reading into a row with
   datetime, temp, humidity, wind_speed and rain, using dict.get() for the optional rain.
4. Step 4 (clean): pd.to_datetime(errors="coerce"), dropna(subset), drop_duplicates, fillna(0)
   for rain only, interpolate().bfill() for temp/humidity/wind, and .dt.date prepare a tidy table.
5. Step 5 (statistics): mean/max/min, np.polyfit slope for the temperature trend,
   groupby("date").agg(...) for the daily summary, and DataFrame.corr() for correlations.
6. Step 6 (visualise): 6 matplotlib charts: temperature line, daily rain bars, wind line,
   temp-vs-humidity scatter (coloured by wind), correlation heatmap and a location bubble.

=======================================================================================
DETAILED THEORY NOTES
=======================================================================================
1. What is a Web API?
   - API = Application Programming Interface: a set of rules that lets one program ask
     another program for data. A Web API works over HTTP using URLs.
   - Here our script is the client; OpenWeatherMap's server is the provider.

2. REST
   - REST (Representational State Transfer) is a popular API style. Each resource has a URL
     and we use HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove).
   - REST is stateless: every request carries all the information the server needs.

3. JSON
   - JSON is the text format of most REST replies: objects {key: value} and lists [..].
   - In Python it becomes dictionaries and lists, so we reach nested values like
     item["main"]["temp"].

4. HTTP status codes
   - 200 OK (success). 400 bad request. 401 unauthorized (wrong or not yet active key).
   - 404 not found (wrong city/URL). 429 too many requests (rate limit). 500 server error.
   - We check the code so the program does not try to read data from a failed call.

5. API key
   - A secret string that identifies you to the provider. It controls access and lets the
     provider limit usage (free plan = limited calls per minute).
   - Never write it into code or GitHub; keep it in an environment variable.

6. Query parameters and timeout
   - params={"q": CITY, "units": "metric", "appid": KEY} are added after ? in the URL.
   - timeout=10 stops the program waiting forever if the server does not answer.

7. Dataset description (5-day / 3-hour forecast)
   - One reading every 3 hours for 5 days = 40 records. Fields used: dt_txt (time),
     main.temp (deg C with units=metric), main.humidity (%), wind.speed (m/s),
     rain.3h (mm in last 3 hours; present only when it rains).
   - Synthetic fallback: 40 readings with a smooth temperature wave around 28 C, opposite
     humidity wave, random wind ~4 m/s and a little rain every 7th reading.

8. Data cleaning in this script
   - Convert text time to datetime; drop rows with invalid time and repeated times.
   - Missing rain is filled with 0 (no rain). Missing temp, humidity and wind are filled by
     interpolate() (straight line between neighbours) then bfill(), because 0 would be a
     false reading (a missing temperature is not 0 C).

9. Aggregation
   - Aggregation summarises many records into fewer: mean, max, min, sum.
   - groupby("date").agg() turns 40 three-hourly readings into 5-6 daily summary rows
     (a forecast crossing midnight spans 6 calendar dates).

10. Correlation
   - Pearson correlation between -1 and +1. Negative = one rises when the other falls.
   - Temperature and humidity are normally negatively correlated (hotter air = lower
     relative humidity).

11. Linear trend
   - polyfit(x, y, 1) fits y = slope*x + intercept. A positive slope = warming over the
     forecast window, negative = cooling.

12. Visualisation choices
   - Line for time series, bar for daily totals, scatter for relation of two variables
     (colour adds a third), heatmap for a correlation matrix, bubble plot for location.

13. Advantages / disadvantages of using APIs
   - Advantages: always fresh data, no manual downloads, easy to automate, structured JSON.
   - Disadvantages: need internet and key, rate limits, format may change, free plans give
     limited history, forecast is not a measurement.

14. Real-world uses
   - Weather apps, travel and farming planners, energy demand forecasting, logistics,
     stock/crypto price feeds, maps, payment gateways and social media data collection.

=======================================================================================
LIBRARIES AND FUNCTIONS USED
=======================================================================================
- datetime: datetime.now() current time, timedelta(hours=...) time step, strftime() format text.
- os: getenv() reads the API key from the environment.
- sys: argv holds command line arguments (city name).
- requests: get() sends the HTTP request; status_code, json(); RequestException for errors.
- numpy (np): random.default_rng(), normal(), random(), sin(), abs(), arange(), polyfit().
- pandas (pd): DataFrame(), to_datetime(), dropna(), drop_duplicates(), fillna(), .dt.date,
  head(), mean()/max()/min(), groupby().agg(), round(), corr(), plot(kind="bar").
- matplotlib.pyplot (plt): subplots(), plot(), scatter(), imshow(), colorbar(), annotate(),
  set(), tick_params(), tight_layout(), show().

=======================================================================================
EXPECTED OUTPUT (real run with NO valid API key, so the SYNTHETIC fallback produced these
numbers; live data would show different values and the line "Loaded LIVE data ...")
=======================================================================================
- HTTP status: 401 (empty key rejected), then "Using SYNTHETIC sample data ...".
- 40 readings; the start timestamp depends on the time of running, so dates differ by day.
- Average temp 29.0 C | Max 33.6 | Min 22.7.
- Temperature trend about -0.016 C per 3-hour step.
- Daily summary depends on run time: about 6 rows (5 if run at midnight); the avg_temp and
  total rain values vary with the start time.
- Correlation: temp vs humidity about -0.95 (strong negative); the others are near 0.
- A figure with 6 charts is produced.

=======================================================================================
DETAILED VIVA Q&A
=======================================================================================
Q1. What is an API?
A1. An API is a defined way for one program to request data or services from another. A Web
    API does this over HTTP, usually replying with JSON, so we can use remote data without
    knowing how the server stores it.

Q2. What is REST?
A2. REST is an architecture style for web APIs in which resources are identified by URLs and
    accessed with standard HTTP methods like GET and POST. It is stateless, meaning each
    request is independent.

Q3. What is JSON and why is it used?
A3. JSON is a lightweight text format of key-value pairs and lists. It is human readable,
    language independent and maps directly to Python dictionaries and lists.

Q4. Why should we check the HTTP status code?
A4. A code other than 200 means the request failed (401 bad key, 404 city not found, 429
    rate limit). Checking it prevents errors from reading data that does not exist and lets
    us show a helpful message or use a fallback.

Q5. What is an API key and how should it be stored?
A5. It is a secret identifier that authenticates you and lets the provider track and limit
    usage. It must be kept in an environment variable or a secret file that is not committed
    to GitHub, never hard-coded.

Q6. Why do we use timeout in requests.get()?
A6. Without it the program could wait forever if the server is slow or down. A timeout raises
    an exception after the set seconds, which we catch and handle.

Q7. Why use .get("rain", {}) instead of it["rain"]?
A7. The API includes the rain field only when it rains. Direct indexing would raise a
    KeyError on dry readings, while .get() returns a default and keeps the loop running.

Q8. What does groupby("date").agg() do here?
A8. It groups the 3-hourly readings by calendar day and calculates summary values for each
    day (mean, max, min, sum), giving one row per day.

Q9. What does a negative correlation between temperature and humidity mean?
A9. When temperature rises, relative humidity tends to fall. The value is near -1 when the
    pattern is strong, as in the synthetic data (about -0.95).

Q10. What is the difference between a forecast API and a history API?
A10. A forecast API predicts future readings (here the next 5 days at 3-hour steps). A history
     API returns past measured values and is usually a paid feature.

=======================================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=======================================================================================
Q1. How does the script behave without internet or with a wrong key?
A1. requests raises a RequestException (offline) which we catch and print, or the status is
    401 so data stays empty. In both cases data.get("list") is falsy and the script builds
    synthetic data so the analysis and plots still run.

Q2. Why is rng = np.random.default_rng(1) used?
A2. The seed 1 makes the random numbers repeatable, so the synthetic noise is the same
    each run (only the timestamps change with the current time).

Q3. What does errors="coerce" do in pd.to_datetime()?
A3. Instead of raising an error for text that is not a date, it converts it to NaT (missing
    time). We then drop those rows with dropna(subset=["datetime"]).

Q4. What does np.polyfit(np.arange(len(df)), df["temp"], 1)[0] return?
A4. The slope of the best straight line of temperature against reading number, i.e. the
    average change in temperature per 3-hour step.

Q5. How do you run the script for another city?
A5. Pass the city as a command line argument: python 2_Weather_API.py Mumbai. sys.argv[1]
    then replaces the default Pune.

Q6. Why is the API key not written in the code?
A6. Code is shared and uploaded to repositories, which would expose the secret. os.getenv
    reads it from the user's environment at run time.

Q7. How could this code be extended to show several cities on a map?
A7. Loop over a list of cities, store each city's lat, lon and mean temperature in lists,
    then call scatter() once with all points, or use folium/geopandas for a real map.

END OF VIVA NOTES
"""
