"""
Assignment 5 : Data Visualization using matplotlib - Air Quality Index (AQI) trends
===================================================================================
INSTRUCTIONS
 1. Install:  pip install pandas numpy matplotlib
 2. Dataset : City_Air_Quality.csv = Kaggle "Air Quality Data in India" (rohanrao/air-quality-data-in-india,
              file city_day.csv, renamed): 29531 rows x 16 columns, 26 cities. Columns used: City, Date,
              PM2.5, PM10, CO, AQI.
              Option A (default): fetched online from the datasets repo.
              Option B: a local copy next to this script (uncomment that line, comment Option A).
 3. Run     : python 5_AQI_Visualization.py
 4. Steps   : load -> explore -> choose variables -> line plots (AQI, pollutants) -> bar plot
              (monthly) -> box/violin (AQI category) -> scatter/bubble -> customise
"""
# --- Step 0: Imports and setup ---
import matplotlib.pyplot as plt
import pandas as pd

# --- Step 1: Import the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/City_Air_Quality.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("City_Air_Quality.csv")

# --- Step 2: Explore the data ---
# Understand the data before plotting: size, column types, summary stats.
print("Shape:", df.shape); df.info(); print(df.describe().round(2))

# --- Step 3: Pick variables and clean ---
# Plot one city only (Delhi, else the first), or lines from several cities would mix.
if "City" in df.columns and df["City"].nunique() > 1:
    df = df[df["City"] == ("Delhi" if "Delhi" in set(df["City"]) else df["City"].iloc[0])].copy()
# Real dates, no gaps in plotted columns, oldest first.
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["AQI", "PM2.5", "PM10", "CO"]).sort_values("Date").reset_index(drop=True)
# Keep the latest 365 records (about one year) for readable charts.
df = df.tail(365).copy()
print(f"\nUsing {len(df)} records from {df.Date.min().date()} to {df.Date.max().date()}")

# AQI bucket (India CPCB bands) used for grouping
# Turn AQI into CPCB band labels, and dates into year-month for monthly grouping.
bins = [0, 50, 100, 200, 300, 400, 10000]
labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
df["Category"] = pd.cut(df["AQI"], bins, labels=labels)
df["Month"] = df["Date"].dt.to_period("M").astype(str)

# --- Steps 4-5: Line plots (trend over time) ---
# One figure with a 2x2 grid of plots.
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
# Line chart: daily AQI with a 14-day average to smooth the noise.
ax[0, 0].plot(df["Date"], df["AQI"], color="#d95f02", lw=1, label="AQI")
ax[0, 0].plot(df["Date"], df["AQI"].rolling(14).mean(), color="k", lw=2, label="14-day mean")
ax[0, 0].set(title="Overall AQI trend", xlabel="Date", ylabel="AQI"); ax[0, 0].legend()
# Line chart: PM2.5 vs PM10 on one axis for direct comparison.
ax[0, 1].plot(df["Date"], df["PM2.5"], color="crimson", label="PM2.5")
ax[0, 1].plot(df["Date"], df["PM10"], color="navy", label="PM10")
ax[0, 1].set(title="Particulate levels", xlabel="Date", ylabel="ug/m3"); ax[0, 1].legend()
# Line chart: CO alone, since its scale differs from the particulates.
ax[1, 0].plot(df["Date"], df["CO"], color="forestgreen", label="CO")
ax[1, 0].set(title="Carbon monoxide", xlabel="Date", ylabel="mg/m3"); ax[1, 0].legend()

# --- Step 6: Bar plot (monthly average AQI) ---
# Bar chart: average AQI per month, shows seasonal differences.
monthly = df.groupby("Month")["AQI"].mean()
ax[1, 1].bar(monthly.index, monthly.values, color="steelblue", edgecolor="black")
ax[1, 1].set(title="Monthly average AQI", xlabel="Month", ylabel="Avg AQI")
ax[1, 1].tick_params(axis="x", rotation=60)
# Same grid and tilted labels on all four plots, then show the figure.
for a in ax.ravel():
    a.grid(True, ls=":", alpha=.6); a.tick_params(axis="x", rotation=40)
plt.tight_layout(); plt.show()

# --- Step 7: Box + violin of pollutant levels by AQI category ---
# Box and violin plots show the spread per group, which a bar of averages hides.
# Keep only the AQI bands that occur in the data, in order.
cats = [c for c in labels if (df["Category"] == c).any()]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# Box plot: PM2.5 per category (median, spread, outliers).
ax[0].boxplot([df.loc[df.Category == c, "PM2.5"] for c in cats], tick_labels=cats, patch_artist=True)
ax[0].set(title="PM2.5 by AQI category (box)", ylabel="PM2.5")
# Violin plot: PM10 per category, also shows the shape of the data.
ax[1].violinplot([df.loc[df.Category == c, "PM10"] for c in cats], showmedians=True)
ax[1].set_xticks(range(1, len(cats) + 1)); ax[1].set_xticklabels(cats)   # violin groups start at 1
ax[1].set(title="PM10 by AQI category (violin)", ylabel="PM10")
for a in ax: a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()

# --- Step 8: Scatter / bubble plot ---
# Bubble chart: AQI vs PM2.5, colour = CO, size = PM10 (four variables in one chart).
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["PM2.5"], df["AQI"], c=df["CO"], s=df["PM10"] * 0.5, cmap="viridis", alpha=.7, edgecolors="k")
plt.colorbar(sc, label="CO level")
plt.title("AQI vs PM2.5 (colour = CO, bubble size = PM10)", fontweight="bold")
plt.xlabel("PM2.5"); plt.ylabel("AQI"); plt.grid(True, ls=":", alpha=.6)
plt.tight_layout(); plt.show()
# Back the picture with numbers: correlation of each column with AQI.
print("\nCorrelation with AQI:\n", df[["PM2.5", "PM10", "CO", "AQI"]].corr()["AQI"].round(3))

"""
============================================================================
CODE EXPLANATION
============================================================================
 1. Imports: matplotlib.pyplot (charts), pandas (data).
 2. Load: Option A (default) pd.read_csv(url) fetches the CSV online from the datasets repo. Option B reads a
    local copy: comment out Option A and uncomment the local pd.read_csv line.
 3. Explore: df.shape, df.info(), df.describe() show size, types and summary statistics.
 4. Choose and clean: filter one city, to_datetime for dates, dropna on the plotted columns, sort_values,
    reset_index, tail(365) for the latest year.
 5. AQI categories: pd.cut applies the CPCB bands; dt.to_period("M") makes a Month column.
 6. Line plots: plt.subplots(2, 2) and ax.plot draw AQI with a 14-day rolling mean, PM2.5 vs PM10 and CO.
 7. Bar plot: groupby("Month")["AQI"].mean() then ax.bar shows the monthly average AQI.
 8. Box and violin: ax.boxplot and ax.violinplot show PM2.5 and PM10 distributions for each AQI category.
 9. Scatter/bubble: plt.scatter plots AQI against PM2.5 with colour = CO and size = PM10; colorbar explains colour.
10. Correlation: df[...].corr()["AQI"] prints how strongly each pollutant moves with AQI.
11. Customisation in every plot: title, axis labels, legend, grid, tick rotation, colours, tight_layout.

============================================================================
DETAILED THEORY NOTES
============================================================================
1. What is data visualization (matplotlib)?
 - Showing data as charts so patterns, trends and outliers are seen quickly.
 - matplotlib is the base Python plotting library; pyplot (plt) is its simple state-based interface.
 - Figure = the whole canvas; Axes = one plot inside it; Axis = the x or y line with ticks.
 - plt.subplots() creates a Figure and Axes together; plt.subplot() creates one axes at a time.

2. Chart types and when to use each
 - Line plot: change over time (trend); here the daily AQI and pollutants.
 - Bar plot: compare values across categories; here the average AQI per month.
 - Histogram: shape of one numeric variable (not used here, but common).
 - Box plot: median, quartiles (middle 50%), whiskers and outliers per group; compares spread between groups.
 - Violin plot: box plot plus a mirrored density curve; shows whether data has one or several peaks.
 - Scatter plot: relationship between two numeric variables; each dot is one observation.
 - Bubble chart: scatter plot where dot size (and colour) carry extra variables.
 - Pie chart: share of a whole (only for a few slices; used in assignment 6).
 - Heatmap: colour-coded table, often for correlation matrices.

3. Rolling (moving) mean
 - Average of the last N values (here 14 days), recomputed for each day.
 - Smooths short-term noise so the long-term trend is visible.
 - The first N-1 values are NaN because a full window is not yet available.

4. AQI and its bands
 - AQI (Air Quality Index) turns pollutant concentrations into one number for easy public understanding.
 - It is decided by the worst pollutant, mainly PM2.5 and PM10, which explains the strong correlation.
 - PM2.5 = particles under 2.5 micrometres (reach deep into lungs); PM10 = under 10 micrometres.
 - CO = carbon monoxide, a gas from incomplete burning (vehicles, fires).
 - CPCB India bands: 0-50 Good, 51-100 Satisfactory, 101-200 Moderate, 201-300 Poor,
   301-400 Very Poor, 401+ Severe (the bins above use these edges).

5. Correlation
 - Pearson correlation r ranges from -1 to +1; near +1 = rise together, near 0 = no linear link.
 - Correlation does not prove causation.

6. Customising plots
 - Title, axis labels with units, legend, grid, readable tick rotation, sensible colours and sizes.
 - alpha (transparency) helps when points overlap; a colorbar explains a colour scale.

7. The dataset
 - Source: Kaggle "Air Quality Data in India" (city_day.csv): 29531 rows, 16 columns, 26 cities, daily readings.
 - Columns used: City, Date, PM2.5, PM10, CO, AQI (the file has many more, e.g. NO2, SO2, O3, AQI_Bucket).
 - One row = one city on one day. Real data has gaps (NaN), so we dropna on the plotted columns.
 - The script picks Delhi (2009 rows), drops rows missing AQI/PM2.5/PM10/CO and keeps the latest 365 records
   (2019-07-03 to 2020-07-01).

8. Advantages and disadvantages of visualization
 - Advantages: patterns and outliers are visible at a glance, easy to communicate, good for exploring data.
 - Disadvantages: wrong chart choice or truncated axes can mislead; static charts hide exact numbers;
   too many variables in one chart (bubble plots) can be hard to read.

9. Real-world uses
 - Government and city dashboards for air quality alerts; health and policy planning.
 - Business reporting, scientific papers, finance (price trends), monitoring systems.

============================================================================
EXPECTED OUTPUT (real numbers from the Kaggle city_day.csv, Delhi)
============================================================================
 Shape: (29531, 16) for the whole file (26 cities); full-file describe(): AQI mean 166.46, max 2049.
 Using 365 records from 2019-07-03 to 2020-07-01 (Delhi, latest 365 rows with no missing AQI/PM2.5/PM10/CO)
 AQI of those 365 rows: mean 200.15, std 118.33, min 51, median 152, max 659
 (PM2.5 mean 94.90, PM10 mean 176.76, CO mean 1.21)
 Correlation with AQI: PM2.5 0.949, PM10 0.946, CO 0.817, AQI 1.000
 Monthly average AQI: 2019-07 141.4, 08 99.3, 09 106.5, 10 251.0, 11 356.1, 12 353.9, 2020-01 308.5, 02 262.1,
   03 135.8, 04 113.0, 05 148.6, 06 125.7, 2020-07 101.0 (only 1 day).
   Pattern: monsoon (Aug-Sep) is cleanest, AQI jumps in Oct and peaks in Nov-Dec (winter, crop burning, cold air),
   then falls from Feb-Mar.
 AQI category counts (days): Good 0, Satisfactory 68, Moderate 157, Poor 63, Very Poor 51, Severe 26
 Charts: 3 figures open (2x2 trend/bar grid, box+violin pair, scatter/bubble). With MPLBACKEND=Agg they are not
 shown, only a harmless "non-interactive" warning is printed.

============================================================================
LIBRARIES AND FUNCTIONS USED
============================================================================
 - matplotlib.pyplot as plt: subplots, figure, plot, bar, boxplot, violinplot, scatter, colorbar, title, xlabel,
   ylabel, grid, tight_layout, show.
 - Axes methods: set, legend, grid, tick_params, set_xticks, set_xticklabels.
 - pandas as pd: read_csv, to_datetime, cut.
 - df methods: info, describe, nunique, dropna, sort_values, reset_index, tail, groupby().mean, corr, loc.
 - Series.rolling(14).mean - 14-day moving average.
 - Series.dt.to_period("M") and astype(str) - month labels.
 - Series.round, Series.min/max - tidy numbers and range checks.

============================================================================
DETAILED VIVA Q&A
============================================================================
Q1. What is matplotlib and what is pyplot?
A1. matplotlib is Python's main plotting library. pyplot is its module with MATLAB-like functions (plot, bar,
    scatter, show) that create and style figures easily.

Q2. Difference between plt.plot and plt.scatter?
A2. plot() joins points with a line and suits ordered data like time series. scatter() draws unconnected dots and
    allows per-point colour and size, so it suits relationships between variables.

Q3. Difference between plt.subplot and plt.subplots?
A3. subplot() adds one axes to a grid at a time. subplots() creates the whole figure and an array of axes in one
    call, which is cleaner for multi-plot layouts like our 2x2 grid.

Q4. Box plot versus violin plot?
A4. A box plot summarises data by median, quartiles, whiskers and outliers. A violin plot adds a density curve
    on both sides, so you can also see the shape and peaks of the distribution.

Q5. What is a rolling mean and why use it?
A5. It averages the last N values for every point (14 days here). It smooths daily ups and downs so the
    underlying trend is visible. The first 13 values are NaN because the window is not full.

Q6. When do you use a line chart, bar chart and scatter plot?
A6. Line chart for trends over time, bar chart to compare categories (months), scatter plot to see the relationship
    between two numeric variables. Choosing the right type keeps the message clear.

Q7. What is AQI and how is it categorised?
A7. AQI is a single number summarising air pollution. In India (CPCB) it is Good 0-50, Satisfactory 51-100,
    Moderate 101-200, Poor 201-300, Very Poor 301-400 and Severe above 400.

Q8. What are PM2.5 and PM10, and why do they dominate AQI?
A8. They are particles smaller than 2.5 and 10 micrometres. They are the usual worst pollutants in Indian cities and
    they harm lungs, so they set the AQI value; hence the correlation of about 0.95 with AQI in our Delhi data.
    CO also correlates strongly here (0.82) because Delhi's CO rises together with PM in winter.

Q9. What is an outlier and how does a box plot show it?
A9. An outlier is a value far from the rest. A box plot draws points beyond the whiskers (1.5 x IQR from the box)
    as separate dots.

Q10. What is correlation and does a high value mean causation?
A10. Correlation measures the linear relationship between two variables from -1 to +1. A high value does not prove
     one causes the other; other factors may drive both.

============================================================================
PRACTICAL AND CODING VIVA QUESTIONS (about this exact code)
============================================================================
Q1. Why do we call pd.to_datetime on the Date column?
A1. The CSV stores dates as text. Converting them gives real dates so sort_values orders them in time and
    matplotlib draws a proper time axis.

Q2. Why is dropna(subset=[...]) used before plotting?
A2. Real pollutant data has NaN gaps, which break lines and distort averages. Dropping rows with gaps in AQI, PM2.5,
    PM10 and CO keeps every plot and the correlation honest.

Q3. How is the 14-day trend line created in the code?
A3. df["AQI"].rolling(14).mean() computes the average of each 14-day window, and it is plotted over the raw AQI line
    in black with a thicker width.

Q4. Why does the violin plot need set_xticks and set_xticklabels?
A4. violinplot numbers the groups 1, 2, 3..., unlike boxplot where we passed tick_labels directly. So we set the
    ticks at 1..N and then give the category names.

Q5. In the scatter plot what do colour and size represent?
A5. c=df["CO"] colours each dot by its CO level (explained by the colorbar) and s=df["PM10"]*0.5 sets the bubble
    size by PM10. Together with x = PM2.5 and y = AQI, four variables appear in one chart.

Q6. Why is cats built with a list comprehension?
A6. Some AQI bands may have no days, and boxplot/violinplot would fail or show empty groups for them. The list keeps
    only the bands that occur, in correct order.

Q7. What does the monthly bar chart show for the real Delhi data?
A7. Delhi is cleanest in the monsoon (Aug-Sep, AQI about 100) and worst in winter: Nov 356 and Dec 354 on average,
    still 308 in Jan. The Oct-Feb jump is the well-known winter pollution season (crop burning, cold still air).

END OF VIVA NOTES
"""
