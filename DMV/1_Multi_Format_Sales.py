"""
Assignment 1 : Data Loading, Storage and File Formats - Sales data from CSV, Excel and JSON
===========================================================================================
INSTRUCTIONS
 1. Install:  pip install pandas numpy matplotlib scikit-learn openpyxl
              (openpyxl is needed by pandas to read/write .xlsx)
 2. Dataset : none to download. The script first CREATES sales.csv, sales.xlsx and sales.json
              (small sample data with a duplicate row and a missing value on purpose) in the
              same folder, then loads them back. Replace them with your own files if you have any.
 3. Run     : python 1_Multi_Format_Sales.py
 4. Steps   : create files -> load each format -> explore -> clean (duplicates, missing)
              -> unify into one DataFrame -> transform (TotalSales, date parts, split column)
              -> analyse (totals, averages, groupby) -> visualise (bar, pie, box)
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Work in the script's folder so the sales files are created next to it.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Step 1: Create sample files in three formats ---
# CSV data: ID 2 is duplicated and Units is missing, on purpose, to practise cleaning.
d1 = {"ID": [1, 2, 2], "Date": ["2024-01-10", "2024-01-15", "2024-01-15"],
      "Region": ["North", "South", "South"], "Cat": ["Tech", "Home", "Home"],
      "Units": [5, None, None], "Price": [1200, 80, 80], "AdSpend": [3000, 800, 800]}
# Excel data: same columns as d1 so the tables can be stacked later.
d2 = {"ID": [3, 4], "Date": ["2024-02-01", "2024-02-15"], "Region": ["West", "East"],
      "Cat": ["Tech", "Cloth"], "Units": [10, 25], "Price": [1100, 40], "AdSpend": [2500, 1200]}
# JSON data: a list of dictionaries, the usual "records" layout.
d3 = [{"ID": 5, "Date": "2024-03-01", "Region": "North", "Cat": "Cloth",
       "Units": 30, "Price": 35, "AdSpend": 1500},
      {"ID": 6, "Date": "2024-03-12", "Region": "South", "Cat": "Home",
       "Units": 18, "Price": 90, "AdSpend": 900}]

# Save each dataset in its own format (index=False skips the row numbers).
pd.DataFrame(d1).to_csv("sales.csv", index=False)
pd.DataFrame(d2).to_excel("sales.xlsx", index=False)
with open("sales.json", "w") as f:
    json.dump(d3, f)

# --- Step 2: Load each format ---
csv_df = pd.read_csv("sales.csv")
xls_df = pd.read_excel("sales.xlsx")
json_df = pd.read_json("sales.json")

# --- Step 3: Explore ---
# Print the same health check for every format: size, missing cells, column types.
for name, d in [("CSV", csv_df), ("Excel", xls_df), ("JSON", json_df)]:
    print(f"--- {name}: shape {d.shape} | missing {int(d.isnull().sum().sum())} | dtypes ---")
    print(d.dtypes.to_string(), "\n")

# --- Step 4: Unify into one DataFrame ---
# Stack the three tables (column names match); ignore_index renumbers the rows.
df = pd.concat([csv_df, xls_df, json_df], ignore_index=True)

# --- Step 5: Clean - duplicates + missing values ---
# ID is the unique key of an order, so repeated IDs are duplicates.
print("Duplicate IDs:", df.duplicated(subset=["ID"]).sum())
df = df.drop_duplicates(subset=["ID"]).copy()
# Fill missing Units with the median, which extreme values do not distort.
df["Units"] = df["Units"].fillna(df["Units"].median())
# Turn Date text into real dates so month/year/day can be read.
df["Date"] = pd.to_datetime(df["Date"])

# --- Step 6: Transform - derived variables, split column ---
# Revenue per order, plus month name and numeric year/month/day from the date.
df["TotalSales"] = df["Units"] * df["Price"]
df["Month"] = df["Date"].dt.month_name()
df["Year"], df["Mon"], df["Day"] = df["Date"].dt.year, df["Date"].dt.month, df["Date"].dt.day
# Join Region and Cat, then split them apart again (practice for splitting a column).
df["Region_Cat"] = df["Region"] + "-" + df["Cat"]
df[["Reg", "Category"]] = df["Region_Cat"].str.split("-", expand=True)  # expand=True gives 2 columns
print("\n--- Cleaned Unified Data ---\n", df)

# --- Step 7: Analyse ---
# Summary statistics and overall revenue figures.
print("\nDescriptive statistics:\n", df[["Units", "Price", "TotalSales"]].describe().round(2))
print(f"\nTotal Revenue       : {df['TotalSales'].sum():,.2f}")
print(f"Average Order Value : {df['TotalSales'].mean():,.2f}")
# Add up sales within each category and each region.
print("\nSales by Category:\n", df.groupby("Cat")["TotalSales"].sum())
print("\nSales by Region:\n", df.groupby("Region")["TotalSales"].sum())
# Each category's share of orders, in percent.
print("\nCategory distribution (%):\n", (df["Cat"].value_counts(normalize=True) * 100).round(1))
# Correlation shows if revenue moves with ad spend (-1 to +1).
print(f"\nCorrelation TotalSales vs AdSpend: {df['TotalSales'].corr(df['AdSpend']):.4f}")

# Forecast the next order with a straight line fitted over the order sequence.
X = np.arange(len(df)).reshape(-1, 1)  # sklearn needs inputs as a 2-D column
model = LinearRegression().fit(X, df["TotalSales"])
print(f"Next order forecast : {model.predict([[len(df)]])[0]:,.2f}")

# --- Step 8: Visualise ---
# Three charts side by side: bar (totals), pie (shares), box (spread).
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("Cat")["TotalSales"].sum().plot(kind="bar", ax=ax[0], color="steelblue", title="Revenue by Category")
df.groupby("Region")["TotalSales"].sum().plot(kind="pie", ax=ax[1], autopct="%1.0f%%", title="Revenue by Region")
df.boxplot(column="TotalSales", by="Cat", ax=ax[2]); ax[2].set_title("TotalSales by Category")
# suptitle("") removes pandas' automatic extra title.
plt.suptitle(""); plt.tight_layout(); plt.show()

"""
=======================================================================================
CODE EXPLANATION
=======================================================================================
1. Step 1 (create files): We built three small datasets in Python and saved them as
   sales.csv (pandas to_csv), sales.xlsx (pandas to_excel, needs openpyxl) and sales.json
   (json.dump). A duplicate ID and a missing Units value are planted on purpose.
2. Step 2 (load): We read the three files back with pd.read_csv, pd.read_excel, pd.read_json.
3. Step 3 (explore): For each table we printed shape, number of missing cells and dtypes
   using .shape, .isnull().sum().sum() and .dtypes.
4. Step 4 (unify): pd.concat stacked the three DataFrames into one because the column names match.
5. Step 5 (clean): duplicated()/drop_duplicates(subset=["ID"]) removed the repeated order,
   fillna(median) filled the missing Units, and pd.to_datetime fixed the Date column type.
6. Step 6 (transform): Created TotalSales = Units * Price, month name and year/month/day using
   the .dt accessor, and split Region_Cat into two columns using str.split(expand=True).
7. Step 7 (analyse): describe(), sum(), mean(), groupby().sum(), value_counts(normalize=True),
   corr() and a LinearRegression forecast of the next order value.
8. Step 8 (visualise): matplotlib/pandas drew a bar chart (category revenue), pie chart
   (region share) and box plot (TotalSales spread per category).

=======================================================================================
DETAILED THEORY NOTES
=======================================================================================
1. What is data loading and storage?
   - Data loading = reading data from a file/database/API into memory (a DataFrame).
   - Data storage = writing results back to a file so they can be shared or reused.
   - A file format is the agreed way the bytes of a file are arranged.

2. File formats used here
   - CSV (Comma Separated Values): plain text, one row per line, values split by commas.
     Small, simple, opens anywhere, but has NO data types and NO formatting. Everything is text
     until the reader guesses the type.
   - Excel (.xlsx): a zipped XML workbook with sheets, formulas, colours and types.
     Friendly for business users, but heavier and slower than CSV and needs openpyxl in Python.
   - JSON (JavaScript Object Notation): text with key-value pairs and lists; it can be nested.
     This is the standard format of web APIs. Flat records convert easily to a table.
   - Other formats (not used here): Parquet (columnar, compressed, very fast), SQL tables, XML.

3. Dataset description (made inside the script)
   - Columns: ID (order id), Date, Region (North/South/East/West), Cat (Tech/Home/Cloth),
     Units (units sold), Price (price per unit), AdSpend (advertising money spent).
   - Size: 3 rows (CSV, 1 duplicate), 2 rows (Excel), 2 rows (JSON) = 7 rows, 6 after cleaning.
   - Meaning: each row is one sales order.

4. Missing-value strategies
   - Drop rows (dropna): simple but loses information.
   - Mean imputation: good for symmetric data, but is pulled by outliers.
   - Median imputation: the middle value; robust to outliers. We used this.
   - Mode imputation: most frequent value; used for categories.
   - Forward/backward fill: copy neighbouring value; used for time series.
   - Model-based imputation (KNN, regression): predicts the missing value from other columns.

5. Duplicates
   - Duplicate = the same record stored more than once. It inflates counts and totals.
   - We used the key column ID (subset=["ID"]) because the same order could differ in a
     small detail but still be the same order. Whole-row matching would miss that.

6. Concatenation vs merge
   - concat stacks tables (rows under rows, or columns side by side) using matching labels.
   - merge joins tables on a key column (like SQL JOIN) to add information sideways.

7. Data types and parsing dates
   - Wrong dtypes (dates as text) block date maths. to_datetime converts text to datetime so
     the .dt accessor can extract year, month, day and weekday.

8. Feature / derived variable creation
   - A derived variable is a new column calculated from old ones (TotalSales = Units * Price).
   - Splitting a column (Region-Cat) separates mixed information into clean columns.

9. Aggregation
   - Aggregation = summarising many rows into one value per group (sum, mean, count).
   - groupby() splits data into groups, applies a function, and combines the result.

10. Descriptive statistics and correlation
   - Mean = average, median = middle value, std = spread around the mean.
   - Correlation (-1 to +1) shows how strongly two numbers move together; it does NOT prove
     one causes the other.

11. Simple forecasting with linear regression
   - Linear regression fits y = a*x + b. Here x is the order number and y is TotalSales.
   - With only 6 points the forecast is only a demonstration, not a reliable prediction.

12. Chart choice
   - Bar = compare categories. Pie = share of a whole (use few slices).
   - Box plot = median, quartiles and outliers (shows spread).

13. Advantages / disadvantages of each format
   - CSV: + tiny, universal, fast; - no types, no multiple sheets, delimiter problems.
   - Excel: + sheets, formatting, formulas; - slow, bigger, needs extra library.
   - JSON: + nested data, standard for APIs; - verbose, harder to read as a table when nested.

14. Real-world uses
   - CSV for data exports and Kaggle datasets; Excel for finance/business reports;
     JSON for web/mobile API responses and configuration; combining all three is common when
     sales come from a shop system (CSV), an accountant (Excel) and an online app (JSON).

=======================================================================================
LIBRARIES AND FUNCTIONS USED
=======================================================================================
- json: dump() writes a Python object to a JSON file.
- os: chdir() changes the working folder; os.path.dirname()/abspath() find the script folder.
- numpy (np): arange() makes the order numbers; reshape() makes a 2-D input array.
- pandas (pd): DataFrame(), to_csv(), to_excel(), read_csv(), read_excel(), read_json(),
  concat(), duplicated(), drop_duplicates(), fillna(), median(), to_datetime(), .dt.month_name(),
  .dt.year/.month/.day, str.split(), describe(), sum(), mean(), groupby(), value_counts(),
  corr(), boxplot(), plot().
- sklearn.linear_model.LinearRegression: fit() learns the line, predict() gives the forecast.
- matplotlib.pyplot (plt): subplots(), suptitle(), tight_layout(), show().

=======================================================================================
EXPECTED OUTPUT (real numbers from a run; data is generated inside the script, so it is
the same every time)
=======================================================================================
- Shapes: CSV (3, 7) with 2 missing; Excel (2, 7) with 0 missing; JSON (2, 7) with 0 missing.
- JSON loads Date as datetime64 while CSV/Excel load it as text.
- Duplicate IDs: 1. After cleaning there are 6 rows; the missing Units become 18 (the median).
- Total Revenue 22,110.00 ; Average Order Value 3,685.00.
- Sales by Category: Cloth 2050, Home 3060, Tech 17000.
- Sales by Region: East 1000, North 7050, South 3060, West 11000.
- Category distribution: Tech 33.3 %, Home 33.3 %, Cloth 33.3 %.
- Correlation TotalSales vs AdSpend = 0.7883 ; Next order forecast = 378.00.
- A window with 3 charts is shown (with MPLBACKEND=Agg it is only rendered, not displayed).

=======================================================================================
DETAILED VIVA Q&A
=======================================================================================
Q1. What is the difference between CSV, Excel and JSON?
A1. CSV is plain text with rows and commas and no data types. Excel is a binary-like zipped
    workbook with sheets and formatting. JSON stores key-value pairs and can be nested, so it
    is the usual format of web APIs.

Q2. Why can pandas stack the three files with concat()?
A2. Because all three tables have the same column names. concat lines up the columns by
    label and appends the rows, so the result is one table with the same 7 columns.

Q3. What is the difference between concat and merge?
A3. concat stacks DataFrames along rows or columns without needing a key. merge combines
    tables side by side using a common key column, like a SQL JOIN.

Q4. How do you handle missing values?
A4. First find them with isnull().sum(). Then either drop rows (dropna) or impute: mean or
    median for numbers, mode for categories, or forward fill for time series. We used the
    median for Units because it is not affected by outliers.

Q5. Why did you drop duplicates by ID and not by whole row?
A5. ID is the unique key of an order, so two rows with the same ID are the same order even if
    another column differs slightly. Whole-row comparison would miss such repeats and still
    double count the order.

Q6. Why convert the Date column with to_datetime()?
A6. Text dates cannot give month, year or day. After conversion the .dt accessor extracts
    them and allows date sorting and date arithmetic.

Q7. What does groupby() do?
A7. It splits the table into groups using a column (e.g. Cat), applies a function such as
    sum to each group, and combines the answers into a new table. It is the pandas way of
    doing aggregation.

Q8. When do you use a bar, a pie and a box plot?
A8. Bar charts compare totals between categories, pie charts show each part's share of a
    whole, and box plots show the median, spread and outliers of a numeric variable.

Q9. Does a correlation of 0.79 prove advertising causes sales?
A9. No. Correlation only says the two move together. Other factors (season, price, product)
    may drive both, and with 6 rows the value is also very unstable.

Q10. Why is the forecast not reliable?
A10. It fits a straight line on only 6 orders ordered by sequence, which ignores season,
     region and product. It is shown only to demonstrate fit() and predict().

=======================================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=======================================================================================
Q1. What does index=False do in to_csv()?
A1. It stops pandas from writing the 0,1,2... row numbers as an extra first column, so the
    file contains only the real data columns.

Q2. Why do we call .copy() after drop_duplicates()?
A2. It makes an independent table so later column assignments (df["TotalSales"] = ...) do
    not trigger a SettingWithCopyWarning or change a view of the old data.

Q3. Why does the CSV Units column show float64 and not int64?
A3. Because it contains missing values (NaN). NaN is a float, so pandas upgrades the whole
    column to float. After fillna with the median the values are whole numbers stored as floats.

Q4. What does str.split("-", expand=True) return?
A4. A DataFrame with one column per piece of the split text (here two columns: Reg and
    Category). Without expand=True it would return a column of lists.

Q5. Why is reshape(-1, 1) needed before LinearRegression.fit()?
A5. scikit-learn expects the inputs X as a 2-D array of shape (rows, features). A single
    feature must therefore be a column with shape (n, 1). The -1 lets numpy work out n.

Q6. How would you make the code read a different Excel sheet?
A6. Pass the sheet name or number: pd.read_excel("sales.xlsx", sheet_name="Sheet2").

Q7. What happens if openpyxl is not installed?
A7. to_excel() and read_excel() raise an ImportError saying openpyxl is required, so you must
    run pip install openpyxl.

END OF VIVA NOTES
"""
