"""
Assignment 4 : Data Wrangling - Real estate market prices
=========================================================
INSTRUCTIONS
 1. Install:  pip install pandas numpy
 2. Dataset : RealEstate_Prices.csv. A convenient public one is the California housing data
              (ageron/handson-ml2 housing.csv): columns longitude, latitude, housing_median_age,
              total_rooms, total_bedrooms, population, households, median_income,
              median_house_value, ocean_proximity.
              Option A (default): fetched online from the datasets repo.
              Option B: a local copy next to this script (uncomment that line, comment Option A).
 3. Run     : python 4_RealEstate_Wrangling.py
 4. Steps   : clean column names -> missing values -> merge a second table -> filter/subset
              -> encode categoricals -> aggregate -> outliers (IQR)
"""
# --- Step 0: Imports and setup ---
import pandas as pd

# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/RealEstate_Prices.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("RealEstate_Prices.csv")

print("Raw shape:", df.shape)

# --- Step 1: Task 1 - Clean column names ---
# Make names lower_case_with_underscores to avoid typing mistakes.
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", "_", regex=True))
print("=== Task 1: Cleaned columns ===\n", df.columns.tolist())

# --- Step 2: Task 2 - Handle missing values ---
# Count the gaps per column.
miss = df.isnull().sum()
print("\n=== Task 2: Missing values ===\n", miss[miss > 0].to_dict())
# Numbers get the median (not distorted by outliers); text gets the most common value.
for c in df.select_dtypes("number").columns:
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes("object").columns:
    df[c] = df[c].fillna(df[c].mode()[0])
print("Nulls after:", int(df.isnull().sum().sum()))

# --- Step 3: Task 3 - Merge with an additional dataset (location lookup) ---
# Small lookup table with made-up flood risk and amenity scores per area type.
lookup = pd.DataFrame({"ocean_proximity": ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
                       "flood_risk_score": [2.5, 1.2, 0.8, 3.1, 4.0],
                       "amenity_index": [8, 7, 4, 9, 3]})
# Left join on the key keeps every property row.
df = df.merge(lookup, on="ocean_proximity", how="left")
print("\n=== Task 3: Merged sample ===\n", df[["ocean_proximity", "flood_risk_score", "amenity_index"]].drop_duplicates())

# --- Step 4: Task 4 - Filter / subset ---
# Keep expensive (> 200000) AND new (< 20 years) properties; each condition needs brackets.
subset = df[(df["median_house_value"] > 200000) & (df["housing_median_age"] < 20)]
print(f"\n=== Task 4: Expensive & new properties: {len(subset)} of {len(df)} rows ===")
print("Inland only:", len(df[df["ocean_proximity"] == "INLAND"]), "rows")

# --- Step 5: Task 5 - Encode categoricals ---
# Models need numbers, so encode ocean_proximity two ways: one-hot and label codes.
enc = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=int)
df["proximity_code"] = df["ocean_proximity"].astype("category").cat.codes
print("\n=== Task 5: One-hot columns ===\n", [c for c in enc.columns if c.startswith("ocean_proximity_")])

# --- Step 6: Task 6 - Aggregate ---
print("\n=== Task 6: Avg price by neighbourhood type ===")
# Summary of price and count for each neighbourhood type.
print(df.groupby("ocean_proximity").agg(avg_price=("median_house_value", "mean"),
                                        median_price=("median_house_value", "median"),
                                        properties=("median_house_value", "size")).round(2))
# Group houses into age bands, then average the price in each band.
print("\nAvg price by age band:\n", df.groupby(pd.cut(df["housing_median_age"], [0, 10, 20, 35, 60]),
                                               observed=True)["median_house_value"].mean().round(0))

# --- Step 7: Task 7 - Outliers (IQR capping) ---
# Tukey fences: values outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR count as outliers.
col = "median_house_value"
q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\n=== Task 7: Outliers: {((df[col] < lo) | (df[col] > hi)).sum()} beyond [{lo:,.0f}, {hi:,.0f}] ===")
# Cap extremes at the fences instead of deleting rows.
df[col] = df[col].clip(lo, hi)
print(f"Max after capping: {df[col].max():,.0f}")

"""
============================================================================
CODE EXPLANATION
============================================================================
 1. Imports: pandas for tables.
 2. Load: Option A (default) pd.read_csv(url) fetches the CSV online from the datasets repo. Option B reads a
    local copy: comment out Option A and uncomment the local pd.read_csv line.
 3. Task 1 (clean names): str.strip, str.lower and str.replace (regex) make names lower_case_with_underscores.
 4. Task 2 (missing values): isnull().sum() counts NaN; fillna(median) for numbers and fillna(mode) for text.
 5. Task 3 (merge): pd.DataFrame builds a lookup table; df.merge(on="ocean_proximity", how="left")
    adds flood_risk_score and amenity_index by matching the KEY, not the row position.
 6. Task 4 (filter): boolean masks with & select expensive (> 200000) and new (< 20 years) rows.
 7. Task 5 (encode): pd.get_dummies = one-hot (drop_first=True); astype("category").cat.codes = label encoding.
 8. Task 6 (aggregate): groupby().agg() with named aggregations (mean, median, size) per
    ocean_proximity, and groupby(pd.cut(...)) for age bands.
 9. Task 7 (outliers): quantile() gives Q1/Q3, IQR fences are computed, and clip() caps the extremes.

============================================================================
DETAILED THEORY NOTES
============================================================================
1. What is data wrangling (munging)?
 - Turning raw, messy data into a clean, analysis-ready table.
 - Typical steps: rename columns, fix missing values, merge sources, filter, encode, aggregate, treat outliers.
 - It usually takes 60-80% of a data project's time, so good habits here matter most.

2. Cleaning column names
 - Consistent names (lower case, underscores, no symbols) avoid typing errors like df["Total Rooms "].
 - str.strip removes spaces at the ends; str.replace with a regex removes unwanted characters.

3. Missing values and strategies
 - Missing value = an empty cell, stored as NaN (Not a Number) in pandas.
 - Drop rows (dropna): simple but loses data; fine when only a tiny share is missing.
 - Mean imputation: replaces gaps with the average; sensitive to outliers.
 - Median imputation: replaces gaps with the middle value; robust, so we use it for numbers.
 - Mode imputation: replaces gaps with the most frequent value; used for categories.
 - Forward/backward fill: copies the previous/next value; good for time series.
 - Model-based imputation (KNN, regression): predicts the gap from other columns; powerful but complex.

4. Merge, join and concat
 - merge(): SQL-style join of two tables on key column(s); pandas' main combining tool.
 - join(): shortcut that merges on the INDEX instead of a column.
 - concat(): simply stacks tables on top of each other (axis=0) or side by side (axis=1); no key matching.
 - Inner join: keeps only keys present in BOTH tables.
 - Left join: keeps all rows of the left table, fills NaN where the right has no match (used here).
 - Right join: keeps all rows of the right table.
 - Outer join: keeps all keys from both tables, NaN where there is no match.
 - Always join on a key. Matching by row position can attach the wrong values.

5. Filtering / subsetting
 - Boolean indexing: df[condition] keeps rows where the condition is True.
 - Combine conditions with & (and), | (or), ~ (not); wrap each condition in brackets.

6. Encoding categorical variables
 - Models need numbers, so text categories must be converted.
 - One-hot encoding: one 0/1 column per category; right for nominal data with no order (ocean_proximity).
 - drop_first=True drops one dummy column to avoid the dummy-variable trap (perfect multicollinearity).
 - Label encoding: each category gets an integer code; suitable for ordered data or tree models.
   For unordered data it can wrongly suggest that INLAND (code 1) is "less than" NEAR BAY (code 3).

7. Aggregation (groupby)
 - groupby follows split-apply-combine: split rows by key, apply a function (mean/size), combine the results.
 - agg() allows several summaries at once; named aggregation (new_name=(column, func)) gives readable output.
 - pd.cut() bins a number into ranges (age bands) so numeric data can be grouped like categories.

8. Outliers and the IQR method
 - Outlier = a value far from the rest (error or genuinely extreme case).
 - IQR = Q3 - Q1. Lower fence = Q1 - 1.5*IQR, upper fence = Q3 + 1.5*IQR (Tukey's rule).
 - Treatments: remove, cap/winsorize (clip, used here), transform (log), or keep if genuine.
 - Z-score method flags values more than 3 standard deviations from the mean; it assumes roughly
   normal data and is itself pulled by outliers, whereas IQR is robust.
 - Note: the lower fence here is negative (house values cannot be negative), so it removes nothing at the bottom.

9. The dataset (California housing, 1990 census)
 - 20640 rows, 10 columns; each row is a census block group (a small district), not a single house.
 - longitude, latitude: location of the block group.
 - housing_median_age: median age of houses in years.
 - total_rooms, total_bedrooms: total rooms/bedrooms in the block (total_bedrooms has 207 missing values).
 - population, households: people and households in the block.
 - median_income: median income in tens of thousands of dollars.
 - median_house_value: median house price in dollars; capped at 500001 (censored values).
 - ocean_proximity: category - <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN.

10. Advantages and disadvantages of wrangling
 - Advantages: cleaner and more reliable analysis, better model accuracy, consistent formats, repeatable scripts.
 - Disadvantages: time consuming, imputation and capping can introduce bias or hide real extremes,
   needs domain knowledge, and wrong choices silently distort results.

11. Real-world uses
 - Preparing property data for price prediction (Zillow-style models) and market reports.
 - Banking/insurance risk data cleaning, healthcare record cleaning, retail and telecom analytics.
 - Any machine-learning pipeline: feature cleaning, encoding and scaling before training.

============================================================================
EXPECTED OUTPUT (real numbers; source: the California housing file fetched from the datasets repo)
============================================================================
 Raw shape: (20640, 10)
 Task 2: Missing values {'total_bedrooms': 207}; Nulls after: 0
 Task 3: 5 distinct rows - NEAR BAY 2.5/8, <1H OCEAN 1.2/7, INLAND 0.8/4, NEAR OCEAN 3.1/9, ISLAND 4.0/3
 Task 4: Expensive & new properties: 2186 of 20640 rows; Inland only: 6551 rows
 Task 5: One-hot columns: INLAND, ISLAND, NEAR BAY, NEAR OCEAN (<1H OCEAN dropped as the base)
 Task 6: avg_price - <1H OCEAN 240084.29 (9136 rows), INLAND 124805.39 (6551), ISLAND 380440.00 (5),
         NEAR BAY 259212.31 (2290), NEAR OCEAN 249433.98 (2658)
         Age bands avg price: (0,10] 200263, (10,20] 191182, (20,35] 207057, (35,60] 219905
 Task 7: 1071 outliers beyond [-98,088, 482,412]; max after capping 482,412
 (A pandas warning about select_dtypes("object") may be printed; it is harmless.)

============================================================================
LIBRARIES AND FUNCTIONS USED
============================================================================
 - pandas as pd: read_csv - load CSV from a URL or local file.
 - pd.DataFrame - build the lookup table.
 - df.columns.str.strip/lower/replace - clean names.
 - df.isnull().sum() - count missing values.
 - df.select_dtypes - pick numeric or text columns.
 - Series.fillna, Series.median, Series.mode - fill missing values.
 - df.merge - join the lookup table on a key.
 - drop_duplicates - show unique rows only.
 - Boolean indexing df[mask] - filter rows; len() counts them.
 - pd.get_dummies - one-hot encoding.
 - astype("category").cat.codes - label encoding.
 - groupby().agg - grouped summaries; pd.cut - make age bands.
 - Series.quantile - Q1 and Q3; Series.clip - cap outliers; Series.max - check result.

============================================================================
DETAILED VIVA Q&A
============================================================================
Q1. What is data wrangling and why is it needed?
A1. Data wrangling is converting raw data into a clean, structured format ready for analysis. Real data has
    wrong names, missing cells, mixed sources and extremes, and models give poor results if these are not fixed.

Q2. Difference between merge, join and concat?
A2. merge() combines tables on key columns like an SQL join. join() does the same but on the index. concat()
    does not match keys; it only stacks tables vertically or horizontally.

Q3. Explain inner, left, right and outer joins.
A3. Inner keeps only matching keys; left keeps all left rows; right keeps all right rows; outer keeps all rows
    from both. Unmatched cells are filled with NaN. We used a left join so no property row is lost.

Q4. Why use the median instead of the mean for imputation?
A4. The mean is pulled by extreme values, but the median is the middle value and ignores them. House prices and
    rooms are skewed, so the median is a safer fill value. For text columns we use the mode.

Q5. One-hot encoding versus label encoding?
A5. One-hot makes a separate 0/1 column per category and implies no order, so it suits nominal data. Label encoding
    assigns integers 0,1,2..., which implies an order and is only safe for ordinal data or tree-based models.

Q6. What is the dummy-variable trap and what does drop_first do?
A6. If all dummy columns are kept, one column can be predicted from the others (they sum to 1), which causes
    perfect multicollinearity in linear models. drop_first=True removes one column to avoid it.

Q7. How does the IQR method detect outliers?
A7. Compute Q1 and Q3, then IQR = Q3 - Q1. Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers. It uses
    percentiles, so it is not distorted by the outliers themselves, unlike mean/standard-deviation methods.

Q8. IQR versus z-score for outliers?
A8. Z-score = (x - mean)/std and flags |z| > 3; it works best for roughly normal data. IQR makes no normality
    assumption and is robust for skewed data like house prices.

Q9. What does groupby().agg() do?
A9. groupby splits the data into groups by a key, agg applies one or more functions (mean, median, size) to each
    group, and pandas combines the results into a new table. This is the split-apply-combine pattern.

Q10. What is missing data and what are its types?
A10. Missing data means absent values (NaN). MCAR means missing completely at random, MAR means it depends on
     other observed columns, and MNAR means it depends on the missing value itself. The type decides the best fix.

============================================================================
PRACTICAL AND CODING VIVA QUESTIONS (about this exact code)
============================================================================
Q1. Why do we merge on "ocean_proximity" and not by row position?
A1. Position matching can attach the wrong flood score to a row if the order differs. Merging on the key matches
    each row to the correct lookup row, and how="left" keeps all 20640 original rows.

Q2. Why are "&" and brackets used in the filter on line "subset = df[...]"?
A2. Python's "and" cannot work on whole columns, so pandas uses the element-wise "&". The brackets are needed
    because & binds tighter than comparison operators like > and <.

Q3. Why is the lower IQR fence negative, and what does clip do with it?
A3. Q1 - 1.5*IQR falls below zero for this skewed data, and no price is below it, so nothing is capped at the
    bottom. clip() only lowers the high prices to the upper fence 482,412 (1071 rows were beyond the fences).

Q4. Why does "<1H OCEAN" not appear among the one-hot columns?
A4. drop_first=True drops the first category alphabetically, which is "<1H OCEAN". Its rows are the ones where all
    four remaining dummy columns are 0, so no information is lost.

Q5. Why do we use pd.cut with observed=True in the age band grouping?
A5. pd.cut turns housing_median_age into categorical bands (0,10], (10,20], (20,35], (35,60]. observed=True
    makes groupby show only bands that actually contain rows and avoids a warning.

Q6. How do you switch between online and local data?
A6. Option A reads the CSV from the datasets repo URL with pd.read_csv(url). To use a local copy instead, comment
    out Option A and uncomment Option B, which calls pd.read_csv on the file kept next to the script.

END OF VIVA NOTES
"""
