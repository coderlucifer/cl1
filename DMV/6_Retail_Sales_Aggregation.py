"""
Assignment 6 : Data Aggregation - Sales performance by region in a retail company
=================================================================================
INSTRUCTIONS
 1. Install:  pip install pandas matplotlib
 2. Dataset : Retail_Sales_Data.csv = Kaggle Superstore sales data (vivek468, "Superstore Dataset Final")
              reduced to 5 columns: TransactionDate, Region, ProductCategory, Quantity, SalesAmount
              (9994 rows, amounts in US dollars). Option A (default): fetch it online from the
              datasets repo. Option B: load a local copy kept next to this script (comment out
              Option A and uncomment the Option B line).
 3. Run     : python 6_Retail_Sales_Aggregation.py
 4. Steps   : load -> explore -> pick variables -> total sales by region -> bar/pie
              -> top regions -> region x category totals -> stacked and grouped bar plots
"""
# --- Step 0: Imports and setup ---
import matplotlib.pyplot as plt
import pandas as pd

# --- Step 1: Task 1 - Import ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/Retail_Sales_Data.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("Retail_Sales_Data.csv")

# --- Step 2: Task 2 - Explore ---
print("=== Task 2: Info & sample ===")
# Column types, first rows, and total missing cells.
df.info(); print(df.head(3)); print("\nMissing:", int(df.isnull().sum().sum()))
# Clean first: missing keys would make the totals wrong (no transaction id, so no duplicate drop).
df = df.dropna(subset=["Region", "SalesAmount"])
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

# --- Step 3: Task 3 - Choose variables ---
# Dimensions = the columns we group BY; metric = the number we add up.
print("\n=== Task 3: Dimensions: 'Region', 'ProductCategory' | Metric: 'SalesAmount' ===")

# --- Step 4: Task 4 - Total sales by region ---
# Total sales per region, largest first.
region_sales = df.groupby("Region")["SalesAmount"].sum().sort_values(ascending=False)
print("\n=== Task 4: Total sales by region ===")
for r, t in region_sales.items():
    print(f" {r:<8}: {t:>12,.2f}")
# Total, average sale and number of sales per region.
print("\nFull summary:\n", df.groupby("Region")["SalesAmount"].agg(["sum", "mean", "count"]).round(2))

# --- Step 5: Task 6 - Top performers ---
print("\n=== Task 6: Top-performing regions ===")
# After sorting, the first item is the best region; share = its total / grand total.
print(f"Best: {region_sales.index[0]} ({region_sales.iloc[0]:,.2f}) | share {region_sales.iloc[0] / region_sales.sum():.1%}")
print("Top 2:", list(region_sales.head(2).index))

# --- Step 6: Task 7 - Region x category ---
# Region x category totals as a matrix, rows ordered best region first, empty cells = 0.
rc = df.groupby(["Region", "ProductCategory"])["SalesAmount"].sum().unstack(fill_value=0).loc[region_sales.index]
print("\n=== Task 7: Region x Category matrix ===\n", rc)
print("\nBest category in each region:\n", rc.idxmax(axis=1))

# --- Step 7: Tasks 5 & 8 - Plots ---
# One figure with four plots side by side.
fig, ax = plt.subplots(1, 4, figsize=(21, 5))
# Bar chart: total sales per region, good for ranking.
region_sales.plot(kind="bar", ax=ax[0], color="#2b5c8f", edgecolor="black", rot=0)
ax[0].set(title="Total sales by region", ylabel="Sales amount")
# Pie chart: each region's share of total sales (only 4 slices, so it stays readable).
ax[1].pie(region_sales, labels=region_sales.index, autopct="%1.1f%%", startangle=140)
ax[1].set_title("Regional share (%)")
# Stacked bar: total per region and how each category contributes.
rc.plot(kind="bar", stacked=True, ax=ax[2], colormap="viridis", edgecolor="black", rot=0)
ax[2].set(title="Region x category (stacked)", ylabel="Sales amount"); ax[2].legend(title="Category", fontsize=8)
# Grouped bar: categories side by side within each region.
rc.plot(kind="bar", ax=ax[3], colormap="viridis", edgecolor="black", rot=0)
ax[3].set(title="Region x category (grouped)"); ax[3].legend(title="Category", fontsize=8)
# Light horizontal grid on the bar charts only (a pie has no axes).
for a in (ax[0], ax[2], ax[3]):
    a.grid(axis="y", ls=":", alpha=.6)
plt.tight_layout(); plt.show()

"""
============================================================================
CODE EXPLANATION
============================================================================
 1. Imports: matplotlib.pyplot (charts), pandas (aggregation).
 2. Task 1 (import): Option A: pd.read_csv(url) fetches Retail_Sales_Data.csv from the datasets repo (needs
    internet). Option B: comment out Option A and uncomment pd.read_csv("Retail_Sales_Data.csv") to load a local copy.
 3. Task 2 (explore): info(), head(3) and isnull().sum() inspect the data; dropna removes rows with a missing
    Region or SalesAmount; to_datetime converts dates. Duplicates are kept (no transaction id).
 4. Task 3 (variables): dimensions = Region and ProductCategory; metric = SalesAmount.
 5. Task 4 (totals): groupby("Region")["SalesAmount"].sum() then sort_values(ascending=False); agg(["sum","mean","count"])
    gives a fuller summary.
 6. Task 6 (top performers): index[0], iloc[0] and head(2) pick the best and top-2 regions; share = best / total.
 7. Task 7 (region x category): groupby on two keys + unstack(fill_value=0) builds a matrix; idxmax(axis=1) finds
    the best category in each region.
 8. Tasks 5 and 8 (plots): plt.subplots(1, 4) holds a bar chart, a pie chart, a stacked bar and a grouped bar.
    pandas .plot(kind="bar") and ax.pie() draw them; legend, set and grid customise them.

============================================================================
DETAILED THEORY NOTES
============================================================================
1. What is data aggregation?
 - Collapsing many detailed rows into fewer summary rows using functions like sum, mean, count, min, max.
 - Answers questions such as "total sales per region" instead of showing every transaction.
 - In SQL it is GROUP BY; in pandas it is groupby().

2. groupby: split-apply-combine
 - Split: divide rows into groups by one or more keys (Region).
 - Apply: run a function on each group (sum of SalesAmount).
 - Combine: join the group results into a new table or Series.
 - Grouping by two keys (Region, ProductCategory) gives one result per combination.

3. agg and common aggregate functions
 - sum = total, mean = average, count = number of non-null values, size = number of rows, min/max = extremes.
 - agg() applies several functions at once (here sum, mean, count).
 - sum vs mean as a KPI: sum shows total business size; mean shows the typical size of one sale.

4. Pivot, unstack and pivot_table
 - unstack() moves one index level of a grouped result into columns, giving a cross-table (Region x Category).
 - pd.pivot_table(df, index="Region", columns="ProductCategory", values="SalesAmount", aggfunc="sum")
   gives the same result in a single call and also supports margins (totals).
 - fill_value=0 replaces combinations with no sales so charts do not break.

5. Ranking top performers
 - sort_values(ascending=False) orders the totals; head(n) takes the top n; idxmax() returns the name of the maximum.
 - Share of total = region total / grand total, shown as a percentage.

6. Chart types and when to use each
 - Bar chart: compare totals across categories and rank them (regions).
 - Pie chart: share of a whole; use only with few slices (4 here) and when parts add to 100%.
 - Stacked bar chart: total per group AND how it splits into categories; hard to compare inner segments.
 - Grouped (clustered) bar chart: compare categories side by side within each group; easy to compare, but wider.
 - Line chart: trend over time (e.g. sales per month), not used in this script.

7. The dataset
 - Retail_Sales_Data.csv: the Kaggle Superstore sample (vivek468) reduced to 5 columns; one row per order line,
   9994 rows, no missing values.
 - TransactionDate: date of sale; Region: West/East/Central/South; ProductCategory: Furniture/Office Supplies/Technology;
   Quantity: units sold; SalesAmount: revenue in US dollars.

8. Data quality before aggregating
 - Missing keys (Region) or amounts (SalesAmount) give wrong totals, so dropna runs first.
   Identical-looking rows are kept because there is no transaction id, so they are real sales.
 - Dates must be converted with to_datetime before any time-based grouping.

9. Advantages and disadvantages of aggregation
 - Advantages: simple summaries for decisions, faster reports, less data to chart, reveals patterns hidden in rows.
 - Disadvantages: loses detail (a region total hides which store or day drove it); averages can hide spread;
   Simpson's paradox can appear when groups are combined.

10. Real-world uses
 - Sales and revenue reports by region/product/month, inventory planning, KPI dashboards (Power BI, Tableau).
 - Banking (spend by customer segment), healthcare (cases per district), web analytics (visits per page).

============================================================================
EXPECTED OUTPUT (real numbers from the Kaggle Superstore data; amounts in US dollars)
============================================================================
 Info: 9994 rows, 5 columns, Missing: 0. Sample rows: 2016-11-08 South Furniture 2 261.96,
 2016-11-08 South Furniture 3 731.94, 2016-06-12 West Office Supplies 2 14.62.
 Task 4 total sales by region: West 725,457.82 | East 678,781.24 | Central 501,239.89 | South 391,721.91
 Summary (sum / mean / count): Central 501239.89 / 215.77 / 2323; East 678781.24 / 238.34 / 2848;
                               South 391721.90 / 241.80 / 1620; West 725457.82 / 226.49 / 3203
 Task 6: Best: West (725,457.82) share 31.6%; Top 2: ['West', 'East']
 Task 7 matrix (rows West, East, Central, South; columns Furniture, Office Supplies, Technology):
   West 252612.74 220853.25 251991.83 | East 208291.20 205516.06 264973.98
   Central 163797.16 167026.42 170416.31 | South 117298.68 125651.31 148771.91
 Best category per region: West Furniture, East Technology, Central Technology, South Technology
 One figure with 4 plots; with MPLBACKEND=Agg only a harmless "non-interactive" warning appears.
 Note: South has the fewest sales (1620) and the lowest total, but the highest mean sale (241.80).

============================================================================
LIBRARIES AND FUNCTIONS USED
============================================================================
 - pandas as pd: read_csv, to_datetime.
 - df.info, df.head, df.isnull().sum - explore data.
 - df.dropna - clean data.
 - df.groupby(...)[col].sum / agg - aggregation.
 - Series.sort_values, head, items, index, iloc - rank and read results.
 - unstack(fill_value=0) and .loc - build the Region x Category matrix in region order.
 - DataFrame.idxmax(axis=1) - best category per region.
 - matplotlib.pyplot as plt: subplots, tight_layout, show.
 - Axes/pandas plotting: .plot(kind="bar", stacked=True), ax.pie, set, set_title, legend, grid.

============================================================================
DETAILED VIVA Q&A
============================================================================
Q1. What is data aggregation?
A1. Aggregation summarises many rows into fewer ones using functions such as sum, mean and count. For example,
    thousands of transactions become one total per region, which is easier to understand and decide from.

Q2. Explain split-apply-combine.
A2. groupby splits the data into groups by a key, applies a function to each group, and combines the results into
    one table. Our code splits by Region, applies sum to SalesAmount and gets four totals.

Q3. Difference between groupby and pivot_table?
A3. groupby gives a grouped result as a Series or long table; pivot_table builds a cross-table directly with one
    key on rows and another on columns, and supports margins. groupby + unstack gives the same matrix.

Q4. What does unstack do?
A4. It moves the innermost index level of a grouped result into columns. A (Region, ProductCategory) index
    becomes Region rows and ProductCategory columns, which is ideal for stacked bar charts.

Q5. Sum versus mean as a KPI?
A5. Sum measures overall performance such as total revenue. Mean measures the typical size of one sale. A region
    can have a high total just because it has many sales, so we show sum, mean and count together.

Q6. When do you use a pie chart and when not?
A6. Use a pie to show parts of a whole with few slices (here 4 regions). Avoid it with many slices or similar-size
    values because angles are hard to compare; a bar chart is better.

Q7. Stacked versus grouped bar chart?
A7. Stacked shows each group's total and its composition but makes inner segments hard to compare. Grouped places
    bars side by side so categories are easy to compare, but the group total is not visible.

Q8. Why clean data before aggregating?
A8. Missing region or amount values silently change totals and counts, so dropna removes them. We do not
    drop duplicates: with no transaction id, identical-looking rows are real separate sales.

Q9. Difference between count and size in pandas?
A9. count() counts non-null values in a column; size counts all rows in a group including those with NaN. Here
    SalesAmount has no NaN after cleaning, so they agree.

Q10. How do you find the top-n performers?
A10. Sort the aggregated Series with sort_values(ascending=False) and take head(n); idxmax gives the single best
     name. We also compute the best region's percentage share of total sales.

============================================================================
PRACTICAL AND CODING VIVA QUESTIONS (about this exact code)
============================================================================
Q1. How do you switch between online and local data?
A1. Option A (active) reads the CSV from the datasets repo URL with pd.read_csv(url). For Option B, comment out
    Option A and uncomment pd.read_csv("Retail_Sales_Data.csv"), which reads a local copy next to the script.

Q2. Why is .loc[region_sales.index] added after unstack?
A2. unstack sorts regions alphabetically. Using .loc with the sorted region_sales index reorders the matrix rows to
    best-first (West, East, Central, South) so the stacked and grouped bars follow the same ranking as the first chart.

Q3. Why fill_value=0 in unstack?
A3. If a region had no sales for a category the cell would be NaN, which breaks sums and leaves gaps in bars.
    fill_value=0 shows it as zero sales instead.

Q4. How is the share of the best region calculated?
A4. region_sales.iloc[0] / region_sales.sum() divides the best region's total (West) by the grand total, and the format :.1%
    prints it as a percentage (31.6% for the real data).

Q5. What does rc.idxmax(axis=1) return?
A5. For each row (region) it returns the column label with the maximum value, so it shows the best-selling category
    in each region (for example West -> Furniture, East -> Technology).

Q6. Why do only three of the four axes get a grid?
A6. The loop applies grid(axis="y") to the three bar charts (ax[0], ax[2], ax[3]). The pie chart has no x/y axes, so
    a grid would be meaningless.

Q7. Why does region_sales.plot(kind="bar", ax=ax[0]) work?
A7. A pandas Series/DataFrame has a plot method built on matplotlib. Passing ax=ax[0] tells pandas to draw into
    that subplot of our figure instead of creating a new one.

END OF VIVA NOTES
"""
