"""
Assignment 3 : Data Cleaning and Preparation - Telecom customer churn
=====================================================================
INSTRUCTIONS
 1. Install:  pip install pandas numpy scikit-learn
 2. Dataset : Telecom_Customer_Churn.csv (Telco customer churn, Kaggle blastchar/telco-customer-churn).
              Option A (default): fetched online from the datasets repo.
              Option B: a local copy next to this script (uncomment that line, comment Option A).
 3. Run     : python 3_Telecom_Churn_Cleaning.py
 4. Steps   : import -> explore -> missing values -> duplicates -> inconsistent text -> dtypes
              -> outliers (IQR) -> feature engineering -> encode + scale -> train/test split -> export
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Task 1: Import ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/Telecom_Customer_Churn.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("Telecom_Customer_Churn.csv")

# --- Task 2: Explore ---
print("=== Task 1 & 2: Structure ===")
# Size, column types, a few rows and summary statistics.
print("Shape:", df.shape); df.info()
print(df.head(3)); print("\nSummary:\n", df.describe().round(2))

# --- Task 3: Missing values ---
# TotalCharges has blank strings that isnull() cannot see, so convert them to NaN first.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\n=== Task 3: Missing values ===\n", df.isnull().sum()[df.isnull().sum() > 0])
# New customers (tenure 0) have paid nothing yet, so tenure x MonthlyCharges is correct.
df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
print("Nulls left:", int(df.isnull().sum().sum()))

# --- Task 4: Duplicates ---
# Keep one row per customer id so nobody is counted twice.
before = len(df); df = df.drop_duplicates(subset="customerID")
print(f"\n=== Task 4: Duplicates removed: {before - len(df)} ===")

# --- Task 5: Inconsistent data (spelling / spacing / case / equivalent labels) ---
# Strip stray spaces from every text column except the id.
for c in df.select_dtypes("object").columns.drop("customerID"):
    df[c] = df[c].str.strip()
# These labels mean the same as "No", so merge them to cut categories.
df = df.replace({"No internet service": "No", "No phone service": "No"})
print("\n=== Task 5: Standardised labels ===")
print(df["Contract"].value_counts())

# --- Task 6: Data types ---
# SeniorCitizen is a Yes/No group, not a number; Churn becomes 1/0 for models.
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("\n=== Task 6: dtypes ===\n", df.dtypes.value_counts())

# --- Task 7: Outliers (IQR capping / winsorising) ---
print("\n=== Task 7: Outliers (IQR) ===")
for col in ["MonthlyCharges", "TotalCharges"]:
    # IQR rule: outliers lie beyond 1.5 x IQR from the quartiles.
    q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"{col}: {((df[col] < lo) | (df[col] > hi)).sum()} outliers (limits {lo:.1f} .. {hi:.1f})")
    # Cap extremes at the limits instead of deleting rows.
    df[col] = df[col].clip(lo, hi)

# --- Task 8: Feature engineering ---
# Average monthly spend; replace(0, 1) avoids dividing by zero for new customers.
df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, 1), df["MonthlyCharges"])
# Bin tenure (months) into year groups; -1 includes tenure 0.
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-1y", "1-2y", "2-4y", "4-6y"])
print("\n=== Task 8: Engineered features ===")
print(df[["tenure", "TenureGroup", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "Churn"]].head())

# --- Task 9: Encode + scale ---
# One-hot encode text columns; drop the id and target, and one dummy per category.
X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True, dtype=int)
y = df["Churn"]
num = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
# Task 10: split BEFORE scaling so test data does not leak into the scaler.
# stratify=y keeps the same churn ratio in train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Learn mean/std from train only, then apply the same scaling to both parts.
sc = StandardScaler().fit(X_train[num])
X_train[num], X_test[num] = sc.transform(X_train[num]), sc.transform(X_test[num])
print("\n=== Task 9: Scaled train columns (mean 0, std 1) ===")
print(X_train[num].describe().round(2).T[["mean", "std", "min", "max"]])
print("\n=== Task 10: Split ===")
print("Train:", X_train.shape, y_train.shape, "| Test:", X_test.shape, y_test.shape)
# Churn is 0/1, so its mean is the churn rate.
print("Churn rate train/test: %.3f / %.3f" % (y_train.mean(), y_test.mean()))

# --- Task 11: Export ---
out = "Cleaned_Telecom_Customer_Churn.csv"
df.to_csv(out, index=False)
print("\n=== Task 11: Saved ->", out)

"""
=======================================================================================
CODE EXPLANATION
=======================================================================================
1. Task 1 (import): Option A (default) fetches the CSV online from the datasets repo with
   pd.read_csv(url). Option B reads a local copy: comment out Option A and uncomment the
   local pd.read_csv line.
2. Task 2 (explore): .shape, .info(), .head(), .describe() reveal size, types and statistics.
3. Task 3 (missing): pd.to_numeric(errors="coerce") converts blank " " strings in
   TotalCharges to NaN; isnull().sum() counts them; fillna(tenure * MonthlyCharges) fills them.
4. Task 4 (duplicates): drop_duplicates(subset="customerID") keeps one row per customer.
5. Task 5 (inconsistent data): str.strip() removes spaces; replace() merges "No internet
   service" and "No phone service" into "No"; value_counts() verifies labels.
6. Task 6 (dtypes): astype("category") for SeniorCitizen, map({"Yes":1,"No":0}) for Churn.
7. Task 7 (outliers): quantile() gives Q1/Q3, the IQR rule gives limits, clip() caps values.
8. Task 8 (feature engineering): np.where for AvgMonthlySpend, pd.cut for TenureGroup.
9. Task 9 (encode and scale): pd.get_dummies(drop_first=True) one-hot encodes; StandardScaler
   scales the numeric columns.
10. Task 10 (split): train_test_split(test_size=0.2, stratify=y, random_state=42) BEFORE the
    scaler is fitted, so only training data decides the mean and std.
11. Task 11 (export): to_csv() saves Cleaned_Telecom_Customer_Churn.csv.

=======================================================================================
DETAILED THEORY NOTES
=======================================================================================
1. Data cleaning vs data preparation
   - Data cleaning = fixing errors in the data: missing values, duplicates, wrong types,
     inconsistent text, outliers.
   - Data preparation (preprocessing) = shaping clean data for a model: feature
     engineering, encoding, scaling, splitting.
   - Why: "garbage in, garbage out". Models and statistics are only as good as the data.

2. Missing-value strategies
   - Drop rows/columns (dropna): easy but loses data.
   - Mean imputation: for symmetric numeric data. Median imputation: robust to outliers.
   - Mode: most frequent value for categories. Constant/domain fill: use business logic
     (we use tenure x MonthlyCharges because new customers have paid nothing yet).
   - Forward/backward fill for time series; KNN/model imputation for complex cases.
   - Hidden missing values: blanks " ", "?", "N/A" or 0 look valid; isnull() cannot see them.

3. Duplicates
   - Repeated records inflate counts and bias a model (the same customer is seen twice,
     maybe in both train and test). Use a unique key (customerID) to detect them.

4. Inconsistent data
   - Extra spaces, different case, spelling variants or labels meaning the same thing
     create false categories. Fix with strip(), lower(), replace(). Merging
     "No internet service" with "No" removes redundancy.

5. Data types
   - Numbers stored as text cannot be averaged. Categories stored as ints (0/1) can be
     misread as numeric. Convert to the right dtype: numeric, category, datetime.

6. Outliers
   - An outlier is a value far from the rest. It can be an error or a real rare value.
   - IQR method: IQR = Q3 - Q1; outliers are < Q1 - 1.5*IQR or > Q3 + 1.5*IQR. Works for
     skewed data and does not assume a distribution.
   - Z-score method: z = (x - mean) / std; values with |z| > 3 are outliers. Assumes data
     is roughly normal.
   - Treatment: remove, cap/winsorise (clip), transform (log) or keep. We capped.

7. Feature engineering
   - Creating new useful columns from existing ones. Examples here: AvgMonthlySpend
     (total / tenure) and TenureGroup (binning tenure in months into year groups).
   - Binning turns a number into ordered groups and can reveal non-linear patterns.

8. Encoding categorical data
   - Models need numbers. Label encoding gives each category an integer (0,1,2); good for
     ordered categories or tree models but implies a false order for nominal data.
   - One-hot encoding creates one 0/1 column per category; no false order but adds columns.
   - drop_first=True removes one dummy column to avoid the dummy-variable trap
     (perfect multicollinearity: one column can be predicted from the others).

9. Scaling
   - Standard scaling (z-score): (x - mean) / std gives mean 0, std 1. Good for most models.
   - Min-max scaling: (x - min) / (max - min) maps to 0..1; sensitive to outliers.
   - Needed for distance/gradient based models (KNN, SVM, logistic regression, neural nets);
     not needed for tree models.

10. Train/test split and data leakage
   - Split: train data fits the model, test data checks it on unseen data.
   - Data leakage = information from the test set (or the future) leaking into training,
     which gives over-optimistic results. Fitting a scaler on the full data is a leak, so
     we split first and fit the scaler on train only.
   - stratify=y keeps the same class ratio (about 26.5% churn) in both parts, important for
     imbalanced data. random_state makes the split repeatable.

11. Dataset description (Telco Customer Churn)
   - 7043 customers and 21 columns: customerID, gender, SeniorCitizen, Partner,
     Dependents, tenure (months with the company), PhoneService, MultipleLines,
     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
     StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
     MonthlyCharges, TotalCharges, Churn (Yes = the customer left).
   - Quirk: TotalCharges is text because 11 new customers (tenure 0) have a blank " ".
   - Churn is imbalanced: about 26.5% Yes.

12. Advantages / disadvantages of the cleaning choices
   - Clipping outliers: + keeps all rows; - changes real values and may hide rare behaviour.
   - Median/domain imputation: + keeps rows; - can reduce variance or add bias if the rule is wrong.
   - One-hot encoding: + simple and safe; - many columns for high-cardinality features.
   - Standard scaling: + comparable features; - needs the same scaler at prediction time.

13. Real-world uses
   - Telecom churn prediction to target retention offers; banking credit scoring;
     healthcare records cleaning; e-commerce customer segmentation; any ML pipeline.

=======================================================================================
LIBRARIES AND FUNCTIONS USED
=======================================================================================
- numpy (np): where().
- pandas (pd): read_csv(), info(), head(), describe(), to_numeric(),
  isnull().sum(), fillna(), drop_duplicates(), select_dtypes(), str.strip(), replace(),
  value_counts(), astype(), map(), quantile(), clip(), cut(), get_dummies(), to_csv().
- sklearn.model_selection.train_test_split: splits into train and test with stratify.
- sklearn.preprocessing.StandardScaler: fit() learns mean/std, transform() applies scaling.

=======================================================================================
EXPECTED OUTPUT (real run; source = the Telco CSV fetched from the datasets repo, Option A)
=======================================================================================
- Shape (7043, 21). TotalCharges is text (object/str) at first.
- Missing values: TotalCharges 11 -> after fillna, Nulls left 0.
- Duplicates removed: 0 (the real data has unique customerIDs).
- Contract counts: Month-to-month 3875, Two year 1695, One year 1473.
- Outliers (IQR): MonthlyCharges 0 (limits -46.0 .. 171.4); TotalCharges 0 (limits
  -4683.5 .. 8868.7), so clip() changes nothing on this data.
- Engineered rows e.g. tenure 1 -> TenureGroup 0-1y, AvgMonthlySpend 29.85.
- Scaled train columns: mean 0.0 and std 1.0 for tenure, MonthlyCharges, TotalCharges,
  AvgMonthlySpend.
- Split: Train (5634, 27) and Test (1409, 27); churn rate train/test 0.265 / 0.265.
- Cleaned_Telecom_Customer_Churn.csv is saved in the same folder.

=======================================================================================
DETAILED VIVA Q&A
=======================================================================================
Q1. What is the difference between data cleaning and data preparation?
A1. Cleaning removes errors such as missing values, duplicates, wrong types and outliers.
    Preparation reshapes the clean data for modelling by creating features, encoding
    categories, scaling numbers and splitting into train and test sets.

Q2. How do you handle missing values?
A2. First find them (isnull().sum()) and also look for hidden ones like blank strings. Then
    drop, or impute with mean, median, mode or a rule based on domain knowledge. We used
    tenure x MonthlyCharges because the missing customers are new and have paid 0.

Q3. Mean vs median imputation?
A3. The mean is fine for symmetric data but is pulled by outliers. The median is the middle
    value and stays stable when data is skewed or has extreme values, so it is safer.

Q4. Why is isnull() not enough for TotalCharges?
A4. The missing entries are blank strings " " and not NaN, so isnull() says there are none.
    pd.to_numeric(errors="coerce") turns those blanks into NaN so they can be counted and filled.

Q5. How does the IQR method detect outliers? How is it different from z-score?
A5. IQR = Q3 - Q1 and values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR are outliers. It is
    robust and works for skewed data. The z-score flags |z| > 3 using mean and std, so it
    assumes a roughly normal distribution and is affected by the outliers themselves.

Q6. Label encoding vs one-hot encoding?
A6. Label encoding replaces categories with integers, which suits ordered categories but
    creates a false order for nominal ones. One-hot encoding makes a 0/1 column per
    category, avoiding false order but increasing the number of columns.

Q7. Standard scaling vs min-max scaling?
A7. Standard scaling gives mean 0 and std 1 using (x - mean)/std. Min-max maps values to
    0-1 using (x - min)/(max - min) and is more sensitive to outliers.

Q8. What is data leakage and how do you avoid it?
A8. It is when information from the test data (or the future) is used during training, giving
    unrealistically good results. We avoid it by splitting first and fitting the scaler only
    on the training data, then applying it to the test data.

Q9. Why use stratify=y?
A9. The churn classes are imbalanced (about 26.5% churn). Stratification keeps the same
    churn ratio in train and test, so the evaluation is fair and representative.

Q10. What is the dummy-variable trap and what does drop_first=True do?
A10. If all dummy columns of one category are kept, one column can be predicted from the
     others (perfect multicollinearity). drop_first=True removes one column per category to
     avoid this.

=======================================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=======================================================================================
Q1. Why do we fill the missing TotalCharges with tenure * MonthlyCharges?
A1. The 11 missing rows are customers with tenure 0 who have not been billed yet, so the
    formula gives 0, which is the logically correct total, better than a mean.

Q2. What does clip(lo, hi) do and why not delete outlier rows?
A2. clip() replaces values below lo with lo and above hi with hi. Deleting rows would lose
    customers' other information, while capping keeps every row.

Q3. Why is replace(0, 1) used in AvgMonthlySpend?
A3. To avoid dividing by zero for customers with tenure 0. np.where evaluates both branches,
    so the denominator must never be 0 even where the result is not used.

Q4. Why is the scaler fitted on X_train[num] and not on X?
A4. Fitting on all rows would let the test set influence the mean and std, which is data
    leakage. We fit on train only and use transform() on both parts.

Q5. What does pd.cut(tenure, [-1, 12, 24, 48, 72], labels=...) do?
A5. It bins tenure in months into four groups: -1 to 12 (0-1y), 12 to 24 (1-2y), 24 to 48
    (2-4y) and 48 to 72 (4-6y). The lower edge -1 is used so tenure 0 is included.

Q6. Why do we drop customerID and Churn before get_dummies?
A6. customerID is a unique identifier with no predictive meaning, and Churn is the target;
    including the target in the features would be leakage.

Q7. Why is the result 27 columns in X?
A7. The 21 feature columns (19 original after dropping customerID and Churn, plus 2
    engineered: AvgMonthlySpend and TenureGroup) are expanded by get_dummies for text and category columns, with drop_first removing one
    dummy per category. For the real data that gives 27 columns.

Q8. How do you save the cleaned file and why index=False?
A8. df.to_csv(path, index=False). index=False avoids adding an unwanted extra column of
    row numbers to the file.

END OF VIVA NOTES
"""
