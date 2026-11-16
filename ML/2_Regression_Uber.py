"""
Practical 2 : Regression Analysis - Uber fare prediction (Linear, Ridge, Lasso)
===============================================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn
 2. Dataset : uber.csv.
              Option A (active): fetch it online from the datasets repo.
              Option B: load a local copy kept next to this script (comment out A, uncomment B).
 3. Run     : python 2_Regression_Uber.py
 4. Tasks   : pre-process -> outliers -> correlation -> Linear/Ridge/Lasso -> compare R2, RMSE
"""
# --- Step 0: Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Load the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/uber.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("uber.csv")
print(df.head(), "\nShape:", df.shape)

# --- Step 2: Pre-process (clean the data and build features) ---
# Drop id columns and rows with missing values (models cannot use NaN)
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
print("\nMissing values:\n", df.isnull().sum())
df = df.dropna()
# Turn the date text into hour and year numbers, then drop the raw date
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["hour"] = df["pickup_datetime"].dt.hour
df["year"] = df["pickup_datetime"].dt.year
df = df.drop(columns="pickup_datetime").dropna()

# distance feature (haversine, km) - far more useful than raw lat/long
# Haversine needs radians; it gives the real trip distance in km (6371 = Earth radius)
lat1, lon1, lat2, lon2 = map(np.radians, [df.pickup_latitude, df.pickup_longitude,
                                          df.dropoff_latitude, df.dropoff_longitude])
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

# --- Step 3: Outliers (box plot + IQR rule on fare, hard limits on the rest) ---
# Box plot first: dots beyond the whiskers are outliers
df.boxplot(column=["fare_amount", "distance_km"]); plt.title("Before outlier removal"); plt.show()
q1, q3 = df.fare_amount.quantile([.25, .75]); iqr = q3 - q1
# Keep fares inside 1.5*IQR of the quartiles, plus simple sanity limits on the other columns
mask = (df.fare_amount.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr) & (df.fare_amount > 0)
        & df.passenger_count.between(1, 6) & df.distance_km.between(0.1, 100)
        & df.pickup_latitude.between(-90, 90) & df.dropoff_latitude.between(-90, 90))
print("\nRows removed as outliers:", (~mask).sum())
df = df[mask]

# --- Step 4: Correlation ---
# See which columns move with fare, as numbers and as a heat map
corr = df.corr(numeric_only=True)
print("\nCorrelation with fare:\n", corr["fare_amount"].sort_values(ascending=False))
plt.imshow(corr, cmap="coolwarm"); plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation matrix"); plt.tight_layout(); plt.show()

# --- Step 5: Models (split data and define the 3 regressors) ---
# X = inputs, y = fare to predict; hold out 20% of rows for testing
X, y = df.drop(columns="fare_amount"), df["fare_amount"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)  # same split every run
# alpha = penalty strength; Linear has none
models = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=1.0), "Lasso": Lasso(alpha=0.01)}

# --- Step 6: Evaluate (train each model, score on unseen test data) ---
# Train each model and score it on the test set (R2 and RMSE in dollars)
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr); p = m.predict(X_te)
    rows.append([name, r2_score(y_te, p), np.sqrt(mean_squared_error(y_te, p))])
print("\n", pd.DataFrame(rows, columns=["Model", "R2", "RMSE"]).round(4).to_string(index=False))

"""
=====================================================================
CODE EXPLANATION
=====================================================================
1. Imports: numpy/pandas for data, matplotlib for plots, and scikit-learn for the split,
   the three regression models and the metrics.
2. Load: Option A (active) read_csv(url) fetches uber.csv online from the datasets repo. Option B is a
   commented line, read_csv("uber.csv"), for a local copy next to the script; uncomment it and comment
   out Option A to switch.
3. Pre-process: drop id columns, drop rows with missing values, convert pickup_datetime with
   to_datetime(), extract hour and year, then drop the raw date column.
4. Feature engineering: the Haversine formula (numpy sin/cos/arcsin/sqrt) turns four
   latitude/longitude columns into one meaningful number, distance_km.
5. Outliers: a box plot shows them; then the IQR rule (quantile(), between()) plus common-sense
   limits (fare > 0, 1-6 passengers, distance 0.1-100 km, valid latitude) removes bad rows.
6. Correlation: df.corr() and an imshow() heat map show which features relate to fare.
7. Split: train_test_split() with test_size=0.2 and random_state=0 (80% train, 20% test).
8. Models: LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.01) are trained with fit().
9. Evaluate: predict() on the test set; r2_score() and sqrt(mean_squared_error()) (RMSE) are
   collected in a table and printed.

=====================================================================
EXPECTED OUTPUT (real run, REAL uber.csv downloaded from the datasets repo)
=====================================================================
 Shape: (200000, 9)
 Rows removed as outliers: 24281
 Correlation with fare: distance_km 0.725, year 0.136, passenger_count 0.013, hour -0.014,
   raw coordinates about 0.00
 Model    R2      RMSE
 Linear   0.4895  2.9541
 Ridge    0.4895  2.9542
 Lasso    0.4844  2.9689
 Reading it: R2 is only about 0.49 because fare depends on more than straight-line distance
 (traffic, route taken, surge pricing). The three models score almost the same; Lasso is slightly
 worse because alpha=0.01 shrinks the weights a little. Plain Linear is enough here.

=====================================================================
DETAILED THEORY NOTES
=====================================================================
1. WHAT IS REGRESSION?
   - Supervised learning where the target is a CONTINUOUS number (the fare in dollars).
   - Linear regression fits y = w1*x1 + w2*x2 + ... + b by choosing weights that minimise the
     sum of squared errors (Ordinary Least Squares).
   - Assumptions: linear relation, independent errors, constant error variance, little
     multicollinearity, roughly normal errors.

2. TYPES / VARIANTS
   - Simple linear regression (1 feature) and multiple linear regression (many features).
   - Ridge (L2): loss + alpha * sum(w^2). Shrinks all weights towards zero but not to exactly zero.
     Good with multicollinearity.
   - Lasso (L1): loss + alpha * sum(|w|). Can push some weights to EXACTLY zero, so it also
     does feature selection.
   - ElasticNet: mix of L1 and L2.
   - Polynomial regression, logistic regression (actually classification) and tree/SVR regressors
     are related but not used here.

3. KEY CONCEPTS AND TERMS
   - Feature (X) and target (y): inputs and the value to predict.
   - Weight / coefficient: how much the prediction changes per unit of a feature.
   - Regularisation: adding a penalty on weight size to reduce overfitting.
   - alpha: strength of that penalty. alpha=0 is ordinary linear regression. We used
     Ridge alpha=1.0 and Lasso alpha=0.01 (Lasso needs a small alpha or it zeros out too much).
   - Overfitting: model memorises training noise and does poorly on new data.
   - Multicollinearity: input features strongly correlated with each other; makes weights unstable.
   - Outlier: an extreme value (e.g. negative fare, coordinates (0,0)) that drags a least squares
     line away from the truth.
   - IQR rule: IQR = Q3 - Q1. Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers.
   - Correlation: number between -1 and +1 showing linear relation between two columns.
   - Haversine distance: great-circle distance between two lat/long points on a sphere.
   - Feature engineering: creating new useful columns (hour, year, distance_km).

4. EVALUATION METRICS
   - R2 (coefficient of determination): 1 - (error of model / error of predicting the mean).
     1 = perfect, 0 = no better than predicting the average, can be NEGATIVE if the model is
     worse than the average. Ours: about 0.49, so ~49% of fare variance is explained.
   - MSE: mean of squared errors. RMSE = sqrt(MSE) is in the SAME UNITS as the target (dollars);
     lower is better. Ours: about $2.95 average error.
   - MAE (not used): mean absolute error, less sensitive to outliers.
   - Adjusted R2 (not used): R2 corrected for the number of features.

5. TRAIN / TEST SPLIT
   - train_test_split(test_size=0.2) keeps 20% of rows unseen for honest testing; 80% train the model.
   - random_state=0 fixes the shuffle so results are reproducible. Any integer works; the
     number itself has no special meaning.
   - Scores on the test set tell us how the model generalises, training scores would be too optimistic.

6. DATASET
   - Real: Kaggle "Uber Fares Dataset", about 200,000 rides in New York with columns key,
     fare_amount, pickup_datetime, pickup/dropoff longitude and latitude, passenger_count.
   - Target: fare_amount. Engineered features: hour, year, distance_km.
   - No scaling is used: plain linear models do not need it, though Ridge/Lasso penalties do
     depend on feature scale, which is a known caveat.

7. ADVANTAGES
   - Fast, simple, interpretable (weights show feature impact).
   - Ridge handles correlated features; Lasso selects features.
   - Works well when the relation is roughly linear.

8. DISADVANTAGES
   - Assumes linearity; cannot capture complex curves by itself.
   - Sensitive to outliers and to multicollinearity (plain Linear).
   - Lasso can arbitrarily keep one of several correlated features.
   - Alpha must be tuned (cross-validation).

9. REAL-WORLD USES
   - Price/fare estimation, house price prediction, sales forecasting, risk scoring, demand planning.

10. LINEAR vs RIDGE vs LASSO
   - Linear: no penalty, can overfit when features are many or correlated.
   - Ridge: shrinks weights smoothly, keeps all features.
   - Lasso: sparse solution, some weights become exactly zero.
   - With few and weakly correlated features (our case) all three give nearly the same score.

=====================================================================
LIBRARIES AND FUNCTIONS USED
=====================================================================
 - numpy (np): radians(), sin(), cos(), arcsin(), sqrt().
 - pandas (pd): read_csv(), DataFrame(), to_datetime(), dt.hour, dt.year, drop(),
   isnull().sum(), dropna(), quantile(), between(), corr(), sort_values(), head(), to_string().
 - matplotlib.pyplot (plt): boxplot (via df.boxplot), imshow(), colorbar(), xticks(), yticks(), title(),
   tight_layout(), show().
 - sklearn.model_selection.train_test_split: splits data into train and test parts.
 - sklearn.linear_model: LinearRegression, Ridge, Lasso - fit() trains, predict() predicts.
 - sklearn.metrics: mean_squared_error() and r2_score() score the predictions.

=====================================================================
DETAILED VIVA Q&A
=====================================================================
Q1. What is regression and how is it different from classification?
A1. Regression predicts a continuous number (fare in dollars); classification predicts a category
    (spam or not). Both are supervised, but they use different metrics (R2/RMSE vs accuracy/F1).

Q2. What is the cost function of linear regression?
A2. Mean squared error: the average of (actual - predicted)^2. Least squares chooses the weights
    that make this smallest.

Q3. Explain Ridge and Lasso.
A3. Both add a penalty to the loss to prevent overfitting. Ridge adds alpha times the sum of squared
    weights (L2) and shrinks weights; Lasso adds alpha times the sum of absolute weights (L1) and can
    make some exactly zero.

Q4. L1 vs L2 regularisation - which gives feature selection?
A4. L1 (Lasso), because its diamond-shaped constraint makes weights hit zero. L2 (Ridge) only makes
    them small.

Q5. What is multicollinearity and why is it a problem?
A5. It is when input features are highly correlated. The model cannot tell which one deserves the
    credit, so the weights become unstable and hard to interpret. Ridge helps.

Q6. What does R2 mean? Can it be negative?
A6. R2 is the fraction of target variance explained by the model (1 is best). It can be negative on
    test data when the model predicts worse than simply using the mean of y.

Q7. Why is RMSE preferred over MSE for reporting?
A7. RMSE is in the same unit as the target (dollars), so it is easy to interpret. MSE is in dollars squared.

Q8. How do you detect and handle outliers?
A8. Use box plots, the IQR rule or z-scores to detect them; then remove, cap (winsorise) or transform them.
    We used the IQR rule and domain limits.

Q9. How do you choose alpha?
A9. By cross-validation (e.g. RidgeCV, LassoCV or GridSearchCV): try several alpha values and pick the one
    with the best validation score.

Q10. What is the bias-variance trade-off?
A10. High bias = too simple (underfit); high variance = too flexible (overfit). Increasing alpha raises
     bias but lowers variance.

=====================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=====================================================================
P1. Why did we create distance_km instead of using the raw lat/long?
A.  Fare depends on how far the trip is, which is a non-linear combination of four coordinate columns.
    A linear model cannot build that itself, so we engineer it with the Haversine formula. Correlation
    with fare is 0.72 vs about 0.00 for raw coordinates.

P2. Why convert degrees to radians with np.radians?
A.  numpy sin and cos expect radians, not degrees.

P3. Why did we extract hour and year and drop pickup_datetime?
A.  A raw timestamp is not a number the model can use. Hour and year are numeric and may affect price.

P4. Why random_state=0 in train_test_split?
A.  So the same rows go to train and test every time, giving reproducible R2 and RMSE.

P5. What happens if we compute R2 on the training set instead of the test set?
A.  It will usually look better (optimistic) because the model has already seen those rows; it hides
    overfitting.

P6. Why is Lasso alpha=0.01 and not 1.0?
A.  Lasso penalises the absolute weights strongly; a large alpha would shrink many weights to zero and
    underfit. A small alpha keeps most of the signal.

P7. Why is the Lasso score slightly lower here?
A.  The penalty shrinks the useful distance_km weight a little, introducing a small bias (R2 0.4844 vs 0.4895).
    With so few features, regularisation brings no benefit.

P8. Why do we use errors="ignore" in drop() and errors="coerce" in to_datetime()?
A.  "ignore" lets the script run if the id columns ("Unnamed: 0", "key") do not exist; "coerce" turns
    unparseable dates into NaT instead of crashing, and they are dropped afterwards.

P9. Why is the data not scaled?
A.  Linear regression is scale-invariant. Ridge/Lasso penalties do depend on scale, so in a stricter
    pipeline you would add StandardScaler; here results are acceptable without it.

P10. Is this result real Uber performance?
A.  Yes. It uses the real Uber dataset, and R2 about 0.49 is realistic because fare also depends on
     traffic, route and surge.

P11. How do you switch between online and local data?
A.  Option A fetches uber.csv from the datasets repo with read_csv(url). For Option B, put uber.csv next
     to the script, comment out Option A and uncomment df = pd.read_csv("uber.csv").

END OF VIVA NOTES
"""
