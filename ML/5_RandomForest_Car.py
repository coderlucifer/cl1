"""
Practical 5 : Ensemble Learning - Random Forest to predict car safety / acceptability
=====================================================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn
 2. Dataset : Car Evaluation (Kaggle elikplim/car-evaluation-data-set = UCI car.data, no header).
              Option A (active): fetch car_evaluation.csv online from the datasets repo.
              Option B: load a local copy kept next to this script (comment out A, uncomment B).
 3. Run     : python 5_RandomForest_Car.py
 4. Steps   : load + name columns -> encode categories -> split -> RandomForest(100 trees)
              -> accuracy, report, confusion matrix -> feature importance
"""

# --- Step 1: Import the libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# --- Step 2: Load the dataset (file has no header row) ---
# Name the columns ourselves because the raw file has no header.
cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/car_evaluation.csv"
df = pd.read_csv(url, header=None, names=cols)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("car_evaluation.csv", header=None, names=cols)
# Peek at the data and see how many cars are in each class.
print(df.head(), "\nShape:", df.shape, "\n\nClass counts:\n", df["class"].value_counts())

# --- Step 3: Encode text categories as numbers (ordered categories keep their natural order) ---
# Value order (smallest to largest) for each feature, so low=0, med=1, high=2 ...
order = [["low", "med", "high", "vhigh"], ["low", "med", "high", "vhigh"], ["2", "3", "4", "5more"],
         ["2", "4", "more"], ["small", "med", "big"], ["low", "med", "high"]]
# Encode with our order, and keep column names for the importance plot later.
X = pd.DataFrame(OrdinalEncoder(categories=order).fit_transform(df[cols[:-1]]), columns=cols[:-1])
# Target: the car class (text labels are fine for a classifier).
y = df["class"]

# --- Step 4: Split into training and testing data ---
# Keep 20% for testing; stratify keeps class proportions the same in both parts.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 5: Train the Random Forest ---
# Train 100 trees, then let them vote on each test car.
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
pred = rf.predict(X_te)

# --- Step 6: Evaluate on the unseen test data ---
# Accuracy plus per-class precision, recall and F1 (classes are imbalanced).
print("\nAccuracy:", round(accuracy_score(y_te, pred), 4))
print(classification_report(y_te, pred))
# Confusion matrix: true class vs predicted class.
ConfusionMatrixDisplay.from_predictions(y_te, pred, xticks_rotation=45); plt.tight_layout(); plt.show()

# --- Step 7: Feature importance (which inputs mattered most) ---
# Importance per feature (sums to 1), sorted small to large for the bar chart.
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
print("\nFeature importance:\n", imp.sort_values(ascending=False).round(3))
imp.plot.barh(title="Random Forest feature importance"); plt.show()

'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 5 : RANDOM FOREST ON CAR EVALUATION DATA
================================================================================

CODE EXPLANATION
----------------
1. Imports: pandas (data table), matplotlib (plots), and from scikit-learn
   train_test_split, OrdinalEncoder, RandomForestClassifier and the metric tools.
2. Load data: the raw UCI file has no header, so we pass header=None and our own column names.
   Option A (active) fetches car_evaluation.csv online from the datasets repo; Option B is a
   commented line that reads a local copy next to the script (uncomment it, comment out A).
   We print head(), shape and the class counts (1728 rows, 7 columns).
3. Encode: all 6 features are text but ordered (low < med < high < vhigh etc.). We pass the
   explicit order list to OrdinalEncoder(categories=order) so the numbers keep this order.
4. Split: train_test_split with test_size=0.2, random_state=42, stratify=y gives 1382 training
   rows and 346 test rows with the same class mix.
5. Train: RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr) builds
   100 trees. predict(X_te) gives the majority-vote prediction for the test rows.
6. Evaluate: accuracy_score, classification_report (precision/recall/F1 per class) and a
   confusion matrix plot with ConfusionMatrixDisplay.from_predictions.
7. Feature importance: rf.feature_importances_ placed in a pandas Series, sorted, printed and
   drawn as a horizontal bar chart.

EXPECTED OUTPUT (real run)
--------------------------
   Shape: (1728, 7)
   Class counts: unacc 1210, acc 384, good 69, vgood 65
   Accuracy: 0.9855   (about 98.6 percent on 346 test cars)
   Per class (precision / recall / F1 / support):
      acc    0.95 / 1.00 / 0.97 / 77
      good   0.93 / 0.93 / 0.93 / 14
      unacc  1.00 / 1.00 / 1.00 / 242
      vgood  1.00 / 0.77 / 0.87 / 13
   Macro avg F1 0.94, weighted avg F1 0.99.
   Feature importance: safety 0.306, persons 0.237, maint 0.159, buying 0.144,
                       lug_boot 0.088, doors 0.065
   (Small differences of a few decimals are possible with other library versions.)

DETAILED THEORY NOTES
---------------------
1. WHAT IS RANDOM FOREST?
   - Definition: an ensemble learning method that builds many decision trees and combines
     their outputs. For classification the final answer is the MAJORITY VOTE; for
     regression it is the average.
   - Type: SUPERVISED learning (we have labelled examples: class of each car).
   - Intuition: "wisdom of the crowd". One tree can memorise noise (overfit) but many
     different trees make different mistakes, and voting cancels them out.
   - How it works: (a) draw a bootstrap sample (random rows chosen WITH replacement) for each
     tree; (b) grow the tree, but at every split look at only a random subset of the
     features; (c) repeat for n_estimators trees; (d) combine by vote.
   - Maths in words: each tree is a low-bias, high-variance model; averaging many weakly
     correlated trees lowers the variance while keeping the bias low. Random feature subsets
     make the trees less correlated, so the averaging works better.

2. TYPES AND VARIANTS
   - Bagging (Bootstrap Aggregating) : parallel independent models on bootstrap samples.
     Random Forest = bagging + random feature selection at each split.
   - Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM) : trees built one after
     another, each fixing the errors of the previous. Reduces bias, can overfit if not tuned.
   - Stacking : different model types combined by a meta-model.
   - Extra Trees : like Random Forest but split thresholds are chosen randomly (even faster).
   - Random Forest Regressor : same idea for numeric targets.

3. KEY CONCEPTS AND TERMS
   - Decision tree : a flowchart of if-else questions on features that ends in a class.
   - Ensemble : a group of models whose predictions are combined.
   - Bootstrap sample : a random sample of the training rows with replacement (about 63
     percent unique rows per tree).
   - Out-of-bag (OOB) samples : the roughly 37 percent of rows a tree did NOT see; they give
     a free validation score (oob_score=True). Not enabled in our code.
   - n_estimators=100 : number of trees. More trees = more stable, slower, with diminishing
     returns. Default is 100.
   - random_state=42 : fixes the seed for bootstrap sampling and feature choices so results
     repeat.
   - Other hyper-parameters (not changed here): max_depth (limit tree depth), max_features
     (features tried per split, default sqrt(n_features)), min_samples_leaf (minimum rows in
     a leaf), min_samples_split, criterion ("gini" by default).
   - Gini impurity : measures how mixed the classes are in a node (0 = pure). Splits are
     chosen to reduce it the most.
   - feature_importances_ : average reduction in impurity that each feature gives across all
     trees, normalised so all values sum to 1. Higher = more useful feature.
   - Ordinal encoding : converting ordered categories to integers 0,1,2,... keeping the
     order. Different from one-hot encoding, which is used for unordered categories.
   - Stratified split : the train and test sets keep the same class percentages as the full
     data.

4. EVALUATION METRICS
   - Confusion matrix : table with true classes as rows and predicted classes as columns.
     Diagonal = correct predictions, off-diagonal = mistakes. For this 4-class problem it is
     a 4 x 4 grid.
   - Accuracy = correct predictions / all predictions. Can be misleading with imbalance: a
     model that always says "unacc" would already get about 70 percent.
   - Precision (per class) = TP / (TP + FP): of the cars predicted as this class, how many
     really are.
   - Recall (per class) = TP / (TP + FN): of the cars that really are this class, how many
     were found. Our vgood recall is 0.77, so some vgood cars were missed.
   - F1-score = 2 * precision * recall / (precision + recall): the harmonic mean; high only if
     both are high.
   - Support : number of true test samples of that class.
   - Macro average : simple average over classes (treats each class equally).
     Weighted average : average weighted by support (dominated by big classes).

5. TRAIN/TEST SPLIT
   - We train on 80 percent (1382 rows) and test on 20 percent (346 rows) that the model has
     never seen, so the score estimates real-world performance.
   - random_state=42 fixes the shuffle, so everyone running the script gets the same split
     and the same numbers.
   - stratify=y keeps class ratios (about 70 / 22 / 4 / 4 percent) in both parts; otherwise a
     tiny class like vgood might be nearly missing from the test set.

6. THE DATASET (CAR EVALUATION, UCI)
   - 1728 rows, 6 input features, 1 target. It covers every possible feature combination
     (4 x 4 x 4 x 3 x 3 x 3 = 1728), generated from a simple hierarchical decision model.
   - Features: buying price (low, med, high, vhigh), maintenance cost (same), doors (2, 3, 4,
     5more), persons capacity (2, 4, more), luggage boot size (small, med, big), safety
     (low, med, high).
   - Target class: unacc (1210), acc (384), good (69), vgood (65) - strongly imbalanced.
   - No missing values; all features are categorical.

7. ADVANTAGES
   - High accuracy and robust; much less overfitting than one tree.
   - Handles many features, mixed types and non-linear relations without feature scaling.
   - Gives feature importance and (optionally) a free OOB estimate.
   - Works well with default settings.

8. DISADVANTAGES
   - Less interpretable than a single tree (a "black box" of 100 trees).
   - Slower and uses more memory for large forests.
   - May still favour majority classes on imbalanced data (use class_weight="balanced").
   - Cannot extrapolate beyond the range of the training data (for regression).

9. REAL-WORLD USES
   - Credit scoring and fraud detection.
   - Medical diagnosis and risk prediction.
   - Customer churn prediction.
   - Remote sensing / land-cover classification.
   - Recommendation and ranking features in e-commerce.

10. COMPARISON WITH RELATED METHODS
   - Random Forest vs single decision tree : forest has lower variance and better accuracy,
     but the tree is easier to explain.
   - Bagging vs boosting : bagging trains models in parallel on random samples to reduce
     variance; boosting trains models in sequence where each focuses on earlier mistakes to
     reduce bias. Boosting is often more accurate but more sensitive to noise and tuning.
   - Random Forest vs SVM : forest needs no scaling and handles categories easily; SVM can
     be better on small, clean, high-dimensional numeric data.
   - Random Forest vs plain bagged trees : the extra random feature subset at each split
     decorrelates the trees.

LIBRARIES AND FUNCTIONS USED
----------------------------
- pandas (pd) : read_csv (load, header=None, names=cols), DataFrame (wrap encoded array),
  Series (importance with names), head, shape, value_counts, sort_values, round.
- matplotlib.pyplot (plt) : tight_layout, show; Series.plot.barh draws the bar chart.
- sklearn.model_selection.train_test_split : splits data (test_size, random_state, stratify).
- sklearn.preprocessing.OrdinalEncoder : categories=order, fit_transform - ordered text to
  numbers.
- sklearn.ensemble.RandomForestClassifier : n_estimators, random_state; fit, predict,
  feature_importances_.
- sklearn.metrics.accuracy_score : overall accuracy.
- sklearn.metrics.classification_report : precision, recall, F1, support per class.
- sklearn.metrics.ConfusionMatrixDisplay.from_predictions : confusion matrix plot.

DETAILED VIVA Q&A
-----------------
Q1. What is a Random Forest?
A1. It is an ensemble of many decision trees. Each tree is trained on a random bootstrap
    sample of rows and uses a random subset of features at every split. The final class is
    the majority vote of all trees, which makes it more accurate and less overfit than a
    single tree.

Q2. What is bagging?
A2. Bagging means Bootstrap Aggregating. We create several random samples (with
    replacement) of the training data, train one model on each, and combine their outputs by
    voting or averaging. It mainly reduces variance.

Q3. Bagging vs boosting?
A3. In bagging the models are independent and trained in parallel; it reduces variance.
    In boosting the models are trained one after another and each tries to correct the
    previous model's errors; it reduces bias. Random Forest is bagging; AdaBoost and XGBoost
    are boosting.

Q4. Why does Random Forest use random features at each split?
A4. If every tree could use all features, they would all pick the same strong feature first
    and look alike. Random feature subsets make the trees different (decorrelated), so
    averaging their votes reduces the error more.

Q5. What is OOB (out-of-bag) error?
A5. Each tree is trained on about 63 percent of the rows; the remaining 37 percent are its
    out-of-bag rows. Predicting those rows with the trees that did not see them gives a free
    validation estimate without a separate test set. It is switched on with oob_score=True.

Q6. How does Random Forest reduce overfitting compared with one tree?
A6. A deep single tree memorises the training data. Averaging many trees trained on
    different samples cancels out their individual random errors, so the combined model
    generalises better.

Q7. How is feature importance calculated?
A7. For every split, the drop in Gini impurity is credited to the feature used. These drops
    are summed over all trees and normalised to total 1. In our run safety (0.306) and
    persons (0.237) are the most important.

Q8. Why is accuracy alone not enough here?
A8. The classes are imbalanced (1210 unacc versus 65 vgood). A model could get high accuracy
    by favouring the big class. Per-class precision, recall and F1 show how well rare classes
    such as vgood are handled (vgood recall is only 0.77).

Q9. Do we need feature scaling for Random Forest?
A9. No. Trees split on thresholds, so the scale of a feature does not change the splits.
    This is one advantage over SVM, KNN or logistic regression.

Q10. Ordinal encoding vs one-hot encoding?
A10. Ordinal encoding maps ordered categories to integers that keep the order (low < med <
     high). One-hot makes a separate 0/1 column per category and is used when categories
     have no order. Ordinal is the right choice here because the values are ordered.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Why do we pass header=None and names=cols to read_csv?
A1. The car data file has no header row. header=None stops pandas treating the first car as
    column names, and names=cols gives each column a proper name.

Q2. What does stratify=y do in train_test_split?
A2. It makes the train and test sets contain the same percentage of each class as the full
    data. Without it the rare classes (good, vgood) could be badly under-represented in the
    test set.

Q3. Why use OrdinalEncoder with a categories list instead of LabelEncoder or default
    encoding?
A3. The default would sort values alphabetically ("high" < "low" < "med" < "vhigh"), which
    breaks the real order. By passing categories=order we make low=0, med=1, high=2,
    vhigh=3 so the numbers mean something.

Q4. What do n_estimators=100 and random_state=42 do?
A4. n_estimators=100 builds 100 trees in the forest. random_state=42 fixes the random seed
    so the bootstrap samples and feature choices, and therefore the accuracy, are the same
    every run.

Q5. How is the confusion matrix produced in the code?
A5. ConfusionMatrixDisplay.from_predictions(y_te, pred) compares true labels with the
    predictions on the test set and draws the 4 x 4 grid. xticks_rotation=45 tilts the
    labels, and plt.tight_layout() stops them being cut off.

Q6. Why do we call .sort_values() on the importance Series twice?
A6. The first sort (ascending) is for the bar chart so the largest bar ends at the top of
    the barh plot. The second sort (descending, only for the print) lists the biggest
    importance first, which reads more naturally.

Q7. How would you improve the vgood recall of 0.77?
A7. Use class_weight="balanced", tune max_depth or min_samples_leaf, or use cross-validation
    and oversampling (for example SMOTE). The test set has only 13 vgood cars, so one or two
    mistakes already move the recall a lot.

Q8. How do you switch between online and local data?
A8. Option A fetches car_evaluation.csv from the datasets repo with read_csv(url, header=None,
    names=cols). For Option B, keep car_evaluation.csv next to the script, comment out Option A and
    uncomment the local read_csv line.

END OF VIVA NOTES
'''
