"""
Practical 1 : Feature Transformation - PCA on Wine dataset
===========================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn
 2. Dataset : wine.csv. Column 'Wine' = class (1/2/3), remaining 13 columns = measurements.
              Option A (active): fetch it online from the datasets repo.
              Option B: load a local copy kept next to this script (comment out A, uncomment B).
 3. Run     : python 1_PCA_Wine.py     (or paste cells into Jupyter)
 4. Steps   : load -> split X / y -> standardise -> PCA(2) -> explained variance -> scatter plot
"""
# --- Step 0: Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Step 1: Load dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/wine.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("wine.csv")
# Quick check that the load worked
print(df.head(), "\nShape:", df.shape)

# --- Step 2: Split into features (X) and target (y) ---
# (target is the FIRST column here)
# First column is the class label, the other 13 are measurements
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# --- Step 3: Standardise the features (PCA is scale sensitive) ---
# Why: PCA chases variance, so a big column like Proline (~1000) would dominate Hue (~1).
X_scaled = StandardScaler().fit_transform(X)

# --- Step 4: PCA - full fit first to see how much variance each PC holds ---
# Keep all 13 PCs and see how much variance each one holds
full = PCA().fit(X_scaled)
print("\nExplained variance ratio (all PCs):\n", full.explained_variance_ratio_.round(3))
print("Cumulative:", full.explained_variance_ratio_.cumsum().round(3))

# --- Step 5: Reduce to 2 components ---
# Keep the top 2 PCs so the data can be plotted in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nVariance captured by 2 PCs: %.2f%%" % (pca.explained_variance_ratio_.sum() * 100))

# --- Step 6: Visualise ---
# One colour per wine class
plt.figure(figsize=(8, 6))
for cls, c in zip(sorted(y.unique()), ["r", "g", "b"]):
    m = (y == cls).values
    plt.scatter(X_pca[m, 0], X_pca[m, 1], c=c, label=f"Class {cls}", edgecolor="k")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA - Wine Dataset"); plt.legend()
plt.show()

"""
=====================================================================
CODE EXPLANATION
=====================================================================
1. Imports: pandas for the table, matplotlib for the plot,
   StandardScaler for scaling and PCA for the reduction (all from scikit-learn).
2. Load: Option A (active) read_csv(url) fetches wine.csv online from the datasets repo. Option B
   is a commented line, read_csv("wine.csv"), for a local copy next to the script; uncomment it and
   comment out Option A to switch. head() and shape confirm we loaded 178 rows and 14 columns.
3. Split: iloc[:, 0] is the label (Wine class 1/2/3) and iloc[:, 1:] are the
   13 chemical measurements. PCA never sees the label; y is used only to colour the plot.
4. Scale: StandardScaler().fit_transform(X) makes every feature mean 0, std 1 so
   no feature dominates just because its numbers are large.
5. Full PCA: PCA().fit() keeps all 13 components. explained_variance_ratio_ and
   cumsum() tell us how much information each component carries.
6. Reduce: PCA(n_components=2).fit_transform() projects the 13-D data onto the 2
   directions of maximum variance (PC1, PC2).
7. Plot: matplotlib scatter() draws PC1 vs PC2, one colour per wine class.

=====================================================================
EXPECTED OUTPUT (real run)
=====================================================================
 Shape: (178, 14)
 Explained variance ratio (all PCs):
   [0.362 0.192 0.111 0.071 0.066 0.049 0.042 0.027 0.022 0.019 0.017 0.013 0.008]
 Cumulative: [0.362 0.554 0.665 0.736 0.802 0.851 0.893 0.92 0.942 0.962 0.979 0.992 1.0]
 Variance captured by 2 PCs: 55.41%
 Plot: three fairly well separated clusters (red, green, blue) along PC1.
 About 10 PCs are needed to cross 95% cumulative variance (0.962 at PC10).

=====================================================================
DETAILED THEORY NOTES
=====================================================================
1. WHAT IS PCA?
   - Principal Component Analysis is an UNSUPERVISED, LINEAR dimensionality
     reduction technique. It does not use class labels.
   - Intuition: imagine a cloud of points. PCA finds the direction in which the
     cloud is most stretched (PC1), then the next most stretched direction that is
     at right angles to the first (PC2), and so on.
   - Maths in words: compute the covariance matrix of the standardised data, find
     its eigenvectors (directions) and eigenvalues (variance along each direction),
     sort by eigenvalue, keep the top k and project the data onto them.
   - Steps: standardise -> covariance matrix -> eigen decomposition -> sort ->
     keep top k -> project (multiply data by the chosen eigenvectors).
   - sklearn actually uses SVD (singular value decomposition), which gives the same result.

2. TYPES / VARIANTS
   - Standard (linear) PCA: what we used.
   - Kernel PCA: handles non-linear structure using the kernel trick.
   - Incremental PCA: processes data in batches when it does not fit in memory.
   - Sparse PCA: components use only a few original features, easier to interpret.
   - Randomised PCA: faster approximation for large data.

3. KEY CONCEPTS AND TERMS
   - Feature / dimension: one column (here 13 chemical measurements).
   - Principal Component (PC): a new axis made from a weighted mix of the original features.
   - Eigenvector: the direction of a PC. Eigenvalue: the variance along that direction.
   - explained_variance_ratio_: eigenvalue / total variance. PC1 = 0.362 means
     PC1 alone holds 36.2% of the information. Higher = more important component.
   - Cumulative explained variance: running total; used to decide how many PCs to keep
     (common rule: keep enough PCs for 90-95%).
   - Scree plot: bar/line plot of variance per PC; look for the "elbow".
   - n_components: how many PCs to keep. We used 2 (for a 2D plot). PCA() with no
     argument keeps all 13.
   - Standardisation: (x - mean) / std. Needed because PCA is variance based.
   - fit vs transform: fit learns the parameters (mean, std, directions); transform applies them.
   - Orthogonal: the PCs are at 90 degrees, so they are uncorrelated.

4. EVALUATION
   - PCA is not a predictor, so there is no accuracy. We judge it by explained
     variance (55.41% for 2 PCs here) and by whether classes look separated in the plot.
   - No train/test split is used in this script, and so no random_state is needed
     (PCA with the default solver is deterministic on this small data).
   - Reading the result: 55% means we lost about 45% of the information by going
     to 2D, yet the classes still separate, so the lost part is mostly not class related.

5. DATASET (Wine)
   - 178 samples, 13 numeric features, 1 target (3 cultivars of wine from Italy).
   - Note: the manual mentions red vs white wine, but this dataset has 3 cultivar classes (1/2/3),
     which PCA separates well.
   - Features: Alcohol, Malic.acid, Ash, Alcalinity, Magnesium, Phenols, Flavanoids,
     Nonflav.phenols, Proanth, Color.int, Hue, OD, Proline.
   - Class sizes are roughly 59 / 71 / 48 (class 1 / 2 / 3).
   - Scales differ a lot: Proline ~ 300-1700, Hue ~ 0.5-1.7, hence scaling is vital.

6. ADVANTAGES
   - Removes correlated (redundant) features.
   - Reduces noise and overfitting, speeds up later models.
   - Enables 2D/3D visualisation of high-dimensional data.
   - Simple, fast, and has a clear mathematical basis.

7. DISADVANTAGES
   - Only captures LINEAR relationships.
   - New components are hard to interpret (mixes of many features).
   - Sensitive to scaling and to outliers.
   - Ignores class labels, so the top PCs are not guaranteed to separate classes.
   - Some information is always lost.

8. REAL-WORLD USES
   - Image compression and face recognition (eigenfaces).
   - Gene expression analysis, finance (risk factors), noise removal in signals.
   - Pre-processing before clustering or classification.

9. PCA vs LDA
   - PCA: unsupervised, maximises total VARIANCE, ignores labels.
   - LDA (Linear Discriminant Analysis): supervised, maximises SEPARATION between classes
     (between-class variance / within-class variance), max components = classes - 1.
   - Use PCA to compress/visualise; use LDA when the goal is classification.
   - PCA vs feature selection: PCA creates new features; selection just keeps a subset of old ones.

=====================================================================
LIBRARIES AND FUNCTIONS USED
=====================================================================
 - pandas (pd): read_csv() loads the file; head() previews; shape gives size; iloc[] selects columns;
   unique() lists the class labels.
 - matplotlib.pyplot (plt): figure(), scatter(), xlabel(), ylabel(), title(), legend(), show().
 - sklearn.preprocessing.StandardScaler: fit_transform() scales each column to mean 0, std 1.
 - sklearn.decomposition.PCA: fit() learns the components; fit_transform() learns and projects;
   explained_variance_ratio_ gives variance share per PC.
 - Python built-ins: zip() pairs classes with colours; sorted() orders class labels; print().
 - numpy methods on arrays: round() trims decimals; cumsum() running total; sum() adds up.

=====================================================================
DETAILED VIVA Q&A
=====================================================================
Q1. What is PCA?
A1. PCA is an unsupervised linear technique that converts correlated features into a smaller
    set of uncorrelated new features called principal components. Each PC is a direction of
    maximum variance, and the PCs are ordered by how much variance they keep.

Q2. Is PCA supervised or unsupervised? Why?
A2. Unsupervised, because it uses only the feature matrix X and never looks at the labels y.
    In our code y is used only to colour the scatter plot.

Q3. What are eigenvalues and eigenvectors in PCA?
A3. Eigenvectors of the covariance matrix are the directions of the principal components.
    The matching eigenvalue is the amount of variance along that direction, so a bigger
    eigenvalue means a more important component.

Q4. What is explained variance ratio?
A4. It is the fraction of the total variance carried by one component (eigenvalue divided by
    the sum of all eigenvalues). In our run PC1 = 36.2% and PC2 = 19.2%, together 55.41%.

Q5. How do you choose the number of components?
A5. Use the cumulative explained variance (keep enough PCs to reach about 90-95%), the scree
    plot elbow, or just the need of the task (2 for plotting). Here 95% needs about 10 PCs.

Q6. Why must data be standardised before PCA?
A6. PCA follows variance, and variance depends on units. Proline (~1000) would swamp Hue (~1)
    and PC1 would just point at Proline. Standardising gives every feature equal footing.

Q7. PCA vs LDA?
A7. PCA is unsupervised and maximises overall variance; LDA is supervised and maximises class
    separation. LDA can give at most (classes - 1) components, PCA up to the number of features.

Q8. What are the limitations of PCA?
A8. It is linear only, components are hard to interpret, it is sensitive to outliers and scaling,
    and it may discard directions that matter for classification but have low variance.

Q9. Are principal components correlated with each other?
A9. No, they are orthogonal, which means uncorrelated. This is why PCA is used to remove multicollinearity.

Q10. What is the curse of dimensionality and how does PCA help?
A10. With many features, data becomes sparse, distances lose meaning and models overfit.
     PCA cuts the number of features while keeping most of the information.

=====================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=====================================================================
P1. Why did we use StandardScaler?
A.  So every feature has mean 0 and std 1. Without it PCA would be dominated by large-valued
    columns such as Proline.

P2. What does fit_transform() do?
A.  It first fit()s (learns mean and std for the scaler, or the principal directions for PCA)
    and then transform()s (applies them) in one call. On new data you would call only transform().

P3. Why do we call PCA() first with no n_components?
A.  To keep all 13 components and see the variance of each one, so we can make an informed
    choice of k before reducing to 2.

P4. What would happen if we skipped scaling?
A.  PC1 would be almost entirely Proline, the explained variance of PC1 would be close to
    100% (misleading) and the class separation in the plot would be poor.

P5. Why is y not passed to PCA?
A.  PCA is unsupervised. We pass only X; y is kept aside to colour the points.

P6. What is the shape of X_pca?
A.  (178, 2): 178 samples, each now described by 2 numbers (PC1, PC2).

P7. What if n_components equals the number of features (13)?
A.  No reduction happens; the data is only rotated and cumulative variance reaches 100%.

P8. Why do we use iloc[:, 0] for y here?
A.  In this wine.csv the class label 'Wine' is the first column, unlike many datasets where it is last.

P9. Why is there no random_state or train/test split?
A.  We are only transforming and visualising the whole dataset, not building a predictive model,
    and PCA here is deterministic.

P10. How do you switch between online and local data?
A.  Option A fetches wine.csv from the datasets repo with read_csv(url). For Option B, put wine.csv
    next to the script, comment out Option A and uncomment df = pd.read_csv("wine.csv").

END OF VIVA NOTES
"""
