"""
Practical 4 : Clustering Analysis - K-Means on Iris, elbow method
=================================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn
 2. Dataset : Iris. Option A uses scikit-learn's built-in copy (no download); Option B reads a local
              Iris.csv (Kaggle uciml/iris) - see the two options in Step 2.
 3. Run     : python 4_KMeans_Iris.py
 4. Steps   : load -> drop labels -> elbow (WCSS for k=1..10) -> choose k -> fit KMeans
              -> plot clusters + centroids -> compare with true species
"""

# --- Step 1: Import the libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --- Step 2: Load the dataset ---
# Option A: Use the Iris data built into scikit-learn (no file needed)
iris = load_iris()
X, y = iris.data, iris.target    # X = 4 measurements, y = species 0/1/2 (only used to check results)

# Option B: Load from a local Iris.csv (Kaggle uciml/iris) kept next to this script
# Comment out Option A above and uncomment the 4 lines below to switch:
# df = pd.read_csv("Iris.csv")
# X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
# y = pd.factorize(df["Species"])[0]
print("Data shape:", X.shape)

# --- Step 3: Elbow method - find a good number of clusters k ---
# WCSS (inertia) for k = 1..10; lower means tighter clusters.
wcss = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in range(1, 11)]
# Plot WCSS against k and look for the bend (the elbow).
plt.plot(range(1, 11), wcss, "bo-"); plt.xlabel("k"); plt.ylabel("WCSS (inertia)")
plt.title("Elbow method"); plt.grid(True); plt.show()

# --- Step 4: Fit the final model with k = 3 (elbow bends at 3) ---
# Fit K-Means with 3 clusters and get a cluster number for every flower.
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X)
print("Centroids:\n", km.cluster_centers_.round(2))
# Silhouette needs no labels; Adjusted Rand compares clusters with true species.
print("Silhouette:", round(silhouette_score(X, labels), 3),
      "| Adjusted Rand vs species:", round(adjusted_rand_score(y, labels), 3))

# --- Step 5: Plot the clusters using petal length vs petal width ---
# Colour each flower by its cluster.
plt.scatter(X[:, 2], X[:, 3], c=labels, cmap="viridis", edgecolor="k")
# Mark the 3 centroids with big red X's.
plt.scatter(km.cluster_centers_[:, 2], km.cluster_centers_[:, 3], c="red", s=200, marker="X", label="Centroids")
plt.xlabel("Petal length"); plt.ylabel("Petal width"); plt.title("K-Means (k=3) on Iris"); plt.legend(); plt.show()

'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 4 : K-MEANS CLUSTERING ON IRIS
================================================================================

CODE EXPLANATION
----------------
1. Imports: pandas (only needed for Option B), matplotlib for graphs, sklearn for KMeans,
   the Iris loader and the two evaluation scores.
2. Load data: Option A (active) uses sklearn.datasets.load_iris(), giving X (4 measurements)
   and y (species 0/1/2). Option B (commented) reads a local Iris.csv with pd.read_csv and
   turns species names into numbers with pd.factorize. y is kept only for checking at the
   end (K-Means never sees y).
3. Elbow method: loop k = 1..10, fit KMeans(n_clusters=k, n_init=10, random_state=42) and
   store inertia_ (WCSS). Plot WCSS vs k with plt.plot and look for the bend.
4. Final model: KMeans(n_clusters=3, n_init=10, random_state=42). fit_predict(X) gives the
   cluster number of every flower. cluster_centers_ gives the 3 centroids.
5. Evaluation: silhouette_score(X, labels) judges the clusters without labels;
   adjusted_rand_score(y, labels) compares the clusters with the true species.
6. Visualisation: scatter plot of petal length vs petal width, coloured by cluster, with the
   centroids drawn as red X marks.

EXPECTED OUTPUT (real run, Option A: sklearn built-in Iris)
----------------------------------------------------------------------
   Data shape: (150, 4)
   Centroids (approx): [5.90 2.75 4.39 1.43], [5.01 3.43 1.46 0.25], [6.85 3.07 5.74 2.07]
   Silhouette: 0.553 | Adjusted Rand vs species: 0.73
   (Cluster numbers 0/1/2 can be in a different order on another machine; the values stay
   about the same. Two plots open: the elbow curve and the cluster scatter plot.)

DETAILED THEORY NOTES
---------------------
1. WHAT IS K-MEANS?
   - Definition: K-Means is a partition-based clustering algorithm. It splits the data into
     k groups so that points in the same group are close to each other.
   - Type: UNSUPERVISED learning. There are no target labels; the algorithm finds structure
     by itself. (We use species only afterwards to check.)
   - Intuition: imagine k magnets (centroids) dropped on the data. Every point sticks to its
     nearest magnet, then each magnet moves to the middle of its points. Repeat until nothing
     moves.
   - Maths in words: it tries to minimise WCSS = the sum, over all clusters, of the squared
     distance from each point to its cluster centroid. The centroid is simply the mean of
     the points in that cluster.
   - Algorithm (Lloyd's): (a) choose k starting centroids; (b) ASSIGN each point to the
     nearest centroid (Euclidean distance); (c) UPDATE each centroid to the mean of its
     assigned points; (d) repeat b and c until assignments stop changing (convergence).

2. TYPES AND VARIANTS
   - K-Means++ : smarter starting centroids (spread far apart). Default init in sklearn.
   - Mini-batch K-Means : uses small random batches, much faster on big data.
   - K-Medoids (PAM) : centre is a real data point (medoid) instead of a mean; more robust
     to outliers, but slower.
   - Hierarchical clustering : builds a tree (dendrogram); no need to fix k in advance;
     agglomerative (bottom-up) or divisive (top-down).
   - DBSCAN : density based; finds odd shaped clusters and marks outliers as noise.
   - Gaussian Mixture Models : soft clustering, each point has a probability for each cluster.

3. KEY CONCEPTS AND TERMS
   - Cluster : a group of similar points.
   - Centroid : the mean position of all points in a cluster (cluster_centers_).
   - k (n_clusters) : number of clusters. Must be chosen by us. We used 3.
   - WCSS / inertia : within-cluster sum of squares. Smaller = tighter clusters. It always
     falls as k grows (k = n points gives 0), so we cannot just pick the smallest value.
   - Elbow method : plot WCSS vs k. WCSS drops sharply at first, then flattens. The bend
     (elbow) is the best trade-off between few clusters and tight clusters. For Iris the
     elbow is at k = 3 (WCSS roughly 681 at k=1, 152 at k=2, 79 at k=3, then slow decrease).
   - n_init=10 : number of times K-Means is restarted with different starting centroids; the
     run with lowest WCSS is kept. Protects against a poor local minimum.
   - random_state=42 : fixes the random seed (for the starting centroids), so results are the
     same each time. 42 is only a convention; any integer works.
   - fit_predict : trains and returns cluster number for each row in one step.
   - Convergence : the point where assignments no longer change.

4. EVALUATION METRICS
   - WCSS (inertia_) : see above. Used to choose k; lower is tighter.
   - Silhouette score : for every point, s = (b - a) / max(a, b), where a = average distance
     to points in its own cluster and b = average distance to points in the nearest other
     cluster. Range -1 to +1. Near +1 = well separated; near 0 = overlapping clusters;
     negative = probably in the wrong cluster. Ours is 0.553, i.e. reasonable structure.
   - Adjusted Rand Index (ARI) : compares two groupings by counting pairs of points that are
     together/apart in both. 1 = identical, 0 = random, can be negative. "Adjusted" means
     corrected for chance. Ours is 0.73, i.e. clusters match species fairly well. It needs
     true labels, so it is an EXTERNAL measure; silhouette and WCSS are INTERNAL measures.
   - Confusion matrix / precision / recall / F1 / accuracy : these are for supervised
     classification and need predicted vs true class. They are not directly valid for
     clustering because cluster numbers are arbitrary (cluster 0 need not mean species 0).
     ARI is used instead.

5. TRAIN/TEST SPLIT
   - Not used. Clustering is unsupervised: we fit on all 150 rows because there is nothing
     to predict. random_state is used only to fix the K-Means starting points.

6. THE DATASET (IRIS)
   - 150 flowers, 3 species (Setosa, Versicolor, Virginica), 50 of each (balanced).
   - 4 numeric features in cm: sepal length, sepal width, petal length, petal width.
   - Target: species (not used for clustering, used only for ARI).
   - Setosa is clearly separate; versicolor and virginica overlap a little, so some points
     get grouped wrongly (that is why ARI is 0.73 and not 1).
   - Feature scale: all four are in cm and similar in size, so scaling was not needed here.

7. ADVANTAGES
   - Simple, easy to explain and implement.
   - Fast and scales well to large data (about O(n * k * iterations)).
   - Works well when clusters are round (spherical) and of similar size.

8. DISADVANTAGES
   - k must be chosen in advance.
   - Sensitive to starting centroids (mitigated by k-means++ and n_init).
   - Assumes round clusters of similar size and density; fails on elongated or ring shapes.
   - Sensitive to outliers (the mean gets pulled) and to feature scale (use StandardScaler
     if features have different units).
   - Only numeric data; uses Euclidean distance.

9. REAL-WORLD USES
   - Customer segmentation in marketing.
   - Image compression / colour quantisation.
   - Document or news grouping.
   - Anomaly detection (points far from every centroid).
   - Grouping genes, stores, or sensors with similar behaviour.

10. COMPARISON WITH RELATED METHODS
   - K-Means vs K-Medoids : K-Means uses the mean (can be a point that does not exist), fast
     but outlier sensitive; K-Medoids uses a real point, robust, slower.
   - K-Means vs Hierarchical : K-Means needs k first and is fast; hierarchical gives a full
     dendrogram and needs no k, but is slow (about O(n^2)) on big data.
   - K-Means vs DBSCAN : DBSCAN finds arbitrary shapes and noise; K-Means finds only round
     clusters and forces every point into a cluster.
   - K-Means vs KNN : K-Means is unsupervised (clustering); KNN is supervised
     (classification). Different goals despite the similar name.

LIBRARIES AND FUNCTIONS USED
----------------------------
- pandas (pd) : (Option B only) pd.read_csv loads the CSV; pd.factorize converts species names to numbers;
  df[[...]].values selects feature columns as an array.
- matplotlib.pyplot (plt) : plot (line graph), scatter (dots), xlabel/ylabel/title/legend/
  grid (decoration), show (display).
- sklearn.cluster.KMeans : the model; parameters n_clusters, n_init, random_state; methods
  fit, fit_predict; attributes inertia_ and cluster_centers_.
- sklearn.datasets.load_iris : built-in Iris data (fallback when the CSV is missing).
- sklearn.metrics.silhouette_score : internal cluster quality score.
- sklearn.metrics.adjusted_rand_score : compares clusters with true species.

DETAILED VIVA Q&A
-----------------
Q1. What is K-Means clustering?
A1. It is an unsupervised algorithm that divides data into k clusters. It repeatedly assigns
    each point to the nearest centroid and moves each centroid to the mean of its points,
    until nothing changes. The goal is to minimise WCSS.

Q2. What is WCSS (inertia) and how do you read it?
A2. WCSS is the sum of squared distances between every point and the centroid of its own
    cluster. A smaller value means tighter clusters. It always decreases as k increases, so
    we look at how fast it drops, not the lowest value.

Q3. Explain the elbow method.
A3. We run K-Means for k = 1 to 10 and plot WCSS against k. The curve falls steeply and then
    flattens; the bend point is the elbow. Adding more clusters after the elbow gives very
    little improvement, so that k is chosen. For Iris it is 3.

Q4. Why do you drop the species labels before clustering?
A4. Clustering is unsupervised, so the model must find groups using only the measurements.
    If we gave it the labels it would be cheating and it would no longer be clustering. We
    keep the labels aside only to check the result with the adjusted Rand score.

Q5. What is k-means++ and why use it?
A5. It is a smart way to choose the starting centroids: the first is random and each next one
    is picked far away from those already chosen. This gives better and more stable results
    than purely random starts, and it is the default in scikit-learn.

Q6. What does the silhouette score tell you?
A6. It measures how similar a point is to its own cluster compared with the nearest other
    cluster, from -1 to +1. Values near 1 mean well separated clusters, near 0 mean
    overlapping, and negative means a wrong assignment. We got 0.553.

Q7. K-Means vs K-Medoids?
A7. K-Means centres are means of points and may not be real data points; it is fast but
    sensitive to outliers. K-Medoids uses an actual data point as the centre, so it is more
    robust to outliers but slower.

Q8. K-Means vs hierarchical clustering?
A8. K-Means needs k in advance and is fast on large data. Hierarchical clustering builds a
    dendrogram, needs no k up front and lets you cut the tree anywhere, but it is slower and
    memory heavy.

Q9. What are the limitations of K-Means?
A9. You must choose k, results depend on initial centroids, it assumes round clusters of
    similar size, it is sensitive to outliers and to feature scale, and it works only on
    numeric data.

Q10. Why is accuracy or a confusion matrix not used here?
A10. Cluster numbers are arbitrary: cluster 0 may be the Virginica group. So direct
     comparison of numbers is meaningless. Adjusted Rand Index is label-order independent
     and is the right way to compare clusters with true classes.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. What does n_init=10 do in this code?
A1. KMeans is run 10 times with different starting centroids and the best run (lowest WCSS)
    is kept. This reduces the chance of getting stuck in a poor local solution.

Q2. Why random_state=42?
A2. It fixes the random seed used for choosing starting centroids so that the same numbers
    appear every time we run the script. Without it the clusters could change slightly.

Q3. What does fit_predict(X) return and how is it different from fit(X)?
A3. fit_predict trains the model and returns an array of 150 cluster numbers (0, 1 or 2),
    one per flower. fit only trains and returns the model itself (we used .fit(X).inertia_
    in the elbow loop because we only needed the WCSS).

Q4. Why did we pick k = 3 and what in the code shows it?
A4. The elbow plot (wcss list for k = 1..10) bends at 3, and we know Iris has 3 species. The
    value is hard coded in KMeans(n_clusters=3, ...) for the final model.

Q5. Which columns are plotted and why?
A5. X[:, 2] (petal length) and X[:, 3] (petal width). Petal measurements separate the
    species best, so the clusters look clean. The red X marks are cluster_centers_ at those
    same two columns.

Q6. How do you switch between the built-in Iris and your own Iris.csv?
A6. Comment out Option A (load_iris) and uncomment the 4 lines of Option B. Both give the same
    150 x 4 data, so the rest of the script does not change.

Q7. What do the printed Silhouette 0.553 and ARI 0.73 mean?
A7. Silhouette 0.553 means moderately well separated clusters. ARI 0.73 means the clusters
    agree with the real species for most pairs of flowers; the mismatch comes from the
    overlap of versicolor and virginica.

END OF VIVA NOTES
'''
