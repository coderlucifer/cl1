"""
Practical 3 : Classification Analysis - SVM on handwritten digits (0-9)
=======================================================================
INSTRUCTIONS
 1. Install:  pip install numpy matplotlib scikit-learn
 2. Dataset : sklearn built-in load_digits() (1797 images, 8x8 pixels, 10 classes) - no download.
 3. Run     : python 3_SVM_Digits.py
 4. Steps   : load -> view samples -> split -> scale -> SVC(rbf) -> accuracy/report/confusion matrix
              -> compare kernels -> predict one image
"""
# --- Step 0: Imports ---
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# --- Step 1: Load the dataset ---
# Each image is flattened to 64 pixel values (X); y is the digit 0-9
digits = load_digits()
X, y = digits.data, digits.target
print("Images:", digits.images.shape, "| Features:", X.shape[1], "| Classes:", list(digits.target_names))

# --- Step 2: Look at a few images ---
# Show the first 10 images in a 2x5 grid with their labels
fig, ax = plt.subplots(2, 5, figsize=(8, 4))
for a, img, lab in zip(ax.ravel(), digits.images, digits.target):
    a.imshow(img, cmap="gray_r"); a.set_title(lab); a.axis("off")
plt.show()

# --- Step 3: Split + scale ---
# Hold out 25% for testing; stratify keeps the same digit mix in both parts
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)  # same split every run
# Learn the scaling from training data only so no test information leaks in
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

# --- Step 4: Train SVM (RBF kernel) ---
# RBF = curved boundary; C=10 fits training data harder; gamma="scale" is picked automatically
clf = SVC(kernel="rbf", C=10, gamma="scale").fit(X_tr, y_tr)
pred = clf.predict(X_te)

# --- Step 5: Evaluate ---
# Accuracy, per-digit report, and a confusion matrix (rows = actual, columns = predicted)
print("\nAccuracy:", round(accuracy_score(y_te, pred), 4))
print(classification_report(y_te, pred))
ConfusionMatrixDisplay.from_predictions(y_te, pred); plt.title("SVM confusion matrix"); plt.show()

# --- Step 6: Kernel comparison ---
# Default C and gamma for every kernel so the comparison is fair
for k in ["linear", "poly", "rbf", "sigmoid"]:
    print(f"{k:8s} accuracy: {SVC(kernel=k).fit(X_tr, y_tr).score(X_te, y_te):.4f}")

# --- Step 7: Predict a single image ---
# Slice X_te[i:i+1] so predict() gets a 2D (1, 64) array
i = 0
print("\nTest image", i, "-> predicted", clf.predict(X_te[i:i+1])[0], "| actual", y_te[i])

"""
=====================================================================
CODE EXPLANATION
=====================================================================
1. Imports: matplotlib for pictures, and scikit-learn for the dataset, split, scaler, SVC model
   and the metrics.
2. Load: load_digits() gives 1797 images of 8x8 pixels. digits.data holds each image flattened to
   64 numbers (X) and digits.target holds the digit 0-9 (y).
3. View: subplots(2, 5) and imshow() show the first 10 images with their labels, just to see the data.
4. Split + scale: train_test_split(test_size=0.25, random_state=42, stratify=y) gives 1347 train and
   450 test images. StandardScaler is fitted on the training part only and then applied to both.
5. Train: SVC(kernel="rbf", C=10, gamma="scale").fit() builds the classifier; predict() labels the test set.
6. Evaluate: accuracy_score(), classification_report() and ConfusionMatrixDisplay show the quality.
7. Kernel comparison: loop over linear, poly, rbf and sigmoid kernels with default C and print each
   test accuracy using .score().
8. Single prediction: predict() on one test image (X_te[0:1]) and compare to the true label.

=====================================================================
EXPECTED OUTPUT (real run)
=====================================================================
 Images: (1797, 8, 8) | Features: 64 | Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 Accuracy: 0.98   (macro avg precision/recall/F1 all 0.98; test support = 450 images)
 Per-digit F1: lowest for 8 (0.96, recall 0.93); digits 0, 2 and 3 are perfect (1.00).
 Kernel accuracies: linear 0.9800 | poly 0.9711 | rbf 0.9800 | sigmoid 0.9467
 Test image 0 -> predicted 1 | actual 1
 (Printed classes may show np.int64(...) wrappers on newer numpy versions; values are still 0-9.)

=====================================================================
DETAILED THEORY NOTES
=====================================================================
1. WHAT IS SVM?
   - Support Vector Machine is a SUPERVISED learning algorithm used mainly for classification
     (also regression as SVR).
   - Intuition: among all lines (hyperplanes) that separate two classes, pick the one with the MAXIMUM
     MARGIN, i.e. the widest empty street between the classes. Points touching the street edges
     are the SUPPORT VECTORS; only they define the boundary.
   - Maths in words: maximise the margin 2/||w|| (equivalently minimise ||w||^2) subject to every
     point being on the correct side; the soft-margin version allows some errors, controlled by C.
   - Multi-class: SVC uses one-vs-one internally (45 binary classifiers for 10 digits).

2. TYPES / VARIANTS
   - Hard margin (no errors allowed) vs soft margin (some errors allowed, controlled by C).
   - Linear SVM vs kernel SVM. Kernels: linear, polynomial (poly), RBF (Gaussian) and sigmoid.
   - SVC (classification), SVR (regression), OneClassSVM (outlier detection), LinearSVC (fast linear).

3. KEY CONCEPTS AND TERMS
   - Hyperplane: the decision boundary (a line in 2D, a plane in 3D, a flat surface in higher dimensions).
   - Margin: distance between the hyperplane and the nearest points of each class.
   - Support vectors: the training points closest to the boundary; removing others changes nothing.
   - Kernel trick: computes similarity in a higher dimensional space without actually going there,
     so non-linear data becomes separable.
   - kernel: shape of the boundary. We used "rbf" for the main model; linear gives straight boundaries.
   - C: penalty for misclassification. Small C = wide margin, more mistakes allowed (underfit risk);
     large C = narrow margin, few mistakes on training (overfit risk). We used C=10.
   - gamma: reach of a single sample in RBF/poly/sigmoid. Large = tight, wiggly boundary (overfit);
     small = smooth boundary. "scale" = 1 / (n_features * X.var()), a sensible automatic value.
   - Feature scaling: SVM uses distances, so unscaled features distort the boundary.
   - stratify: keeps class proportions the same in train and test.

4. EVALUATION METRICS
   - Accuracy: correct predictions / total. Ours 0.98 (441 of 450 correct approx.). Fine here as classes are balanced.
   - Precision: of the images predicted as digit d, how many truly are d.
   - Recall: of the images that truly are d, how many we found.
   - F1-score: harmonic mean of precision and recall; balances the two.
   - Support: number of test samples of each class (about 45 each).
   - Macro avg: simple average over classes; weighted avg: average weighted by support.
   - Confusion matrix: table of actual vs predicted; diagonal = correct, off-diagonal = confusions
     (e.g. 8 mistaken for another digit).

5. TRAIN / TEST SPLIT
   - test_size=0.25: 450 test images and 1347 training images.
   - random_state=42: fixed seed so the shuffle (and our results) is reproducible. 42 is just a convention.
   - stratify=y: each digit appears in the same proportion in both parts.
   - Scaler is fitted ONLY on training data to avoid data leakage (test statistics must not influence training).

6. DATASET (sklearn digits)
   - 1797 samples, each an 8x8 grayscale image (pixel values 0-16), flattened to 64 features.
   - Target: digit 0-9 (10 classes, about 180 images each, nearly balanced).
   - A small version of the MNIST idea (MNIST is 28x28 with 70,000 images).

7. ADVANTAGES
   - Effective in high-dimensional spaces and with a clear margin of separation.
   - Kernel trick handles non-linear data.
   - Memory efficient (only support vectors are stored) and robust against overfitting with good C.

8. DISADVANTAGES
   - Training becomes slow on very large datasets (roughly between O(n^2) and O(n^3)).
   - Needs feature scaling and careful tuning of C, gamma and kernel.
   - No direct probability output (needs probability=True, which is slower).
   - Hard to interpret with non-linear kernels.

9. REAL-WORLD USES
   - Handwriting/digit recognition, face detection, text and spam classification, bioinformatics
     (protein/gene classification), image classification.

10. SVM vs RELATED METHODS
   - vs Logistic Regression: logistic gives probabilities and a linear boundary; SVM maximises margin
     and can use kernels.
   - vs KNN: KNN stores all data and is slow at prediction; SVM keeps only support vectors.
   - vs Decision trees/Random Forest: trees need no scaling and are more interpretable; SVM is
     often stronger on dense numeric data like pixels.
   - Linear vs RBF kernel: linear is fast and good when data is linearly separable; RBF is more
     flexible. On digits both reach 0.98 here.

=====================================================================
LIBRARIES AND FUNCTIONS USED
=====================================================================
 - matplotlib.pyplot (plt): subplots(), imshow(), set_title(), axis(), title(), show().
 - sklearn.datasets.load_digits: loads the digits images, flat data and target.
 - sklearn.model_selection.train_test_split: splits into train/test with test_size, random_state, stratify.
 - sklearn.preprocessing.StandardScaler: fit() learns mean/std, transform() applies scaling.
 - sklearn.svm.SVC: fit() trains, predict() predicts, score() returns accuracy.
 - sklearn.metrics: accuracy_score() overall accuracy; classification_report() precision/recall/F1;
   ConfusionMatrixDisplay.from_predictions() draws the confusion matrix.
 - Python built-ins: zip(), round(), list(), print(), f-strings for neat formatting.

=====================================================================
DETAILED VIVA Q&A
=====================================================================
Q1. What is SVM?
A1. A supervised algorithm that finds the hyperplane separating classes with the maximum margin.
    It can handle non-linear data using kernels.

Q2. What are support vectors?
A2. The training points closest to the decision boundary (on or inside the margin). They alone
    determine the hyperplane; other points could be deleted without changing it.

Q3. What is the kernel trick?
A3. Instead of explicitly mapping data to a higher dimension, a kernel function computes the dot
    product in that space directly. This lets SVM draw curved boundaries cheaply.

Q4. What is the role of C?
A4. C controls the trade-off between a wide margin and training errors. Large C punishes mistakes
    heavily (narrow margin, overfitting risk); small C allows more mistakes (smoother, underfitting risk).

Q5. What is the role of gamma?
A5. Gamma sets how far the influence of one training point reaches in RBF kernels. High gamma gives
    very local, wiggly boundaries; low gamma gives smoother ones.

Q6. Hard margin vs soft margin?
A6. Hard margin allows no misclassification and works only for perfectly separable data. Soft margin
    allows some violations, controlled by C, and works on noisy real data.

Q7. How does SVM handle multi-class problems?
A7. SVM is inherently binary. SVC uses one-vs-one (a classifier for each pair of classes, then voting);
    one-vs-rest is the other common strategy.

Q8. Why is feature scaling important for SVM?
A8. SVM relies on distances/dot products. Features with large ranges would dominate, distorting the
    margin and the RBF kernel.

Q9. Which metrics did you use and what do they mean?
A9. Accuracy (overall correct), precision, recall and F1 per class, plus the confusion matrix to see
    which digits are confused. All came out near 0.98.

Q10. How would you tune an SVM?
A10. Use GridSearchCV or RandomizedSearchCV over C, gamma and kernel with cross-validation, and pick
     the combination with the best validation score.

=====================================================================
PRACTICAL AND CODING VIVA QUESTIONS
=====================================================================
P1. Why did we use StandardScaler?
A.  SVM is distance based, so all 64 pixel features should be on the same scale. It gives each
    mean 0 and std 1.

P2. Why is the scaler fitted only on X_tr and not on all data?
A.  To avoid data leakage. If the test data influenced mean/std, the test score would be slightly
    optimistic. We fit on train, then only transform test.

P3. What does stratify=y do?
A.  It keeps the proportion of each digit equal in the train and test sets, so no digit is under-represented.

P4. What does fit_transform vs fit then transform mean here?
A.  They give the same output on the same data. We used fit(X_tr) then transform() on both sets so the
    same learned statistics are applied to train and test.

P5. What happens if C is very large, e.g. 10000?
A.  The margin becomes very narrow and the model tries to classify every training point correctly,
    risking overfitting and lower test accuracy.

P6. Why did all kernels in the loop use default C and gamma?
A.  For a fair, simple comparison of kernel shape only. The main model uses C=10, but the loop uses SVC(kernel=k) defaults (C=1).

P7. Why is X_te[i:i+1] used instead of X_te[i]?
A.  predict() expects a 2D array (n_samples, n_features). Slicing keeps the shape (1, 64); X_te[i] would be 1D and raise an error.

P8. Why does linear kernel do so well on digits?
A.  With 64 features and few samples, digit classes are almost linearly separable in pixel space, so a
    linear boundary already reaches 0.98 here. RBF matches it; sigmoid is weakest at 0.9467.

P9. What does the confusion matrix tell you that accuracy does not?
A.  Which specific digits are confused (e.g. 8 has the lowest recall 0.93), so you know where the model fails.

P10. Will you get the same numbers every run?
A.  Yes, because random_state=42 fixes the split and SVC training is deterministic.

END OF VIVA NOTES
"""
