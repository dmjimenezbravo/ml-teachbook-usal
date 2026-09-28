# 3.3 Anomaly Detection

## Introduction

Anomaly detection is an unsupervised task of great industrial relevance, aimed at identifying unusual patterns or suspicious events within a continuous stream of data. It is widely applied in financial fraud detection, corporate network intrusion prevention, manufacturing quality control, and automated dataset cleaning.

## The philosophy of anomaly detection

At a conceptual level, anomaly detection rests on the premise that normal behaviors (called inliers) are highly frequent and share consistent behavioral patterns, while anomalies (outliers) are rare events whose characteristics deviate significantly from the norm.

### Anomaly detection vs. novelty detection

Although these terms are sometimes used interchangeably in practice, the specialized literature draws a critical methodological distinction based on the contamination of the training dataset:

- **Anomaly/outlier detection**: the algorithm is trained on a dataset that has not been cleaned and may therefore contain an unknown percentage of naturally infiltrated outliers or noisy samples. The goal is both to identify these outliers within the training set itself (data cleaning) and to correctly classify new incoming observations.
- **Novelty detection**: the algorithm is trained under the strict assumption that the training dataset is completely clean and free of anomalies. The goal here is to precisely define the boundary of known "normal" behavior in order to label any new or unknown pattern as a "novelty."

### The unsupervised approach

Anomaly detection is fundamentally addressed through unsupervised learning because, in real-world situations, it is not known in advance what type of anomaly may occur. Attempting to train a traditional supervised classifier (such as logistic regression) proves ineffective due to the severe class imbalance and because future anomalies may present entirely novel patterns the model never saw during training.

## Statistical and reconstruction-based approaches

Before resorting to more complex models, there are two highly efficient classic approaches based on density estimation and the linear projection of the data.

[Normal Data (Inliers)] ──▶ Density Modeling/Projection ──▶ Established Threshold
                                                                           │
  [Anomaly (Low Density/High Error)] ───────────────────────────────────▼──▶ Alert / Outlier

### Density modeling via Gaussian mixtures (GMM)

Under this approach, normal observations are assumed to concentrate in regions of the feature space with high probability density.

- A **Gaussian Mixture Model** (GMM) is trained to approximate the dataset's probability density function.
- To classify a new sample, its **density** under the estimated model is computed. Any instance located in a low-density region, below a preset threshold, is flagged as an anomaly.
- In real-world settings where the historical failure rate is known (for example, 4% defective products in a factory), the **threshold** is set mathematically by selecting the corresponding percentile (the 4% with lowest density under the model).

{numref}`fig-gaussian-anomaly` illustrates the concept: the blue contours represent curves of equal probability density, and the red points marked with an X are anomalies falling outside the high-density regions.

```{figure} ../../_static/generated/figures/en/gaussian_anomaly_detection.png
:name: fig-gaussian-anomaly
:alt: Gaussian density contours with normal data clustered in the center and three anomalies marked outside the density curves
:width: 70%
:align: center

Anomaly detection via Gaussian density estimation.
```

### Reconstruction approach using PCA

This technique is based on the principle that the majority principal components from a PCA analysis capture the directions of maximum variance that characterize the system's normal behavior.

- The dataset is projected into a low-dimensional space using PCA and subsequently reconstructed back into the original space using the inverse matrix.
- For each instance, the **reconstruction error** is computed (the squared distance between the original vector $x$ and its reconstruction $\hat{x}$).
- Since the principal components do not capture the unusual deviations of outliers, anomalies will experience a significantly higher reconstruction error than normal instances, allowing them to be easily identified.

{numref}`fig-pca-reconstruction` illustrates the process: normal data (blue) are projected onto the PC1 line and reconstructed with minimal error (thin lines), while the two anomalies (red) lie far from the data's principal direction and, once reconstructed onto that same line, show a much larger reconstruction error (long arrows).

```{figure} ../../_static/generated/figures/en/pca_reconstruction_anomaly.png
:name: fig-pca-reconstruction
:alt: Scatter plot showing normal data projected onto the first principal component with low reconstruction error, and two anomalies with much higher reconstruction error
:width: 65%
:align: center

Anomaly detection via PCA reconstruction error.
```

## Specific unsupervised algorithms

### Isolation forest

One of the most efficient and scalable algorithms {cite:p}`liu2008isolationforest` for outlier detection, especially designed to work in high-dimensional spaces.

- **Mechanics**: unlike traditional methods that try to model the density or the normal points, isolation forest explicitly seeks to isolate each observation. To do so, it builds a set of random decision trees. At each node of a tree, a feature is randomly selected, and a random cutoff threshold (between the minimum and maximum of that variable) is chosen to split the data in two. This recursive partitioning process continues until each instance is isolated in its own leaf.
- **Intuition**: since anomalies are far from the bulk of the normal data population, they require, on average, significantly fewer random partitions to be isolated. Therefore, instances with a shorter average path length to the root across the forest of trees are immediately flagged as anomalies.

  Normal Data (Dense)   ────────────────▶ Requires many splits to isolate.
  Anomalies (Isolated/Rare) ───────────────▶ Isolated quickly (few branches).

{numref}`fig-isolation-forest` compares both cases: the anomaly (left) is isolated with just 2 random splits, while a normal point (right) requires many more splits to be separated from the rest.

```{figure} ../../_static/generated/figures/en/isolation_forest_concept.png
:name: fig-isolation-forest
:alt: Two panels comparing how an anomaly gets isolated with few random splits versus a normal point that requires many more splits
:width: 100%
:align: center

Isolation Forest: anomalies are isolated in fewer partitions than normal points.
```

### Local Outlier Factor (LOF)

This algorithm {cite:p}`breunig2000lof` bases its operation on analyzing the local density of samples using a k-nearest-neighbors (KNN) approach.

- LOF compares the **local density** of an instance with the density of its nearest neighbors.
- A normal instance will have a local density similar to that of its surroundings. In contrast, a **local outlier** will exhibit a significantly lower density than its nearest neighbors (it will be more isolated relative to the density of its immediate neighborhood).
- This method is extremely useful for detecting **local anomalies** that would not stand out in a global analysis because their absolute values are not extreme, yet they are unusual for the specific context of the cluster they belong to.

{numref}`fig-lof` shows a typical case: the point marked in red is not far from all the data in absolute terms, but its local density is much lower than that of its immediate neighbors, which gives it away as a local anomaly.

```{figure} ../../_static/generated/figures/en/lof_concept.png
:name: fig-lof
:alt: Scatter plot with a dense region, a sparse region, and a point marked as a local outlier for having low density relative to its neighborhood
:width: 70%
:align: center

Local Outlier Factor: a local anomaly does not stand out in a global analysis.
```

### One-class SVM

This algorithm {cite:p}`scholkopf2001oneclasssvm` is specifically optimized for novelty detection in scenarios where a clean training dataset is available.

- **How it works**: instead of finding a hyperplane that separates two classes, one-class SVM projects the data into a high-dimensional feature space via a kernel and seeks to separate the training instances from the origin.
- **Decision boundary**: this is geometrically equivalent to finding the minimum-volume region or hypersphere that encloses nearly all of the training samples. If a new observation falls outside this region, bounded by the border support vectors, it is automatically classified as an anomaly or novelty.

{numref}`fig-ocsvm` shows an example with irregularly shaped data: one-class SVM traces a non-linear boundary (purple) that tightly wraps around the region of normal data (blue), so that any new observation falling outside it (red crosses) is classified as an anomaly.

```{figure} ../../_static/generated/figures/en/one_class_svm_boundary.png
:name: fig-ocsvm
:alt: Scatter plot showing a non-linear decision boundary wrapping around the normal data, with three new observations outside the boundary marked as anomalies
:width: 65%
:align: center

One-class SVM: the boundary wraps the normal region; anything outside it is classified as an anomaly.
```

## Summary

- **Anomalies**: rare, contextual, or collective events.
- **Simple methods**: Gaussian for 1D, PCA-based approaches.
- **Advanced methods**: Isolation Forest, Local Outlier Factor, one-class SVM.

---

Congratulations! You have completed the content of "Introduction to Machine Learning." You now have a solid foundation to explore more advanced topics such as **deep learning**, **natural language processing**, or **computer vision**.
