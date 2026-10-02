# **Clustering**

## Introduction

Unsupervised learning represents a fundamental paradigm of artificial intelligence, characterized by the absence of labels or target variables to guide the training process. In this block, the system does not receive a "ground-truth" criterion; instead, it must explore the intrinsic mathematical relationships within the data to reveal its hidden structure.

Within this paradigm, **clustering** (or grouping) is the most widespread technique, aimed at partitioning a dataset into subgroups whose members share high internal similarity while, at the same time, being highly differentiated from members of other subgroups.

## **Clustering** algorithms

Several **clustering** algorithms exist. The most relevant ones today are explained below.

### The $K$-means algorithm

$K$-means {cite:p}`macqueen1967kmeans` is a grouping algorithm that performs partitions based on centroids. The algorithm is an iterative grouping technique that seeks to partition a set of observations into $K$ distinct clusters. It is a geometric approach based on the concept of centroids, which act as the center of gravity of each group.

{numref}`fig-kmeans-flow` summarizes the algorithm's cycle: after assigning each sample to its centroid and computing the inertia (WCSS), the centroids are updated and the process repeats until convergence.

```{figure} ../../_static/generated/diagrams/en/02_unsupervised_learning_01_clustering_01.svg
:name: fig-kmeans-flow
:alt: Flowchart of the K-means algorithm with the stages of assignment to centroids, inertia computation, and centroid update in a loop until convergence
:width: 100%
:align: center

Iterative cycle of $K$-means.
```

#### Algorithmic mechanics and convergence

The mathematical goal of $K$-means is to minimize intra-cluster inertia, formally known as the Within-Cluster Sum of Squares (WCSS). The optimization process is divided into the following iterative steps:

1. **Initialization**: $K$ points are defined in the feature space to act as initial centroids.
2. **Assignment step** (expectation): each sample in the dataset is assigned to the nearest centroid using a distance metric, typically the Euclidean distance:
    $d(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^{D} (x_i - y_i)^2}$
3. **Update step** (maximization): the position of each centroid is recalculated as the mathematical average (the arithmetic mean) of all samples assigned to that cluster.
4. **Convergence**: steps 2 and 3 are repeated iteratively until the centroids no longer change position significantly, or the sample assignments remain stable.

{numref}`fig-kmeans-process` shows the three stages on a dataset with three natural groups: the centroids (marked with an X) start from random positions and converge toward the center of each group after just a few iterations.

```{figure} ../../_static/generated/figures/en/kmeans_process.png
:name: fig-kmeans-process
:alt: Three panels showing the initialization, an intermediate iteration, and the final convergence of the K-means algorithm
:width: 100%
:align: center

Iterative K-means process: from initialization to convergence.
```

##### The local minima and initialization problem ($K$-means++)

$K$-means is a heuristic algorithm and is therefore highly sensitive to the random initialization of its centroids. If the initial centroids are placed in unfavorable regions of the feature space, the algorithm can converge to suboptimal local minima.

To mitigate this vulnerability, $K$-means++ initialization is used:

- The first centroid is selected uniformly at random among the data points.
- Subsequent centroids are chosen probabilistically, where the probability of selecting a point as the new centroid is proportional to the squared distance to the nearest existing centroid:
    $P(x) \propto D(x)^2$

This smart-seeding method ensures the initial centroids are widely spread across the feature space, speeding up convergence and improving the quality of the final grouping.

#### Selecting model complexity

##### The elbow method

Since the number of clusters $K$ is a hyperparameter that must be defined by the user before training, a scientific protocol is required to select it.

The elbow method consists of plotting the WCSS value as a function of different values of $K$:

- As $K$ increases, WCSS naturally decreases because the clusters become smaller and points move closer to their respective centroids.
- The optimal value of $K$ is located at the inflection point of the curve (the "elbow"), where the decrease in inertia stops being abrupt and becomes linear. This represents an optimal balance between model complexity and group cohesion.

{numref}`fig-elbow` shows a typical example: inertia drops sharply up to $K=3$ and barely improves afterward, signaling that 3 is the most reasonable number of clusters for this data.

```{figure} ../../_static/generated/figures/en/elbow_method.png
:name: fig-elbow
:alt: Plot of inertia (WCSS) against the number of clusters K, with the elbow marked at K=3
:width: 70%
:align: center

Elbow method: inertia stops decreasing sharply beyond K=3.
```

### Hierarchical clustering: agglomerative and dendrograms

Hierarchical **clustering** groups data without needing to define the number of groups $K$ in advance. Its main advantage is that it produces a hierarchical grouping structure that can be easily visualized.

#### Agglomerative approach (bottom-up)

The most common method is agglomerative hierarchical clustering, which operates bottom-up. The algorithm starts by treating each individual observation as its own independent cluster. At each iterative step, the distances between all clusters are computed and the two closest groups are merged. This merging process repeats successively until all samples are integrated into a single global cluster.

#### Dendrograms and interpreting cuts

The hierarchy of merges is graphically represented by a tree diagram called a dendrogram. The vertical axis of the dendrogram represents the merge distance (or cophenetic dissimilarity) between subgroups.

The horizontal axis arranges individual observations so similar branches are adjacent. The user can determine the final number of clusters by making a horizontal cut across the dendrogram at a specific dissimilarity height. The number of vertical lines intersected by the cut defines the resulting number of groups, offering an interpretive flexibility that rigid algorithms like $K$-means lack.

Dissimilarity
    ▲
  8 ┼         ┌─────────┴─────────┐
    │         │                   │
  4 ┼   ┌─────┴─────┐             │
    │   │           │             │
  0 ┴───┴───────────┴─────────────┴───► Observations

{numref}`fig-dendrogram` shows a real dendrogram built on data with three groups: the dashed red line marks a cut that produces exactly 3 clusters (one per color).

```{figure} ../../_static/generated/figures/en/dendrogram.png
:name: fig-dendrogram
:alt: Hierarchical clustering dendrogram with three colored branches and a horizontal cut line producing three clusters
:width: 80%
:align: center

Hierarchical clustering dendrogram with a cut that produces 3 clusters.
```

#### Linkage criteria

To determine which clusters should be merged at each step, we must define how to measure distance or dissimilarity between groups containing multiple observations. The main linkage methods are:

- **Complete linkage**: computes the maximum distance between any member of the first cluster and any member of the second cluster:
    $D(A, B) = \max_{x \in A, y \in B} d(x, y)$
It produces compact dendrograms and well-balanced, spherically shaped clusters.

- **Single linkage**: computes the minimum distance between any member of the first cluster and any member of the second cluster:
    $D(A, B) = \min_{x \in A, y \in B} d(x, y)$
It is sensitive to noise and can produce a chaining effect, where clusters merge into elongated, diffuse shapes.

- **Average linkage**: computes the average of all distances between each point in the first cluster and each point in the second:
    $D(A, B) = \frac{1}{|A||B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$
It is a robust criterion that balances the stability of complete linkage with the tolerance of single linkage.

{numref}`fig-islr-nci60-dendrogram` compares the three linkage criteria on real gene-expression data from the *NCI60* dataset (cancer cell lines): note how single linkage (bottom) produces the characteristic chaining effect, while complete and average linkage (top and middle) yield more balanced groupings.

```{figure} ../../_static/book_figures/islr_fig10_17_nci60_dendrogram.png
:name: fig-islr-nci60-dendrogram
:alt: Three dendrograms of the NCI60 dataset using complete, average, and single linkage, taken from An Introduction to Statistical Learning
:width: 85%
:align: center

Hierarchical clustering of the *NCI60* dataset with complete, average, and single linkage.
Source: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figure 10.17. Springer. Freely distributed for educational use (statlearning.com).
```

### DBSCAN: density-based clustering

The DBSCAN algorithm (Density-Based Spatial Clustering of Applications with Noise) {cite:p}`ester1996dbscan` offers a radically different approach from $K$-means and hierarchical **clustering** by defining groups based on the local density of the data in the feature space. This allows it to discover clusters of arbitrary geometric shapes and naturally isolate noisy samples.

#### Foundations and critical parameters

DBSCAN requires setting two essential hyperparameters that define the local density of the neighborhood:

- **$\epsilon$ (Epsilon)**: the neighborhood radius defining the surroundings around any point.
- **MinPts** (minimum samples): the minimum number of points that must exist within radius $\epsilon$ for that region to be considered dense.

#### Classifying points in DBSCAN

During execution, DBSCAN examines every point in the dataset and classifies it into one of three exclusive categories based on the density conditions of its neighborhood:

- **Core points**: a point is labeled a core point if its $\epsilon$-radius neighborhood contains a number of samples equal to or greater than MinPts.
- **Border points**: points that do not meet the minimum density requirement of MinPts to be considered core, but reside within the $\epsilon$-neighborhood of a point that is a core point.
- **Noise points** (outliers): any point that is classified as neither core nor border. These points are considered anomalies or outliers and are not assigned to any cluster.

{numref}`fig-dbscan-points` illustrates the three types with $\epsilon = 1$ and MinPts $= 4$: the blue points have at least 4 neighbors within their circle (core); the orange ones fall inside the circle of a core point but do not reach MinPts on their own (border); and the red cross has no core point nearby (noise).

```{figure} ../../_static/generated/figures/en/dbscan_point_types.png
:name: fig-dbscan-points
:alt: Diagram with blue core points surrounded by circles of radius epsilon, orange border points inside those circles, and an isolated red noise point
:width: 70%
:align: center

Classification of points in DBSCAN: core, border, and noise.
```

#### Advantages over distance-based methods

- **Robustness to noise**: unlike $K$-means, which forces every sample to belong to a cluster (artificially shifting centroids in the presence of outliers), DBSCAN natively identifies and isolates noise.
- **Geometric flexibility**: it does not assume that clusters must be spherical; it can discover nested, elongated, or complex-shaped low-dimensional clusters within high-dimensional spaces.

{numref}`fig-dbscan-vs-kmeans` compares both algorithms on non-convex data: K-means arbitrarily cuts the group in half, while DBSCAN respects the true shape defined by density.

```{figure} ../../_static/generated/figures/en/dbscan_vs_kmeans.png
:name: fig-dbscan-vs-kmeans
:alt: Side-by-side comparison of K-means and DBSCAN on moon-shaped data, showing how K-means fails to split the non-convex shape while DBSCAN keeps it intact
:width: 100%
:align: center

K-means vs. DBSCAN on non-convex shaped data.
```

## Evaluation techniques in clustering

Evaluating grouping structures is one of the most complex and debated areas of unsupervised learning, since, lacking class labels or a "ground-truth" during training, there is no single, objective success criterion. The quality of a grouping is often measured subjectively based on the model's usefulness to the end user in their application domain.

However, to approximate this analysis rigorously, the scientific literature has consolidated evaluation methods into four broad approaches: (i) internal geometric metrics, (ii) information-theory-based criteria, (iii) unified information metrics and approaches, and (iv) supervised external validations.

### Internal evaluation metrics and geometric structure

These analyze the spatial arrangement of points in the feature space to measure two desirable properties: that the clusters are as compact as possible (low internal variance) and that they are well separated from each other.

#### Silhouette coefficient and score

The silhouette coefficient ($sc$) {cite:p}`rousseeuw1987silhouette` evaluates the quality of each sample's assignment individually, serving as a diagnostic for predominantly spherical groupings. For an instance $i$, it is computed as:

$sc(i) = \frac{b_i - a_i}{\max(a_i, b_i)}$

Where:

- $a_i$ (**cohesion**): the average (usually Euclidean) distance between instance $i$ and all other points belonging to its own cluster. We want this value to be as low as possible.
- $b_i$ (**separation**): the average distance between instance $i$ and all samples in the nearest neighboring cluster (the cluster that minimizes this average distance, excluding its own). We want this value to be as high as possible.

##### Interpreting the results

- The coefficient strictly ranges over $[-1, 1]$.
- A value close to $+1$ indicates the sample is well inside its own group and far from the boundaries of other clusters.
- A value close to $0$ indicates the sample sits on the decision boundary between two clusters.
- A value close to $-1$ strongly suggests the sample has been assigned to the wrong group.

The overall silhouette score is the average of the coefficients across all instances in the dataset. In a silhouette diagram, coefficients are sorted from highest to lowest and plotted per cluster. The height of each silhouette (knife shape) indicates the cluster size, and the width represents the individual coefficient of its points. This allows for visually identifying whether any cluster is excessively large or has too many samples below the overall average score (represented by a dashed vertical line), which reveals low-quality groupings.

{numref}`fig-silhouette` shows a real silhouette diagram with three clusters: most samples in cluster 1 (red) and 0 (blue) exceed the average silhouette, while cluster 2 (green) is more heterogeneous.

```{figure} ../../_static/generated/figures/en/silhouette_diagram.png
:name: fig-silhouette
:alt: Silhouette diagram showing silhouette coefficients grouped by cluster, with a vertical line marking the average silhouette
:width: 70%
:align: center

Silhouette diagram: each "knife" represents a cluster and its grouping quality.
```

#### Gap statistic

Proposed by Tibshirani et al. {cite:p}`tibshirani2001gapstatistic`, the **gap statistic** mathematically formalizes the search for the optimal number of clusters ($K$), overcoming the visual vagueness of the heuristic elbow method.

It compares the log of the observed inertia curve of your data ($\log W_K$) with the mathematical expectation of the inertia computed over multiple artificially generated samples drawn from a uniform distribution (with no grouping structure, i.e., the null hypothesis) over the hyper-rectangle enclosing the real data:

$\text{Gap}(K) = E^*[\log W_K] - \log W_K$

The method estimates the optimal $K$ by maximizing this "gap." Mathematically, the formal rule selects the smallest $K$ such that:

$K^* = \text{argmin}_K \{ K \mid \text{Gap}(K) \ge \text{Gap}(K+1) - s'_{K+1} \}$

Where $s'_{K+1} = s_{K+1}\sqrt{1 + 1/B}$ represents the standard deviation of the simulation adjusted for the number of artificial replicates $B$. Its greatest advantage is that it can determine whether the optimal number of clusters is 1 (i.e., that the data has no natural clusters), a scenario where inertia or the silhouette coefficient fail.

#### Cophenetic correlation coefficient

This technique evaluates the quality of the hierarchical structure imposed by hierarchical clustering. It measures the Pearson correlation between the initial dissimilarity matrix $\{d_{ii'}\}$ of the observations and their corresponding cophenetic dissimilarities $\{C_{ii'}\}$ obtained from the merge heights in the dendrogram.

The cophenetic distance $C_{ii'}$ is the exact height of the link in the dendrogram where observations $i$ and $i'$ are first joined into a common cluster. Since cophenetic distances must strictly satisfy the ultrametric inequality ($C_{ii'} \le \max\{C_{ik}, C_{i'k}\}$), it is uncommon for real input data to fit this metric constraint perfectly. A low cophenetic correlation warns the designer that the dendrogram is forcing an artificial hierarchy where the real data does not exhibit one.

### Probabilistic and information-theory criteria

When working with probability-based grouping models (such as Gaussian Mixture Models, or GMM, trained via the EM algorithm), pure geometric distances become unreliable because groups may adopt elliptical shapes, oblique orientations, or highly variable densities.

#### Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)

If we evaluate a probabilistic model solely by its log-likelihood ($LL$), the model with the largest number of parameters or components will always appear better on the training data, promoting overfitting. To prevent this, AIC and BIC add mathematical penalties that balance fit and complexity:

$\text{AIC} = -2LL + 2d$

$\text{BIC} = -2LL + d \log(N)$

Where $LL$ is the model's maximum log-likelihood, $d$ is the total number of free parameters to learn (means, weights, and covariances of the components), and $N$ is the sample size.

##### Penalization

Since $\log N > 2$ for any real dataset ($N > 7$), BIC penalizes complexity much more severely than AIC, favoring more parsimonious models (with fewer components).

##### Asymptotic behavior

BIC is asymptotically consistent, meaning that if the true model is among the candidates, the probability of selecting it tends to 1 as $N \to \infty$. AIC, in contrast, tends to overfit on infinite samples, choosing overly complex models, although it behaves more competitively than the conservative BIC on small samples.

#### WAIC (Widely Applicable Information Criterion)

AIC and BIC assume that the model's parameters approximate a Gaussian distribution and that the model is non-singular. However, in probabilistic mixtures and deep learning models with correlated or redundant parameters (singular models), these assumptions break down.

WAIC is a Bayesian information metric that estimates the expected log pointwise predictive density (ELPD) via Monte Carlo simulation methods, penalizing complexity based on the posterior variance of each data point's predictions. This provides a robust evaluation framework that works even under singular and over-parameterized geometries.

### Unified information metrics and approaches

#### The minimum description length (MDL) principle

Inspired by information theory and data coding, the MDL principle evaluates clustering under the premise of data compression. If a grouping is natural and of high quality, it should let us describe or transmit the dataset more compactly (using fewer bits of information) than if we described the points without grouping them.

To transmit a set of points using MDL, we must encode two elements:

- The theory or model: the spatial location of the cluster centroids and their respective assignments in bits ($\log_2 K$ bits per sample).
- The data given the model: the feature values of each instance, expressed only as the residual deviation from its assigned centroid.

If the segmentation captures dense, genuine groups, the drastic reduction in bits needed to describe the small individual residuals will more than compensate for the extra cost of transmitting the centroid positions, achieving a minimal total description. If the grouping is not genuine, the total bit cost will increase, signaling that the subdivision is useless.

#### Category utility

Used in incremental and hierarchical clustering methods such as Cobweb or Classit, category utility (CU) measures the increase in the ability to correctly predict a sample's attributes after learning it belongs to a specific cluster, compared to the unconditional prediction of the data.

For categorical or nominal attributes, it is formulated as:

$\text{CU}(C_1, \dots, C_k) = \frac{\sum_{l=1}^k \Pr[C_l] \sum_i \sum_j \left( \Pr[a_i = v_{ij} \mid C_l]^2 - \Pr[a_i = v_{ij}]^2 \right)}{k}$

Where $\Pr[a_i = v_{ij} \mid C_l]$ is the conditional probability that attribute $i$ takes the value $v_{ij}$ in cluster $l$, and $\Pr[a_i = v_{ij}]$ is its overall unconditional probability.

##### Avoiding fragmentation (dividing by $k$)

Without the denominator $k$, the maximum score would be obtained by assigning each individual instance to its own exclusive cluster, where the conditional probability is $1.0$. Dividing by $k$ acts as a heuristic factor that penalizes excessive fragmentation of the model.

##### Minimum variance (acuity)

For continuous attributes, the formula assumes a normal distribution and requires estimating local and global standard deviations. To prevent a cluster with a single element (or zero variance) from producing an infinite, undefined value in the equation, a minimum variance parameter, or acuity, is imposed, emulating sensor measurement error.

### External evaluation using labeled data

When external reference labels are available (labels that were not used to guide the clustering), we can measure how well the discovered boundaries align with known biological or business classes.

#### Purity

Consists of assigning each entire cluster the most frequent class label within it and computing the proportion of correctly classified elements. It is formally defined as:

$\text{Purity} = \sum_i \frac{N_i}{N} \max_j (p_{ij})$

Where $N_i$ is the size of cluster $i$, $N$ is the total data count, and $p_{ij} = N_{ij}/N_i$ is the fraction of cluster $i$ occupied by class $j$.

##### Limitation

Purity does not penalize the number of clusters. If you assign each sample individually to its own cluster ($K=N$), purity will trivially equal $1.0$, invalidating its use unless the number of clusters is independently controlled.

#### Rand Index (RI) and Adjusted Rand Index (ARI)

The Rand index analyzes grouping consistency based on pairs of elements. It examines every possible pair of observations in the dataset and evaluates whether the decision to assign them to the same cluster or to different clusters was consistent between the estimated partition ($U$) and the reference partition ($V$):

$R = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}$

- **TP** (True Positives): number of pairs that are in the same cluster in $U$ and in the same class in $V$.
- **TN** (True Negatives): number of pairs that are in separate clusters in $U$ and in different classes in $V$.
- **FP** (False Positives): number of pairs assigned to the same cluster in $U$ but to different classes in $V$.
- **FN** (False Negatives): number of pairs in separate clusters in $U$ but in the same class in $V$.

RI ranges between $0$ (total disagreement) and $1$ (perfect agreement). However, classic RI rarely reaches 0, since chance tends to inflate its score. The Adjusted Rand Index (ARI) corrects this by subtracting the expected agreement of two random classifications under a generalized hypergeometric distribution model, ensuring a random grouping scores $0.0$ on average.

#### Classes-to-Clusters Evaluation

Popularized by classic software suites such as Weka, this approach trains the clustering model while completely ignoring the class attribute. Once the space has been segmented, the evaluator examines the composition of each cluster to retrospectively assign it the majority class label.

From this assignment, it is possible to map the geometric boundaries as if they were a supervised classifier and build a classic confusion matrix that reveals the exact rates of false positives, false negatives, and overall per-class accuracy.

## Hands-on example in Java with SMILE

The [programacion-avanzada-smile](https://github.com/dmjimenezbravo/programacion-avanzada-smile) repository includes the full example [`Ejemplo05Clustering.java`](https://github.com/dmjimenezbravo/programacion-avanzada-smile/blob/main/src/main/java/es/usal/smile/Ejemplo05Clustering.java), which applies the three algorithms from this section with the [SMILE](https://haifengl.github.io/) library (comments translated into English). It uses the *Iris* dataset **with the species column removed**: the algorithm never sees the labels, but since we know there are three species we can compare the resulting partition against the true one using an external index (ARI).

```java
import java.util.Arrays;
import smile.clustering.CentroidClustering;
import smile.clustering.Clustering;
import smile.clustering.DBSCAN;
import smile.clustering.HierarchicalClustering;
import smile.clustering.KMeans;
import smile.clustering.linkage.WardLinkage;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.MathEx;
import smile.validation.metric.AdjustedRandIndex;

MathEx.setSeed(42);

DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");
double[][] x = iris.drop("species").toArray();          // no labels
int[] especieReal = iris.column("species").toIntArray(); // only for evaluation

// 1. K-means with k = 3: fit(data, k, maxIterations)
CentroidClustering<double[], double[]> kmeans = KMeans.fit(x, 3, 100);
int[] grupos = kmeans.group();
System.out.printf("Distortion (inertia) = %.3f%n", kmeans.distortion());
System.out.printf("ARI against the true species = %.4f%n",
        AdjustedRandIndex.of(especieReal, grupos));

// Assign a new sample to the nearest centroid
double[] nueva = { 5.9, 3.0, 5.1, 1.8 };
System.out.printf("The sample falls in cluster %d%n", kmeans.predict(nueva));

// 2. Elbow method: distortion for different values of k
for (int k = 2; k <= 8; k++) {
    System.out.printf("k = %d  distortion = %.3f%n", k, KMeans.fit(x, k, 100).distortion());
}

// 3. DBSCAN: fit(data, minPts, epsilon). No need to fix k, and it detects noise
DBSCAN<double[]> dbscan = DBSCAN.fit(x, 5, 0.8);
long ruido = Arrays.stream(dbscan.group()).filter(g -> g == Clustering.OUTLIER).count();
System.out.printf("DBSCAN: %d clusters, %d noise points%n", dbscan.k(), ruido);

// 4. Agglomerative hierarchical clustering (Ward linkage), cut into 3 groups
HierarchicalClustering jerarquico = HierarchicalClustering.fit(WardLinkage.of(x));
int[] particion3 = jerarquico.partition(3);
System.out.printf("Hierarchical: ARI = %.4f%n", AdjustedRandIndex.of(especieReal, particion3));
```

A few things worth noticing when you run it:

- `kmeans.distortion()` is the within-cluster inertia (WCSS) that $K$-means minimizes; the loop over $k$ produces the values that would be plotted in the elbow-method chart.
- Points that DBSCAN considers noise receive the special label `Clustering.OUTLIER` instead of a cluster number.
- In hierarchical clustering, the linkage criterion is built first (`WardLinkage`, though `SingleLinkage`, `CompleteLinkage`, or `UPGMALinkage` for average linkage are also available) and the dendrogram is then cut with `partition(k)`.
- The full example also compares X-means (automatic choice of $k$ via BIC) and shows the effect of standardizing the variables before clustering.

To run it from the root of the repository: `mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo05Clustering`.

## Summary

- **$K$-means**: simple, fast, requires a known K.
- **Elbow Method**: a heuristic for choosing K.
- **Hierarchical clustering**: dendrogram, better visualization.
- **DBSCAN**: detects arbitrary shapes and outliers.
- **Silhouette**: a metric for evaluating quality without labels.

---

**Next**: In 3.2 we will learn to reduce dimensionality to simplify and visualize complex data.
