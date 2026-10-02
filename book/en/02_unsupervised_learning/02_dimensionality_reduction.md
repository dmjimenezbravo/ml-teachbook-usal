# Dimensionality Reduction

## Introduction

In modern data analytics and the design of artificial intelligence systems, it is common to face datasets characterized by hundreds or thousands of predictor variables. However, working directly in these multidimensional spaces presents serious practical and theoretical drawbacks. Dimensionality reduction systematically addresses this problem by seeking to project or compress the data into a low-dimensional space (usually 2D or 3D). This process not only drastically speeds up computation and mitigates overfitting, but is also an indispensable tool for **data visualization** (DataViz) and for extracting explainable latent factors.

## The curse of dimensionality

Human intuition is wired to reason in three physical dimensions, which makes it difficult for us to understand the geometric properties of high-dimensional spaces. Many mathematical properties of traditional machine learning algorithms degrade or collapse due to this phenomenon, formally coined by Richard Bellman as the curse of dimensionality.

{numref}`fig-dimension-progression` shows the progression from a point to a $p$-dimensional hypercube: as the dimension grows, the space becomes ultra-sparse and samples tend to sit at the boundary.

```{figure} ../../_static/generated/diagrams/en/02_unsupervised_learning_02_dimensionality_reduction_01.svg
:name: fig-dimension-progression
:alt: Diagram of the progression of dimensions from a point (0D), interval (1D), square (2D), and cube (3D) to a p-dimensional hypercube, with ultra-sparse space and samples at the boundary
:width: 100%
:align: center

Progression of dimensions: from a point to a hypercube.
```

### Loss of neighborhood and sparse space

The main symptom of the curse of dimensionality is that, as the dimension $p$ increases, the volume of the space grows exponentially with respect to the features. This causes any real dataset, however large, to populate the input space extremely sparsely. To illustrate this fact:

- If in a single dimension ($p=1$) a local range of 10% represents a representative neighborhood sample, to capture the same equivalent 10% volume in a 10-dimensional space ($p=10$), a local algorithm (such as $K$-NN) must extend its search to cover 80% of the range of each variable.
- As a result, the points in the neighborhood stop being "local," and the algorithm loses its statistical estimation power.
- To maintain a constant sampling density as predictor variables are added, the required dataset size grows exponentially ($O(N^p)$). In practice, obtaining such a volume of data is infeasible.

### The geometric anomaly of the extremes

In high-dimensional spaces, geometry becomes highly counterintuitive.

- **Attraction to the boundary**: in a two-dimensional unit square (1x1), the probability that a randomly chosen sample lies near the edge (within 0.001 distance) is barely 0.4%. However, in a 10,000-dimensional hypercube, this probability exceeds 99.9999%. Practically all samples naturally reside at the edges and outer corners of the hypercube.
- **Uniformity of distances**: the average distance between two randomly chosen points in a hypercube grows drastically with dimensionality. Relative distances become equalized, making all samples appear to be almost the same distance from each other and neutralizing the effectiveness of standard metrics such as Euclidean distance. This severely increases the risk of overfitting in classifiers, since the model becomes unstable under slight variations.

{numref}`fig-curse-dimensionality` quantifies both effects through simulation: on the left, the percentage of samples near a hypercube's boundary grows rapidly with dimensions; on the right, the ratio between the minimum and maximum distance among random points tends to 1 — that is, all points appear equidistant.

```{figure} ../../_static/generated/figures/en/curse_of_dimensionality.png
:name: fig-curse-dimensionality
:alt: Two plots showing how the percentage of samples near the boundary and the min/max distance ratio evolve as dimensions increase
:width: 100%
:align: center

The curse of dimensionality, quantified through simulation.
```

## Linear projection methods: PCA (Principal Components Analysis)

Principal Component Analysis (PCA), originally developed in the early 20th century by Pearson and Hotelling {cite:p}`pearson1901pca,hotelling1933pca`, is the quintessential linear dimensionality reduction algorithm. Its goal is to orthogonally project the original data $X \in \mathbb{R}^{D}$ onto a low-dimensional linear subspace $Z \in \mathbb{R}^{M}$ (where $M < D$), minimizing information loss under strict statistical criteria.

### Maximum variance perspective

From a geometric point of view, the first principal component (PC1) is defined as the direction or axis of the feature space along which the data varies the most. By projecting the data onto this axis, the largest percentage of the original information's spread is preserved.

{numref}`fig-pca-max-variance` compares two projections of the same data: onto PC1 (left) the projected points stay widely spread along the axis and 90% of the variance is preserved; onto another direction (right) the points pile up and only 13% is retained.

```{figure} ../../_static/generated/figures/en/pca_max_variance_projection.png
:name: fig-pca-max-variance
:alt: Two panels with the same data projected onto the first principal component, which retains 90% of the variance, and onto another direction, which retains only 13%
:width: 100%
:align: center

Projection onto the maximum-variance direction (PC1) versus another direction.
```

The second principal component (PC2) seeks the direction that explains as much of the remaining variance as possible, under the strict constraint of being fully orthogonal to (and therefore uncorrelated with) the first component.

This procedure is repeated successively to generate up to $D$ distinct components.

{numref}`fig-pca-directions` shows this concept on real correlated data: the red arrow (PC1) points in the direction of maximum variance and the green one (PC2), orthogonal to it, captures the remaining variance.

```{figure} ../../_static/generated/figures/en/pca_projection.png
:name: fig-pca-directions
:alt: Scatter plot of correlated data with two arrows showing the directions of the principal components PC1 and PC2
:width: 65%
:align: center

Principal components: PC1 captures the maximum variance, PC2 is orthogonal.
```

### Mathematical derivation and the SVD

To perform PCA, the original data must first be centered (subtracting the arithmetic mean of each variable) and, usually, standardized to have unit variance (preventing variables with arbitrarily large metric scales from dominating the optimization).

The covariance matrix of the data is defined as:

$S = \frac{1}{N} X X^T$

Through eigenvalue decomposition of the covariance matrix (or via Singular Value Decomposition — SVD — of the original data matrix $X$), the principal directions are extracted:

$S = V D^2 V^T$

Where the columns of matrix $V$ correspond to the loading vectors, representing the directions of the principal components. The eigenvalue $\lambda_m$ corresponding to each eigenvector directly measures the amount of variance explained by that specific component.

### Compression, reconstruction, and the linear autoencoder perspective

PCA can be conceptualized mathematically as a linear autoencoder. It consists of two main phases:

- **Encoder**: converts the input vector $x_n$ into a reduced representation $z_n = B^T x_n$, where $B$ contains the eigenvectors with the largest associated eigenvalues.
- **Decoder**: reconstructs the approximate projection back into the original space: $\hat{x}_n = B z_n$.

The average reconstruction error, or squared distortion, minimized by this procedure is exactly equal to the sum of the eigenvalues of the components that were discarded from the analysis.

A real example of PCA's usefulness appears in the *NCI60* dataset, where each sample has 6830 genes (dimensions). {numref}`fig-islr-nci60-pca` projects these samples onto their first three principal components: despite the huge original dimensionality, cell lines of the same cancer type (same color) tend to cluster together in this reduced space.

```{figure} ../../_static/book_figures/islr_fig10_15_nci60_pca.png
:name: fig-islr-nci60-pca
:alt: Two scatter plots showing the projection of the NCI60 cell lines onto the first three principal components, taken from An Introduction to Statistical Learning
:width: 85%
:align: center

Projection of the *NCI60* cancer cell lines (6830 genes) onto their first three principal components.
Source: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figure 10.15. Springer. Freely distributed for educational use (statlearning.com).
```

## Non-linear methods and manifold learning

Although PCA is highly robust and fast, it suffers from an obvious limitation: it assumes that the interesting subspace of the data is linear (a flat hyperplane). When data relationships are intrinsically non-linear, as in the classic Swiss roll example (a curved 3D sheet of paper), PCA's linear orthogonal projection collapses distant points and artificially mixes information.

### The manifold hypothesis

Manifold learning assumes the validity of the manifold hypothesis: all natural real-world data (MNIST images, faces, voices, or text) lie on a low-dimensional manifold embedded within the original high-dimensional space.

- A manifold is a continuous, curved surface that locally resembles a flat Euclidean linear space.
- For example, the space of handwritten digit images (28x28 pixels = 784 dimensions) is constrained by physical laws and invariances (stroke thickness, tilt, continuity) that greatly reduce the system's real degrees of freedom, confining viable samples to a continuous manifold of very low dimension.
- The goal of non-parametric manifold learning is to learn embedded coordinates for each point so that the manifold's internal topology and distances along its surface are faithfully represented.

{numref}`fig-swiss-roll` shows the classic Swiss roll example: the data (left) is rolled up in 3D, but its true structure is a 2D surface that can be "unrolled" (right) while preserving distances along the manifold.

```{figure} ../../_static/generated/figures/en/swiss_roll_manifold.png
:name: fig-swiss-roll
:alt: Comparison between the 3D Swiss roll data and its unrolled 2D version, colored by position along the manifold
:width: 100%
:align: center

The Swiss roll: data rolled up in 3D (left) and its unrolled manifold in 2D (right).
```

### Key non-linear algorithms

#### Kernel PCA (kPCA)

To extend PCA to non-linear problems, the kernel trick is used. kPCA implicitly maps the input data into an ultra-high (or infinite) dimensional Hilbert space $\Phi(x)$, where complex non-linear relationships become linearly separable or projectable.

- Using common kernel functions such as the RBF (Gaussian) kernel, complex projections can be modeled.
- **Drawback**: unlike standard PCA, kPCA does not define a direct, invertible projection mapping for out-of-sample data, and poorly tuned kernels can sometimes expand and distort the space rather than usefully compressing it.

{numref}`fig-kernel-pca-trick` illustrates the kernel trick with a classic example: two classes arranged in concentric circles (left) cannot be separated by a straight line in 2D. Applying the map $\phi(x_1,x_2)=(x_1,x_2,x_1^2+x_2^2)$ (right) lifts the classes to different heights, where a simple horizontal plane separates them perfectly.

```{figure} ../../_static/generated/figures/en/kernel_pca_trick.png
:name: fig-kernel-pca-trick
:alt: Two plots showing two concentric circles that are not linearly separable in 2D and their transformation into a 3D space where a plane separates them
:width: 100%
:align: center

The kernel trick: data that is not separable in 2D becomes separable once projected into an extra dimension.
```

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Proposed by Maaten and Hinton (2008) {cite:p}`vandermaaten2008tsne`, t-SNE is the preferred non-convex technique for the qualitative visualization of complex groupings in two dimensions.

1. **High-dimensional space** (original SNE): converts Euclidean distances between samples into Gaussian conditional probabilities $p_{j|i}$ denoting similarity. Nearby points receive high neighborhood probabilities and distant ones receive infinitesimal probabilities.
2. **The crowding problem**: when high-dimensional data is projected onto a flat 2D space, the available volume of space shrinks exponentially. Average distances grow so much that, using normal approximations, attraction forces push all distant points into a dense, indistinguishable core at the center of the plot.
3. **The Student's t-distribution solution**: t-SNE solves this limitation by using a Student's t-distribution with one degree of freedom (equivalent to a Cauchy distribution) in the low-dimensional space. Its much heavier tails, inverted in the denominator of the latent probability equation $q_{ij}$, eliminate unwanted attraction forces between distant clusters:

$q_{ij} = \frac{(1 + \|z_i - z_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|z_k - z_l\|^2)^{-1}}$

The gradient acts much like a physical attraction-repulsion law (similar to forces between galaxies and stars), allowing clusters to organize and separate optimally in the visual plane.

{numref}`fig-pca-vs-tsne` compares both methods on a classic example: two interleaved "moons" embedded in a 30-dimensional space. PCA (left), being a linear projection, can only rotate the data and fails to separate the two classes, which remain interleaved. t-SNE (right) rearranges the points non-linearly while preserving local neighborhoods, producing two clearly distinct groups.

```{figure} ../../_static/generated/figures/en/pca_vs_tsne.png
:name: fig-pca-vs-tsne
:alt: Comparison between PCA and t-SNE on data shaped as two interleaved moons embedded in 30 dimensions; PCA fails to separate the classes while t-SNE separates them into two distinct groups
:width: 100%
:align: center

PCA (linear projection) versus t-SNE (non-linear projection) on data with non-linear structure.
```

#### UMAP (Uniform Manifold Approximation and Projection)

UMAP {cite:p}`mcinnes2018umap` is one of the most powerful manifold learning techniques available today. Grounded in classical Riemannian geometry and algebraic topology, UMAP assumes the data space is locally connected and that the manifold on which it lies is uniform.

Unlike t-SNE, which focuses almost exclusively on retaining very local neighborhoods (short-range relationships), UMAP is able to preserve both the local and the global structure of the data.

It is mathematically much more efficient, resulting in substantially faster execution speed on massive datasets with millions of samples.

{numref}`fig-umap-digits` shows a 2D UMAP projection of the handwritten digits dataset (64 dimensions): each digit (0-9, one color per class) forms a compact, well-differentiated group, and the relative position between groups is also informative.

```{figure} ../../_static/external_images/umap_digits_projection.png
:name: fig-umap-digits
:alt: Two-dimensional UMAP projection of the handwritten digits dataset, with ten well-separated colored groups, one per digit
:width: 75%
:align: center

UMAP projection of the *Digits* dataset.
Source: umap-learn documentation, "How to Use UMAP" (umap-learn.readthedocs.io). Copyright (c) 2017, Leland McInnes. BSD 3-Clause License.
```

## Hands-on example in Java with SMILE

The [programacion-avanzada-smile](https://github.com/dmjimenezbravo/programacion-avanzada-smile) repository includes the full example [`Ejemplo06PCA.java`](https://github.com/dmjimenezbravo/programacion-avanzada-smile/blob/main/src/main/java/es/usal/smile/Ejemplo06PCA.java), which applies PCA to the *Iris* dataset (four variables) with the [SMILE](https://haifengl.github.io/) library. The following fragment summarizes its main steps (comments translated into English).

```java
import java.util.Arrays;
import smile.classification.KNN;
import smile.data.DataFrame;
import smile.feature.extraction.PCA;
import smile.io.Read;
import smile.tensor.Vector;
import smile.validation.metric.Accuracy;

DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");
double[][] x = iris.drop("species").toArray();
int[] y = iris.column("species").toIntArray();

// 1. Fit PCA on the covariance matrix.
//    PCA.cor(x) would use the correlation matrix (equivalent to standardizing first).
PCA pca = PCA.fit(x);

// 2. Variance explained by each component (the "scree plot" as a table)
Vector proporcion = pca.varianceProportion();
Vector acumulada = pca.cumulativeVarianceProportion();
for (int i = 0; i < proporcion.size(); i++) {
    System.out.printf("PC%d: %.2f%% (cumulative %.2f%%)%n",
            i + 1, 100 * proporcion.get(i), 100 * acumulada.get(i));
}
System.out.println(pca.loadings());   // loadings: weight of each variable in each component

// 3. Projection from 4 dimensions down to 2
PCA proyeccion2D = pca.getProjection(2);
double[][] x2 = proyeccion2D.apply(x);
System.out.println("First projected sample: " + Arrays.toString(x2[0]));

// Alternative: ask for the components needed to retain 95% of the variance
int componentes95 = pca.getProjection(0.95).apply(x)[0].length;

// 4. PCA as a preprocessing step before a classifier.
//    PCA is fit ONLY on the training set and then applied to the test set:
//    fitting it on all the data would be information leakage.
Utiles.ParticionArrays particion = Utiles.split(x, y, 0.7);
PCA pcaTrain = PCA.fit(particion.xTrain()).getProjection(2);
double[][] xTrain2 = pcaTrain.apply(particion.xTrain());
double[][] xTest2 = pcaTrain.apply(particion.xTest());

KNN<double[]> knnCompleto = KNN.fit(particion.xTrain(), particion.yTrain(), 5);
KNN<double[]> knnReducido = KNN.fit(xTrain2, particion.yTrain(), 5);
System.out.printf("KNN with 4 variables   accuracy = %.2f%%%n",
        100.0 * Accuracy.of(particion.yTest(), knnCompleto.predict(particion.xTest())));
System.out.printf("KNN with 2 components  accuracy = %.2f%%%n",
        100.0 * Accuracy.of(particion.yTest(), knnReducido.predict(xTest2)));
```

A few things worth noticing when you run it:

- `varianceProportion()` and `cumulativeVarianceProportion()` are the numerical equivalent of the scree plot and help decide how many components to keep.
- `getProjection(2)` fixes the number of components, whereas `getProjection(0.95)` chooses it automatically from the amount of variance to retain.
- When PCA is used as preprocessing, fitting it on all the data (training and test) would be **information leakage**: PCA must be fit on the training set only.
- The full example also includes kernel PCA (`KernelPCA` with a Gaussian kernel) and probabilistic PCA; SMILE additionally offers $t$-SNE and UMAP in the `smile.manifold` package.

To run it from the root of the repository: `mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo06PCA`.

## Summary

- **PCA**: fast, linear, interpretable original features.
- **t-SNE**: excellent visualization, non-linear, slow for large data.
- **UMAP**: a balance between PCA and t-SNE.
- **Explained variance**: a metric for choosing the number of components.
- **Curse of dimensionality**: the motivation for reduction.
- **Applications**: compression, visualization, preprocessing.

---

**Next**: In 3.3 we will learn to detect anomalies, the final topic in unsupervised learning.
