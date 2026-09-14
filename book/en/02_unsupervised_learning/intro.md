# Section 3: unsupervised learning

Welcome to the third and final main section of the course. Here we will explore techniques for extracting patterns from **unlabeled data**.

## What is unsupervised learning?

Unlike the previous blocks focused on supervised learning, where training is guided by a clear target variable or label, unsupervised learning represents the challenging task of dealing with datasets that lack any kind of expected output or annotation. In this paradigm, we only have a collection of input features $X$ (or predictor variables), but we do not have a "teacher" or supervisor providing the correct answers to guide the learning process.

Technically, while supervised learning seeks to approximate a direct mapping function between inputs and outputs, or to estimate a **conditional distribution** $p(y|x)$, unsupervised learning works "blindly" to discover the **latent structure**, the intrinsic statistical associations, or the **unconditional distribution** of the data itself, $p(x)$. As computer scientist Yann LeCun famously illustrated: "If intelligence is a cake, unsupervised learning is the cake, supervised learning is the icing, and reinforcement learning is the cherry on top." This analogy underscores that the vast majority of information available in the real world is unlabeled, making this paradigm the fundamental pillar on which true data understanding is built.

### Main types of unsupervised learning

1. **Clustering**: aims to group similar data together.
2. **Dimensionality reduction**: simplifies complex data.
3. **Anomaly detection**: identifies rare or unusual cases.

## Practical examples and uses

Unsupervised learning does not seek to make pointwise quantitative or qualitative predictions. Its value lies in extracting knowledge, simplifying structures, and revealing hidden patterns so they can be interpreted by humans or serve as preparation for other models:

- **Market segmentation**: by analyzing large volumes of demographic and transactional consumer data (such as income, occupation, or spending habits), companies can group people into homogeneous niches to target specific advertising campaigns or develop personalized products without needing to define these categories in advance.
- **Visualizing complex data**: projecting extremely high-dimensional datasets (with hundreds of attributes) into readable two- or three-dimensional representations, preserving the structure and geometric relationships so an analyst can intuitively understand how the data is organized.
- **Recommendation systems and collaborative filtering**: grouping users with similar tastes or products with similar characteristics to predict implicit preferences based on historical consumption behavior.
- **Anomaly detection**: modeling the statistical behavior of "normal" data to automatically identify suspicious instances that deviate significantly from the general pattern (for example, to detect financial fraud or manufacturing line failures).

## What Will You Find in This Section?

- **Clustering**: focused on methods designed to find homogeneous subgroups within the observations.
  - Classic centroid-based algorithms, such as $K$-means, will be covered.
  - Hierarchical merging structures, or hierarchical clustering.
  - Models that define groups based on the local density of the geometric space.
- **Dimensionality reduction**: linear projection tools for simplifying data will be explored.
  - Techniques that maximize retained variance, such as PCA.
  - Advanced non-linear manifold learning algorithms, such as t-SNE and UMAP.
  - Algorithms specifically aimed at qualitative visualization and noise removal before training supervised classifiers.

## Important Note

Unsupervised learning is more **art than science**. There is no single "correct" answer. Your domain interpretation and external feedback are crucial.

---

**Let's start**: go to Chapter 3.1 to learn about clustering, the most popular unsupervised approach.
