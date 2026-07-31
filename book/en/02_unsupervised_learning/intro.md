# Section 3: Unsupervised Learning

Welcome to the third and final main section of the course. Here we'll explore techniques to extract patterns from **unlabeled data**.

## What is Unsupervised Learning?

In unsupervised learning:
- We have **only features** (X)
- We **don't have labels** (no y)
- We want to **discover hidden patterns**

It's like exploring unknown territory without a map. The machine must find its own structure.

## Three Main Types

### 1. Clustering
Grouping similar data:
- Segment customers by behavior
- Find communities in social networks
- Classify documents by topic
- Identify cancer cells in images

### 2. Dimensionality Reduction
Simplifying complex data:
- Visualizing high-dimensional data
- Removing noise
- Accelerating later algorithms
- Interpretable understanding

### 3. Anomaly Detection
Identifying rare or unusual cases:
- Fraud in transactions
- System failures
- Unusual user behavior
- Scientific data outliers

## What You'll Find in This Section

### Chapter 3.1: Clustering
- **K-Means**: partition data into groups
- **Elbow Method**: select optimal number of clusters
- **Hierarchical Clustering**: dendrograms and merging
- **DBSCAN**: density-based clustering

### Chapter 3.2: Dimensionality Reduction
- **PCA (Principal Component Analysis)**: maximize explained variance
- **t-SNE and UMAP**: non-linear visualization

### Chapter 3.3: Anomaly Detection
- **Identifying unusual patterns**: what is an anomaly
- **Detection methods**: practical applications

## Main Challenge: Evaluation

Without labels, how do we know our results are correct?

```
Supervised Learning:
Data → Model → Prediction → Compare with y → Error

Unsupervised Learning:
Data → Model → Structure ??? → Is it correct?
```

This is more challenging and requires:
- **Domain intuition**: Does the result make sense?
- **Internal metrics**: silhouette, Davies-Bouldin index
- **External validation**: labels available later (if any)

## Learning Objectives

After completing this section, you'll be able to:
- ✓ Implement K-Means from scratch
- ✓ Choose the optimal number of clusters
- ✓ Visualize high-dimensional data
- ✓ Detect anomalies in real data
- ✓ Understand the limitations of each technique

---

**Let's start**: Go to Chapter 3.1 to learn about Clustering, the most popular unsupervised approach.
