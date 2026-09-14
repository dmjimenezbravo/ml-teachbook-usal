# 2.2 Classification

## Introduction

In classification, our goal is to predict **categories** (classes) rather than continuous values. To classify a sample or instance with a specific class, classification algorithms rely on the sample's attributes. Some typical examples include:

- Email classification: spam vs. not-spam.
- Medical classification: sick vs. healthy.
- Species classification: cat vs. dog, etc.
- Etc.

## Fundamentals and logistic regression

Unlike linear regression, which predicts continuous values, classification starts by estimating the probability of belonging to a category.

- **Model mechanics**: logistic regression uses the **sigmoid function** to transform a linear combination of the input features into a value between 0 and 1. The formula is defined as:
$\sigma(t) = \frac{1}{1 + e^{-t}}$

The input t of this function is called the **logit** (or log-odds), representing the unnormalized log-probabilities of the positive class.
- **Training and optimization**: the model is trained by minimizing a cost function called **cross-entropy** (or log loss), which penalizes predictions that are confident but wrong. Since this function is convex, gradient descent can be used to find the optimal weights iteratively.
- **Evaluation via confusion matrix**: a $K \times K$ table that breaks down the model's performance by comparing actual labels (rows) with predicted labels (columns). Four critical values arise from this:
  - **True Positives** (TP) and **True Negatives** (TN): the model's correct predictions.
  - **False Positives** (FP): Type I error or false alarm.
  - **False Negatives** (FN): Type II error or miss.

{numref}`fig-sigmoid` shows the characteristic "S" shape of the sigmoid function: very negative inputs are squashed toward 0 and very positive inputs toward 1, with the decision point at 0.5.

```{figure} ../../_static/generated/figures/en/sigmoid_function.png
:name: fig-sigmoid
:alt: Plot of the sigmoid function showing how it transforms any real value into a probability between 0 and 1
:width: 75%
:align: center

Sigmoid function used in logistic regression.
```

{numref}`fig-islr-default` shows why this transformation is necessary: on the *Default* dataset, a linear regression fit (left) predicts absurd probabilities outside the [0, 1] range, while logistic regression (right) always produces valid probabilities.

```{figure} ../../_static/book_figures/islr_fig4_2_default_logistic.png
:name: fig-islr-default
:alt: Two plots comparing linear regression and logistic regression on the Default dataset, taken from An Introduction to Statistical Learning
:width: 90%
:align: center

Linear regression (left) vs. logistic regression (right) on the *Default* dataset.
Source: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figure 4.2. Springer. Freely distributed for educational use (statlearning.com).
```

## Classic classification algorithms

These methods stratify the feature space or search for geometric separation boundaries.

### Decision Trees

Trees recursively partition the input space into high-dimensional "boxes" or rectangles.

- **Construction**: they follow a "divide and conquer" approach via the **CART algorithm**, which performs binary splits seeking to maximize node purity (reducing impurity metrics such as the Gini index or entropy).
- **Pruning**: to avoid overfitting, cost-complexity pruning is used, which removes unnecessary branches based on statistical tests or validation sets.

{numref}`fig-tree-partition` illustrates how a tree divides the feature space into rectangular regions through successive axis-aligned cuts.

```{figure} ../../_static/generated/figures/en/decision_tree_partition.png
:name: fig-tree-partition
:alt: Scatter plot showing how a decision tree splits the space into rectangular regions via cuts on X and Y
:width: 65%
:align: center

Recursive partitions of a decision tree over two features.
```

### Random Forests
An ensemble of decision trees designed to reduce variance and improve robustness.

- **Bagging**: uses sampling with replacement (bootstrap) to train each tree on a different version of the data.
- **Decorrelation**: to ensure the trees are diverse, only a random subset of features is considered at each split.

{numref}`fig-bagging` sketches the full process: several bootstrap samples are drawn from the original dataset, each trains a different tree (also using a random subset of features at each split), and the individual predictions are combined through majority voting (classification) or averaging (regression).

```{figure} ../../_static/generated/diagrams/en/01_supervised_learning_02_classification_01.svg
:name: fig-bagging
:alt: Flowchart showing how bagging generates bootstrap samples, trains one tree per sample, and combines their predictions via voting
:width: 90%
:align: center

Bagging: each tree is trained on a different bootstrap sample and predictions are combined by voting.
```

### Support Vector Machines (SVM)

This model seeks to find a **hyperplane** that separates the classes with the **maximum margin** possible.

- **Support vectors**: the decision boundary is determined solely by the samples closest to the hyperplane; moving other points does not change the model.
- **Soft margin and the kernel trick**: SVMs handle non-linearly separable data through slack variables and the kernel trick, which implicitly maps the data to higher-dimensional spaces to find complex separations.

{numref}`fig-svm-margin` shows the optimal hyperplane (solid line), the maximum margin (gray band), and the support vectors (circled in green), which are the only points that determine the boundary's position.

```{figure} ../../_static/generated/figures/en/svm_margin.png
:name: fig-svm-margin
:alt: Scatter plot with two classes separated by a hyperplane, showing the maximum margin and the circled support vectors
:width: 65%
:align: center

SVM: maximum-margin hyperplane and support vectors.
```

## Advanced ensemble methods

These combine multiple models to obtain a prediction superior to that of any individual model.

- **Voting** and **stacking**: majority (hard) voting or averaging probabilities (soft voting) combines independent models. Stacking trains a meta-model that learns to combine the outputs of the base models.
- **Boosting**: unlike the parallel training of random forests, boosting trains models sequentially, where each new predictor tries to correct the errors made by its predecessors. Modern examples include XGBoost and Gradient Boosting.

{numref}`fig-boosting` illustrates boosting's sequential training: each weak model is trained on data reweighted according to the previous model's errors, giving more weight to misclassified instances, and the final prediction combines all models through a weighted sum.

```{figure} ../../_static/generated/diagrams/en/01_supervised_learning_02_classification_02.svg
:name: fig-boosting
:alt: Flowchart showing boosting's sequential training, where each model corrects the errors of the previous one before being combined into a weighted sum
:width: 100%
:align: center

Boosting: models are trained sequentially, each one correcting the previous one's errors.
```

## Introduction to neural networks

These represent learning through successive layers of filtering representations.

- **Architecture**: composed of an input layer, multiple hidden (densely connected) layers, and an output layer.
- **Activation functions**: introduce non-linearity to learn complex patterns. **ReLU** is the standard for hidden layers, while **Softmax** is used in the output layer for multi-class classification.
- **Backpropagation**: the central learning mechanism; it uses the chain rule from calculus to propagate the error backward from the output, adjusting the network's weights to reduce the total loss.

{numref}`fig-nn-architecture` sketches a fully-connected network with an input layer, a hidden layer, and an output layer: each connection represents a weight that gets adjusted during training.

```{figure} ../../_static/generated/figures/en/nn_architecture.png
:name: fig-nn-architecture
:alt: Diagram of a fully connected neural network with an input layer, a hidden layer, and an output layer
:width: 70%
:align: center

Architecture of a fully-connected neural network.
```

## Evaluating classification performance

Evaluation is the fundamental process for measuring a model's generalization ability, i.e., its aptitude for making accurate predictions on data it has never seen before. A model that performs exceptionally on training data but fails on new data is suffering from overfitting and lacks practical utility.

### Validation protocols and data splitting

To measure generalization reliably, it is mandatory to split the available dataset into independent partitions.

- **Training**, **validation**, and **test**: in situations of data abundance, a training set is reserved to fit the model's weights, a validation set to select the best architecture or hyperparameters, and a test set that is kept in a "vault" for a final, unbiased evaluation.
- **$K$-fold Cross-Validation**: the recommended strategy when data is scarce. It splits the data into $K$ partitions; the model is trained $K$ times, each time using a different partition for validation and the remaining $K-1$ for training. The final result is the average of the scores obtained across the $K$ experiments, which reduces the variance of the estimate.
- **Stratification**: in classification tasks, it is vital to ensure that each partition preserves the original class proportion, especially when classes are imbalanced.

### Success metrics for classifiers

The choice of metric must align with the business objective and the nature of the data.

#### The confusion matrix
A two-dimensional tool where rows represent the true classes and columns the predicted ones. It allows four essential values to be identified:

- **True Positives** (TP): positive instances correctly classified.
- **True Negatives** (TN): negative instances correctly classified.
- **False Positives** (FP): Type I error or "false alarm."
- **False Negatives** (FN): Type II error or "miss."

#### Derived metrics
- **Accuracy**: the fraction of correct predictions over the total. Can be misleading on skewed datasets; for example, if 90% of samples belong to one class, a model that always predicts that class will have 90% accuracy without having learned anything.
- **Precision**: the model's ability to avoid labeling a negative sample as positive:
$\frac{TP}{TP+FP}$
- **Recall**: the ability to find all the actual positive samples:
$\frac{TP}{TP+FN}$
- **F1-score**: the harmonic mean of precision and recall, useful when seeking a balance between the two.
- **ROC AUC**: the area under the receiver operating characteristic curve measures the probability that the model ranks a random positive sample above a random negative one. A value of 1.0 is perfect and 0.5 is equivalent to random classification.

{numref}`fig-roc-curve` compares a good classifier (curve far from the diagonal, high AUC) against the behavior of a random classifier (diagonal line, AUC = 0.5).

```{figure} ../../_static/generated/figures/en/roc_curve.png
:name: fig-roc-curve
:alt: ROC curve showing the true positive rate against the false positive rate, compared with the diagonal of a random classifier
:width: 65%
:align: center

ROC curve and area under the curve (AUC).
```

### Training and diagnostic strategies
Evaluation does not only happen at the end — it should guide the entire development process.

- **Beating a baseline**: before starting with complex models, a trivial baseline (such as a random classifier) should be established. If the model cannot beat this common-sense threshold, it is likely that the data does not contain enough information, or the approach is flawed.
- **Monitoring learning curves**: plotting loss and accuracy for both training and validation makes it possible to detect the exact point of overfitting (when training loss decreases but validation loss starts to rise).
- **Early stopping**: a strategy that uses a callback to automatically halt training when the validation metric stops improving, saving time and preventing overfitting.
- **Error analysis**: manually inspecting the samples where the model fails helps understand which specific patterns the system is confusing (for example, confusing a "3" with a "5" in digit recognition).
- **A/B testing**: after deployment, randomized tests are recommended to measure the model's real impact compared to the previous process.

## Summary

- **Logistic Regression**: an extension of linear regression for classification.
- **Confusion Matrix**: visualizes performance.
- **Metrics**: choose based on what matters (precision or recall).
- **Decision Trees**: interpretable but prone to overfitting.
- **Random Forest**: an ensemble that reduces overfitting.
- **SVM**: finds an optimal separation using the kernel trick.
- **Ensembles**: combine several classification algorithms.
- **Neural Networks**: powerful but complex.

---

**Next**: in Section 3 we will explore unsupervised learning, where we work with unlabeled data.
