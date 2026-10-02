# Fundamental Concepts

## Introduction

To work effectively with machine learning models, we need to master certain terminology and key concepts. This chapter builds the vocabulary we will use throughout the course.

## Data, features, and labels

The success of any ML system depends on how information is represented.

```
+----------+---------+---------+---------+--------+
| Sample   | Feature | Feature | Feature | Label  |
|          | 1       | 2       | 3       |        |
+----------+---------+---------+---------+--------+
| Example  | Age     | Weight  | Height  | Type   |
| 1        | 25      | 70      | 175     | Cat    |
| 2        | 8       | 4       | 90      | Cat    |
| 3        | 35      | 80      | 180     | Dog    |
+----------+---------+---------+---------+--------+
```

- **Sample**: the basic unit of information, also called a data point, instance, or example. In a tabular dataset, each row represents a single sample.
- **Attributes and features**: samples are characterized by their features (attributes), which are quantitative or qualitative variables measuring different aspects of the instance. Technically, an "attribute" is a data type (e.g., "age"), while a "feature" usually refers to the attribute together with its specific value (e.g., "age = 25"). In deep learning, all inputs are vectorized so they can be processed as points in a geometric space.
- **Label** or target: in supervised learning, each sample has an associated correct answer called a label or target. The set of labels for an entire dataset constitutes the **"ground-truth."** In classification problems, the goal is to predict a category (class), while in regression the goal is a continuous numeric or scalar value.

```python
# Features: can be numbers, categories, etc.
features = [
    "square_meters",
    "number_of_rooms",
    "location",
    "year_built"
]

# Label: what we want to predict
label = "price"
```

## The tension between optimization and generalization: overfitting and underfitting

The fundamental problem of ML is the tension between fitting the model to known data and its ability to perform on new data.

- **Optimization**: the process of adjusting a model's parameters (weights) to obtain the best possible performance on the training data.
- **Generalization**: refers to how well the trained model performs on data it has never seen before.
- **Overfitting**: occurs when a model is too complex relative to the amount and noise of the data. The model "memorizes" training noise or random patterns that do not exist in the real data, resulting in low training loss but very high validation loss.
- **Underfitting**: happens when the model is too simple to capture the underlying structure of the data. In this state, both training and validation error are high.
- **Bias-variance trade-off**: generalization error is split into:
  - **Bias**: error from incorrect assumptions (e.g., assuming a linear relationship when it is actually quadratic); high bias causes underfitting.
  - **Variance**: error from excessive sensitivity to small variations in the training data; high variance causes overfitting.

{numref}`fig-overfitting-underfitting` illustrates the three scenarios by fitting different models to the same data: a line that is too simple (left), a moderate-degree polynomial that follows the true trend (center), and a very high-degree polynomial that memorizes every point, including the noise (right).

```{figure} ../../_static/generated/figures/en/overfitting_underfitting.png
:name: fig-overfitting-underfitting
:alt: Comparison of underfitting, a good fit, and overfitting using three polynomial models fitted to the same data
:width: 100%
:align: center

Underfitting (high bias) vs. good fit vs. overfitting (high variance).
```

## Evaluation protocols: cross-validation

To measure generalization reliably, it is not enough to evaluate the model on the same data it was trained on.

- **Data splitting**: standard practice is to split the data into three sets: training (to learn the weights), validation (to choose hyperparameters and avoid information "leakage"), and test (for a final, unbiased evaluation).
- **$K$-fold cross-validation**: when data is scarce, a simple split can be unrepresentative. This method splits the data into $K$ partitions (typically 5 or 10). The model is trained $K$ times; each iteration uses a different partition for validation and the remaining $K-1$ for training. The final score is the average of the $K$ results obtained, which reduces the variance of the evaluation.
- **Stratification**: in classification, it is vital that each "fold" preserves the same class proportions as the original dataset to avoid bias, a process known as stratified $K$-fold.

{numref}`fig-kfold-cv` shows an example with $K=5$: in each row (each "fold"), a different block of data acts as the validation set (in red) while the rest is used for training (in blue).

```{figure} ../../_static/generated/figures/en/kfold_cross_validation.png
:name: fig-kfold-cv
:alt: Diagram of 5-fold cross-validation showing which block of data is used for validation in each of the five iterations
:width: 90%
:align: center

$K$-fold cross-validation with $K=5$: each block of data acts once as the validation set.
```

## Performance metrics

Metrics allow us to quantify the model's success and guide technical decisions.

- **Confusion matrix**: a two-dimensional tool used to evaluate a classifier's performance in detail. Its structure is a table where each row represents the true class (ground-truth) and each column represents the class predicted by the model. In a binary classification problem (two classes), the matrix contains four fundamental values that break down the system's hits and misses:
  - **True Positives** (TP): instances the model correctly classified as positive.
  - **True Negatives** (TN): instances the model correctly classified as negative.
  - **False Positives** (FP): occurs when the model incorrectly predicts that an instance is positive when it is actually negative. This type of error is also known as a Type I error or "false alarm."
  - **False Negatives** (FN): occurs when the model incorrectly predicts that an instance is negative when it is actually positive. Also called a Type II error or "miss."

{numref}`fig-confusion-matrix` visually summarizes these four values: the main diagonal (green) represents the model's hits, while the secondary diagonal (red) represents its two types of error.

```{figure} ../../_static/generated/figures/en/confusion_matrix.png
:name: fig-confusion-matrix
:alt: 2x2 confusion matrix with the cells True Positive, False Negative, False Positive, and True Negative
:width: 65%
:align: center

Confusion matrix: hits (green) and Type I and Type II errors (red).
```

- **Accuracy**: the fraction of correct predictions over the total. Can be misleading if classes are imbalanced.
        $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: the model's ability to avoid labeling a negative sample as positive.
        $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: the model's ability to find all the actual positive samples (TP/(TP+FN)).
        $\text{Recall} = \frac{TP}{TP + FN}$
- **F1-score**: the harmonic mean of Precision and Recall; useful when seeking a balance between the two, penalizing extreme values.
        $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **AUC-ROC**: the area under the receiver operating characteristic curve represents the probability that the model ranks a random positive sample above a random negative one. An AUC of 1.0 is perfect, while 0.5 is equivalent to random classification.

## The ML pipeline: the universal workflow

Developing a Machine Learning project follows a universal blueprint that ensures the system not only learns from the data but is also able to generalize its results in production environments. {numref}`fig-pipeline-ml` summarizes the 10 stages and their cyclical nature: hyperparameter tuning usually requires retraining the model several times before the final evaluation.

```{figure} ../../_static/generated/diagrams/en/00_fundamentals_02_fundamental_concepts_01.svg
:name: fig-pipeline-ml
:alt: Flowchart of the machine learning pipeline with its 10 stages, from data collection to deployment, including the retraining loop
:width: 100%
:align: center

The Machine Learning pipeline: 10 stages from data collection to deployment.
```

Below, the stages of the ML pipeline are defined, integrating the theoretical and technical foundations covered so far:

1. **Data collection**: this is the most arduous and costly phase, requiring an understanding of the problem domain and business objectives. It consists of obtaining representative samples from the real environment, which often involves manual annotation or labeling by human experts to generate the ground-truth needed for supervised learning.
2. **Exploration and cleaning**: it is bad practice to treat the dataset as a black box; the data distribution should be visualized using histograms or maps to detect anomalies and outliers. Cleaning includes handling missing values (either by removing records or imputing averages) and detecting labeling errors to prevent noise from degrading model performance.
3. **Feature engineering**: applying human knowledge to perform non-learned transformations that make the algorithm's task easier, such as normalizing numeric scales or standardizing text to remove encoding differences. Although Deep Learning automates much of this process, good feature engineering makes it possible to solve problems with far less data and computational resources.
4. **Train/test split**: to evaluate generalization ability, the data must be strictly split into training, validation, and test sets. Random shuffling is critical to ensure representativeness, except for time series, where the test set must be chronologically later than the training set to avoid leaking information from the future.
5. **Model selection**: at this stage, the appropriate architecture priors are chosen for the task, such as convolutional networks for images or Transformers for natural language. The first technical goal is to develop a simple model that beats a common-sense baseline, demonstrating that there is an exploitable statistical pattern in the data.
6. **Training**: training is an iterative process called the training loop, where the model's weights are randomly initialized and gradually adjusted. Through the forward pass, loss computation, and gradient backpropagation, the system optimizes its internal parameters to minimize error on the training data.
7. **Evaluation**: the validation set is used to monitor the model's performance during training using metrics such as precision, recall, or AUC. This phase makes it possible to detect whether the model is underfitting or has started memorizing the noise in the training data.
8. **Hyperparameter tuning**: hyperparameters are external configurations (such as the number of layers or the learning rate) that are not learned via gradient descent but rather through systematic search. Techniques such as $K$-fold cross-validation or Bayesian optimization are used to find the configuration that maximizes generalization before the final evaluation.
9. **Final evaluation**: once the best model has been selected and its hyperparameters tuned, a single evaluation is performed on the test set, which must have remained in a "vault" without being consulted beforehand. If results here are significantly worse than in validation, it is a sign that overfitting occurred to the validation process or that the data lacked representativeness.
10. **Deployment**: the final model is exported for production use, whether as a web API, in mobile applications, or on embedded devices. The pipeline does not end here; it is imperative to implement constant monitoring, since real-world data usually degrades over time (concept drift), requiring periodic retraining cycles.

## Summary

- **Sample**: a single point in the dataset.
- **Features**: input variables that describe a sample.
- **Label**: the output variable we want to predict.
- **Overfitting**: memorizing training data, poor on test data.
- **Underfitting**: model too simple, poor on both.
- **Cross-Validation**: a technique for evaluating performance without bias.
- **Precision**: ratio of correctly predicted positives.
- **Recall**: ratio of actual positives found.
- **ML Pipeline**: composed of 10 stages, from data collection to production deployment.

---

**Next**: now that you understand the fundamentals, we move on to Section 2: Supervised Learning, where you will learn to build practical models.
