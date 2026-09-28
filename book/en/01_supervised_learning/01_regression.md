# 2.1 Regression

## Introduction

Regression is the problem of predicting a **continuous value**. In other words, regression predicts a quantitative response or continuous scalar value $Y$ from one or more **predictor variables** $X$.

## Linear regression

Linear regression is a fundamental technique that assumes an approximately linear relationship between the predictors and the target variable.

  - **Simple vs. multiple**: **simple linear regression** uses a single predictor to model $Y \approx \beta_0 + \beta_1X$. **Multiple linear regression** extends this concept by assigning a specific coefficient or weight to each of the p available features: $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon$.
  - **Model parameters**: the coefficients $\beta_0$ (intercept or bias) and $\beta_1...\beta_p$ (weights or slopes) represent the knowledge the model has learned from the data. Technically, in a p-dimensional input space, this model represents a **hyperplane**.

{numref}`fig-linear-fit` shows an example with a single predictor: the red line is the OLS fit and the gray vertical lines represent the residuals (the distance between each point and the prediction).

```{figure} ../../_static/generated/figures/en/linear_regression_fit.png
:name: fig-linear-fit
:alt: Scatter plot with a fitted linear regression line and residuals marked as vertical lines between each point and the line
:width: 80%
:align: center

Simple linear regression: fitted line and residuals.
```

A classic example from the literature {cite:p}`james2013islr` is the *Advertising* dataset, where a line is fit to predict `Sales` from TV advertising spending (`TV`):

```{figure} ../../_static/book_figures/islr_fig3_1_advertising.png
:name: fig-islr-advertising
:alt: Scatter plot of sales versus TV advertising spending with a fitted regression line, taken from An Introduction to Statistical Learning
:width: 75%
:align: center

Least squares fit of `Sales` on `TV` in the *Advertising* dataset.
Source: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figure 3.1. Springer. Freely distributed for educational use (statlearning.com).
```

## Ordinary least squares

The ordinary least squares (OLS) method is the standard approach for training linear models, seeking the parameters that best fit the training data.

  - **Residuals and RSS**: a residual $e_i = y_i - \hat{y}_i$ is the difference between the actual value and the predicted one. The goal of OLS is to minimize the **sum of squared residuals** (RSS, Residual Sum of Squares): $RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$.
  - **Analytical solution** (normal equations): unlike other complex models, the linear model has a unique, global solution that can be computed directly via matrix algebra: $\hat{\beta} = (X^TX)^{-1}X^Ty$.
  - **Geometric interpretation**: the prediction vector $\hat{y}$ is the **orthogonal projection** of the observation vector y onto the linear subspace spanned by the columns of the feature matrix $X$.

## Gradient descent

For massive datasets that do not fit in memory, or models with thousands of parameters, solving the normal equation is computationally expensive. Gradient descent (GD) is the iterative optimization algorithm that powers modern machine learning.

  - **Optimization mechanics**: the algorithm initializes the weights randomly and gradually adjusts them by measuring the local gradient of the loss function with respect to the parameters. It moves in the direction opposite the gradient (moving "down" the error curve) until it reaches a minimum.
  - **Learning rate** ($\eta$): a critical hyperparameter that determines the step size at each iteration. A rate that is too high can cause divergence, while one that is too low will make training excessively slow.
  - Variants:
    - **Batch GD**: uses the entire training set for each step.
    - **Stochastic GD** (SGD): uses a single random sample per step, making it much faster and able to handle online data streams.
    - **Mini-batch GD**: processes small groups of data (batches), balancing the stability of Batch GD with the efficiency of SGD.
  - **Momentum**: a technique inspired by physics that helps avoid local minima and accelerates convergence by taking previous weight updates into account.

As shown in {numref}`fig-gradient-descent`, the algorithm starts at an initial point and takes successive steps in the direction of steepest descent until it converges to the minimum of the loss function.

```{figure} ../../_static/generated/figures/en/gradient_descent.png
:name: fig-gradient-descent
:alt: Loss function curve with gradient descent steps converging toward the minimum
:width: 75%
:align: center

Gradient descent: iterative steps toward the minimum of the loss.
```

## Polynomial regression and regularization

When data exhibits curvature, linear models can be extended through feature engineering.

  - **Polynomial regression**: powers of the original predictors are added (e.g., $X^2$, $X^3$) as new features. Although the model is non-linear with respect to the original inputs, it remains linear in its parameters, which allows OLS to be used for training.
  - **The risk of overfitting**: increasing the polynomial degree increases the model's flexibility, which can lead it to memorize the noise in the training data (overfitting) and fail to generalize.
  - **Regularization techniques**: these add a penalty to the cost function for having large weights, forcing the model to be simpler.
    - **Ridge** (L2) {cite:p}`hoerl1970ridge`: adds a penalty proportional to the square of the weights. It shrinks coefficients toward zero but never eliminates them entirely; it works very well when there are many correlated predictors.
    - **Lasso** (L1) {cite:p}`tibshirani1996lasso`: adds a penalty proportional to the absolute value of the weights. It has the unique property of forcing some coefficients to be exactly zero, automatically performing variable selection.
    - **Elastic Net**: a combination of Ridge and Lasso that uses both types of penalty via a mixing ratio.

{numref}`fig-regularization-paths` compares how a model's coefficients evolve as the penalty $\lambda$ increases: Ridge shrinks them smoothly without ever reaching zero, while Lasso drives them exactly to zero, performing variable selection.

```{figure} ../../_static/generated/figures/en/regularization_paths.png
:name: fig-regularization-paths
:alt: Comparison of Ridge and Lasso coefficient paths as the penalty lambda increases
:width: 90%
:align: center

Coefficient paths: Ridge (smooth shrinkage) vs. Lasso (variable selection).
```

On real data, {numref}`fig-islr-ridge-credit` shows how the standardized Ridge coefficients evolve for the *Credit* dataset, both as a function of $\lambda$ (left) and of the relative coefficient norm (right).

```{figure} ../../_static/book_figures/islr_fig6_4_ridge_credit.png
:name: fig-islr-ridge-credit
:alt: Two plots showing the evolution of Ridge regression coefficients for the Credit dataset, taken from An Introduction to Statistical Learning
:width: 90%
:align: center

Standardized Ridge regression coefficients for the *Credit* dataset.
Source: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figure 6.4. Springer. Freely distributed for educational use (statlearning.com).
```

## Validation and diagnostics

Evaluating a regression model requires specific metrics and analysis of fit quality.
  - Error metrics:
    - **MSE** (Mean Squared Error): the average of squared errors; strongly penalizes large deviations.
    $MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
    - **RMSE** (Root MSE): the square root of MSE, which returns the error to the same units as the target variable.
    $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
    - **MAE** (Mean Absolute Error): the average of the absolute values of the errors; more robust to outliers.
    $MAE = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$
  - **Coefficient of determination** (R2): measures the proportion of the variance in $Y$ that is explained by the model. A value of 1 indicates a perfect fit, while 0 indicates the model is no better than always predicting the mean.
  - **Residual analysis**: plotting residuals against fitted values makes it possible to detect problems such as non-linearity. If the plot shows a pattern (such as a U-shaped curve), it indicates the linear model is insufficient.
  - **Validation curves**: validation curves are fundamental graphical tools for diagnosing model behavior as a function of complexity.
    - **Bias-variance diagnosis**: by plotting training and validation error against model complexity (e.g., polynomial degree), we can identify:
      - **High bias**: high error in both training and validation.
      - **High variance**: very low training error but very high validation error; there is a significant gap between the two curves.
    - **One-standard-error rule**: in practice, it is common to choose the simplest model that falls within one standard deviation of the minimum error on the validation curve, to ensure parsimony.

## Summary

- **Linear Regression**: a simple yet powerful model.
- **OLS**: an exact solution, but computationally expensive.
- **Gradient Descent**: a general method that works for any model.
- **Polynomial Regression**: captures non-linear relationships.
- **Regularization**: prevents overfitting (Ridge for retention, Lasso for selection).
- **Validation**: specific metrics for validating regression, together with a study of the training process.

---

**Next**: in the next chapter, we will extend these ideas to **Classification**, where we predict categories instead of continuous numbers.
