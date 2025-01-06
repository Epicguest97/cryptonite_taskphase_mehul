# Logistic Regression: A Comprehensive Overview

Logistic regression is a fundamental algorithm used for classification tasks. Unlike linear regression, which predicts continuous outcomes, logistic regression is used to predict discrete outcomes by estimating probabilities. This report explores the theory, mathematics, and applications of logistic regression.

## Table of Contents

1. [Introduction](#introduction)
2. [Logistic Regression Model](#logistic-regression-model)
   - [The Sigmoid Function](#the-sigmoid-function)
   - [Model Equation](#model-equation)
3. [Cost Function](#cost-function)
4. [Gradient Descent Optimization](#gradient-descent-optimization)
5. [Assumptions](#assumptions)
6. [Applications](#applications)

---

## Introduction

Logistic regression is a linear model used for binary classification problems. It predicts the probability that an instance belongs to a particular class and applies a decision threshold (usually 0.5) to classify the instance.

---

## Logistic Regression Model

### The Sigmoid Function

The sigmoid (or logistic) function maps real numbers to probabilities in the range (0, 1). It is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- $z$ is the linear combination of input features and their weights.
- $\sigma(z)$ outputs the predicted probability.

### Model Equation

The logistic regression model predicts the probability of the positive class ($y = 1$):

$$
P(y=1|\mathbf{x}) = \hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

Where:
- $\mathbf{x}$: Feature vector.
- $\mathbf{w}$: Weight vector.
- $b$: Bias term.
- $\hat{y}$: Predicted probability.

The decision boundary is defined as:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

---

## Cost Function

The cost function for logistic regression is based on the likelihood of the data. The objective is to maximize the likelihood of the observed data under the model. The negative log-likelihood is minimized instead, leading to the logistic regression cost function:

$$
J(\mathbf{w}, b) = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Where:
- $n$: Number of training examples.
- $y_i$: Actual label for the $i$-th example.
- $\hat{y}_i$: Predicted probability for the $i$-th example.

---

## Gradient Descent Optimization

Gradient descent is used to minimize the cost function and find optimal parameters $\mathbf{w}$ and $b$.

### Gradient of the Cost Function

The gradients of the cost function with respect to the parameters are:

1. Gradient with respect to weights:
   $$
   \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i) \mathbf{x}_i
   $$

2. Gradient with respect to bias:
   $$
   \frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)
   $$

### Update Rules

1. Update weights:
   $$
   \mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}
   $$

2. Update bias:
   $$
   b \leftarrow b - \alpha \frac{\partial J}{\partial b}
   $$

Where:
- $\alpha$: Learning rate.

---

## Assumptions

1. **Linearity of Features:** Logistic regression assumes a linear relationship between input features and the log-odds of the target.
2. **Independence of Observations:** Observations are assumed to be independent of each other.
3. **No Multicollinearity:** Independent variables should not be highly correlated.
4. **Sufficient Sample Size:** Logistic regression performs best with a large sample size.

---

## Applications

1. **Healthcare:** Disease diagnosis, patient risk prediction.
2. **Finance:** Credit scoring, fraud detection.
3. **Marketing:** Customer segmentation, churn prediction.
4. **Natural Language Processing:** Sentiment analysis, spam detection.

---

Logistic regression is a foundational tool in machine learning, offering simplicity, interpretability, and effectiveness for binary classification tasks. Understanding its mathematical foundations is essential for its proper application and extension to more complex problems like multi-class classification or regularized logistic regression.

