# Ensemble Models in Machine Learning

Ensemble models combine the predictions of multiple base models to improve the accuracy, robustness, and generalization of machine learning systems. This report discusses ensemble methods, their mathematics, and their advantages.

## Table of Contents

1. [Introduction](#introduction)
2. [Types of Ensemble Methods](#types-of-ensemble-methods)
   - [Bagging](#bagging)
   - [Boosting](#boosting)
   - [Stacking](#stacking)
3. [Mathematics of Ensemble Models](#mathematics-of-ensemble-models)
   - [Weighted Averaging](#weighted-averaging)
   - [Bagging Mathematics](#bagging-mathematics)
   - [Boosting Mathematics](#boosting-mathematics)
4. [Advantages and Applications](#advantages-and-applications)

---

## Introduction

Ensemble learning leverages the strengths of multiple models to achieve better predictive performance. By aggregating predictions from several base learners, ensemble methods reduce bias, variance, or both.

---

## Types of Ensemble Methods

### Bagging

Bagging (Bootstrap Aggregating) reduces variance by training multiple models independently and averaging their outputs.

1. **Algorithm:**
   - Train models on different bootstrap samples (random subsets with replacement).
   - Aggregate predictions by averaging (regression) or majority voting (classification).

2. **Popular Example:** Random Forest

---

### Boosting

Boosting reduces bias by training models sequentially, where each model corrects the errors of its predecessor.

1. **Algorithm:**
   - Assign weights to data points.
   - Train models iteratively, focusing on misclassified instances.
   - Aggregate predictions using weighted averages.

2. **Popular Examples:** AdaBoost, Gradient Boosting

---

### Stacking

Stacking combines predictions of multiple base learners using a meta-model.

1. **Algorithm:**
   - Train base learners on the dataset.
   - Train a meta-learner on predictions from base learners.
   - Final prediction comes from the meta-learner.

2. **Popular Example:** Stacked Generalization

---

## Mathematics of Ensemble Models

### Weighted Averaging

For an ensemble of \( M \) models, the final prediction is:

\[
\hat{y} = \sum_{i=1}^M w_i \hat{y}_i
\]

Where:
- \( \hat{y}_i \): Prediction of the \( i \)-th model.
- \( w_i \): Weight assigned to the \( i \)-th model, such that \( \sum_{i=1}^M w_i = 1 \).

---

### Bagging Mathematics

1. **Bootstrap Sampling:**
   Given a dataset \( D \) of size \( N \), generate \( B \) bootstrap samples \( D_1, D_2, \ldots, D_B \), each of size \( N \).

2. **Model Training:**
   Train \( B \) models independently on \( D_1, D_2, \ldots, D_B \).

3. **Aggregation:**
   - Regression: \( \hat{y} = \frac{1}{B} \sum_{i=1}^B \hat{y}_i \)
   - Classification: \( \hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_B) \)

---

### Boosting Mathematics

1. **Weighted Error:**
   At each iteration \( t \):
   \[
   \varepsilon_t = \frac{\sum_{i=1}^N w_i I(\hat{y}_i \neq y_i)}{\sum_{i=1}^N w_i}
   \]
   Where:
   - \( w_i \): Weight of the \( i \)-th instance.
   - \( I(\cdot) \): Indicator function (1 if true, 0 otherwise).

2. **Model Weight:**
   \[
   \alpha_t = \ln \left(\frac{1 - \varepsilon_t}{\varepsilon_t} \right)
   \]

3. **Weight Update:**
   \[
   w_i \leftarrow w_i \exp(\alpha_t I(\hat{y}_i \neq y_i))
   \]

4. **Final Prediction:**
   \[
   \hat{y} = \text{sign} \left( \sum_{t=1}^T \alpha_t \hat{y}_t \right)
   \]

---

## Advantages and Applications

1. **Advantages:**
   - Improves accuracy and robustness.
   - Reduces overfitting (e.g., Random Forest).
   - Handles complex datasets with boosting.

2. **Applications:**
   - Fraud Detection
   - Sentiment Analysis
   - Image Recognition
   - Financial Forecasting

---

Ensemble models have revolutionized machine learning by leveraging multiple learners to enhance performance. By understanding their principles and mathematics, practitioners can build robust, scalable models for real-world applications.

