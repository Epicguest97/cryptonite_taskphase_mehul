# Support Vector Machines (SVM): A Detailed Overview

Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression tasks. They are particularly effective in high-dimensional spaces and for problems with clear margins of separation. This report delves into the theory, mathematics, and applications of SVMs.

## Table of Contents

1. [Introduction](#introduction)
2. [SVM for Binary Classification](#svm-for-binary-classification)
   - [Hyperplane and Decision Boundary](#hyperplane-and-decision-boundary)
   - [Mathematics of SVM](#mathematics-of-svm)
3. [Kernel Trick](#kernel-trick)
4. [Soft Margin and Regularization](#soft-margin-and-regularization)
5. [Mathematics of SVM Optimization](#mathematics-of-svm-optimization)
6. [Advantages and Limitations](#advantages-and-limitations)
7. [Applications](#applications)

---

## Introduction

Support Vector Machines aim to find the optimal hyperplane that separates data into different classes. For non-linearly separable data, SVM employs kernel functions to map the data into a higher-dimensional space where a linear separator can be found.

---

## SVM for Binary Classification

### Hyperplane and Decision Boundary

The hyperplane is the decision boundary that maximizes the margin between two classes. In a binary classification problem, the goal of SVM is to find the hyperplane defined as:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

Where:
- $\mathbf{w}$: Weight vector.
- $\mathbf{x}$: Feature vector.
- $b$: Bias term.

The classes are separated as:
- $\mathbf{w}^T \mathbf{x} + b > 0$ for class $+1$
- $\mathbf{w}^T \mathbf{x} + b < 0$ for class $-1$

### Mathematics of SVM

The margin is the distance between the hyperplane and the nearest data points from each class (support vectors). The optimization problem can be formulated as:

#### Objective Function:

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
$$

#### Subject to:

$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

Where:
- $\|\mathbf{w}\|$ is the norm of the weight vector (controls margin width).
- $y_i \in \{-1, +1\}$ are the class labels.
- $\mathbf{x}_i$ are the feature vectors.

---

## Kernel Trick

For non-linearly separable data, the kernel trick maps the input features into a higher-dimensional space, allowing a linear hyperplane to separate the data. The kernel function computes the dot product in the transformed feature space without explicitly computing the transformation.

Common kernel functions include:

1. **Linear Kernel:**
   $$
   K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j
   $$

2. **Polynomial Kernel:**
   $$
   K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d
   $$

3. **Gaussian (RBF) Kernel:**
   $$
   K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)
   $$

4. **Sigmoid Kernel:**
   $$
   K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\alpha \mathbf{x}_i^T \mathbf{x}_j + c)
   $$

---

## Soft Margin and Regularization

Real-world data is often noisy and non-linearly separable. To handle such cases, SVM introduces a soft margin, allowing some misclassifications. The optimization problem becomes:

#### Objective Function:

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i
$$

#### Subject to:

$$
\begin{aligned}
& y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \\
& \xi_i \geq 0 \quad \forall i
\end{aligned}
$$

Where:
- $\xi_i$: Slack variable representing the degree of misclassification.
- $C$: Regularization parameter that controls the trade-off between margin width and classification error.

---

## Mathematics of SVM Optimization

To solve the optimization problem, SVM uses the Lagrange multipliers. The dual form of the optimization problem is:

#### Dual Objective Function:

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

#### Subject to:

$$
\begin{aligned}
& \sum_{i=1}^n \alpha_i y_i = 0 \\
& 0 \leq \alpha_i \leq C \quad \forall i
\end{aligned}
$$

Where:
- $\alpha_i$: Lagrange multipliers.
- $K(\mathbf{x}_i, \mathbf{x}_j)$: Kernel function.

The decision function is:

$$
\hat{y} = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

---

## Advantages and Limitations

### Advantages

1. **Effective in High Dimensions:** SVM performs well in spaces with many features.
2. **Kernel Flexibility:** Can model non-linear relationships using kernels.
3. **Robust to Overfitting:** Particularly for high-dimensional datasets.

### Limitations

1. **Computational Cost:** Training can be slow for large datasets.
2. **Choice of Kernel:** Selecting the right kernel and parameters can be challenging.
3. **Sensitivity to Outliers:** SVM is sensitive to outliers, though soft margins mitigate this to some extent.

---

## Applications

1. **Bioinformatics:** Protein classification, gene expression analysis.
2. **Image Recognition:** Handwriting and face recognition.
3. **Finance:** Fraud detection, credit scoring.
4. **Text Classification:** Spam detection, sentiment analysis.

---

Support Vector Machines are versatile and effective tools for classification and regression. Their mathematical rigor and ability to handle complex data structures make them invaluable in a variety of domains. Understanding their foundations is crucial for leveraging their full potential.

