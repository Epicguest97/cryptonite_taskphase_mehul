# k-Nearest Neighbors (k-NN) Algorithm

The k-Nearest Neighbors (k-NN) algorithm is one of the simplest machine learning algorithms and is widely used for both classification and regression tasks. It is a non-parametric algorithm, meaning it makes no assumptions about the underlying data distribution.

---

## How k-NN Works

The k-NN algorithm operates based on the concept of proximity. It classifies a data point by looking at the **k** closest data points in the feature space and assigning the class most common among them (for classification) or taking the average of their values (for regression).

1. **Define k**: Choose the number of nearest neighbors (**k**).
2. **Measure distance**: Calculate the distance between the query point and all other points in the dataset.
3. **Find neighbors**: Identify the **k** nearest neighbors to the query point.
4. **Make a decision**: Assign the class or predict the value based on the neighbors.

---

## Distance Metrics

The performance of k-NN heavily depends on the choice of the distance metric. Commonly used metrics include:

### 1. Euclidean Distance
The Euclidean distance is the straight-line distance between two points in a multidimensional space:

$$
d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

### 2. Manhattan Distance
The Manhattan distance is the sum of absolute differences between the coordinates of two points:

$$
d(x, y) = \sum_{i=1}^n |x_i - y_i|
$$

### 3. Minkowski Distance
The Minkowski distance generalizes both the Euclidean and Manhattan distances:

$$
d(x, y) = \left( \sum_{i=1}^n |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

When **p = 2**, it is equivalent to the Euclidean distance, and when **p = 1**, it is equivalent to the Manhattan distance.

---

## Algorithm Steps

### 1. Initialization
- Choose the number of neighbors (**k**).
- Select a distance metric (e.g., Euclidean).

### 2. Compute Distances
For a given query point \( q \), compute the distance between \( q \) and all points in the training dataset.

### 3. Sort and Select Neighbors
- Sort the training points based on the computed distances.
- Select the **k** points with the smallest distances.

### 4. Predict
- For classification, assign the class label that appears most frequently among the **k** neighbors.
- For regression, compute the average of the values of the **k** neighbors.

---

## Choosing the Value of k

The choice of **k** significantly impacts the performance of the k-NN algorithm:
- A small **k** makes the model sensitive to noise (overfitting).
- A large **k** smoothens the decision boundary but may ignore local patterns (underfitting).

A good practice is to use cross-validation to select the optimal value of **k**.

---

## Advantages of k-NN

- **Simple to understand and implement**: No complex model training.
- **Non-parametric**: No assumptions about data distribution.
- **Versatile**: Can be used for both classification and regression.

---

## Disadvantages of k-NN

- **Computationally expensive**: Requires computing distances for all points in the dataset.
- **Sensitive to feature scaling**: Distance measures are affected by the scale of the features, so feature normalization is often required.
- **Choice of k and distance metric**: These hyperparameters can significantly influence performance.

---

## Example

### Classification Example
Given a dataset with two classes, the task is to classify a new data point based on the labels of its nearest neighbors.

### Regression Example
Predict the house price for a query point based on the average price of the **k** nearest houses.

---

## Conclusion

The k-NN algorithm is a powerful tool for supervised learning tasks when simplicity and interpretability are required. However, its effectiveness relies on careful selection of hyperparameters and preprocessing of data.

