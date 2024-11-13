**Titanic Logistic Regression**

**Introduction**

In this project, I developed a Logistic Regression model from scratch to classify passenger survival based on the Titanic dataset. This task involved data preprocessing, model implementation, training, and evaluation, providing insights into both the model's performance and the challenges encountered during the process.

**Dataset Overview**

The Titanic dataset contains information about passengers, including attributes such as age, gender, class, fare, and whether they survived the voyage. The dataset requires careful preprocessing to handle missing values and categorical variables before training the model.

**Challenges Faced**

1. **Missing Values**: The dataset had missing values, particularly in the 'Age' and 'Embarked' columns. These missing values needed to be addressed to avoid bias in the model's predictions.
   1. **Solution**: I filled missing values in the 'Age' column with the median age and in the 'Embarked' column with the most common embarkation point (mode). This approach helped retain as much data as possible without introducing significant noise.
1. **Irrelevant Features**: Columns such as 'PassengerId', 'Name', 'Ticket', and 'Cabin' do not provide useful information for survival prediction.
   1. **Solution**: I dropped these irrelevant columns to simplify the dataset and focus on features that contribute to the prediction.
1. **Categorical Variables**: The dataset includes categorical features like 'Sex' and 'Embarked', which need to be converted into numerical values for model compatibility.
   1. **Solution**: I encoded the 'Sex' column using binary mapping (0 for male and 1 for female) and applied one-hot encoding to the 'Embarked' column, ensuring the model could interpret these categorical features effectively.
1. **Feature Normalization**: Continuous features like 'Age' and 'Fare' can have varying scales, which may affect the model's performance.
   1. **Solution**: I normalized these continuous features by applying standard scaling (subtracting the mean and dividing by the standard deviation) to ensure they are on a similar scale.

**Model Implementation**

I implemented the Logistic Regression model using a class structure that encompassed initialization, training, and prediction.

**Key Steps:**

- **Initialization**: Set the learning rate and number of iterations, and initialized weights and bias.
- **Training**: Implemented the gradient descent algorithm to optimize weights and bias using the sigmoid function for predictions.
- **Prediction**: Classified predictions based on a threshold of 0.5.

**Model Training and Evaluation**

I split the dataset into training and testing sets (80-20 split) and trained the model using the training set. The evaluation was done using accuracy and a confusion matrix to assess the model's performance.

**Results**

- **Accuracy**: The model achieved an accuracy of approximately **: 0.8324** 
- **Confusion Matrix**: I visualized the confusion matrix to show true vs. predicted classifications, providing insights into the model's strengths and weaknesses.

**Visualization**

A heatmap of the confusion matrix was created to visualize the performance, showing the distribution of true positives, true negatives, false positives, and false negatives.
**Conclusion**

The project enhanced my understanding of implementing a Logistic Regression model from scratch, emphasizing the importance of data preprocessing and feature engineering. I successfully addressed several challenges, enabling me to develop a model that classifies passenger survival with reasonable accuracy. Future work could explore hyperparameter tuning and advanced feature selection techniques to improve the model's performance further.
