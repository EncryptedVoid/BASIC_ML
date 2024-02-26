from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IRIS_DATABASE_SETUP import df

# Extract the feature variables (X) and the target variable (y)
# df.iloc[:, :-1].values selects all rows and all columns except the last one as features
X = df.iloc[:, :-1].values
# df.iloc[:, -1].values selects all rows and only the last column as the target
y = df.iloc[:, -1].values

# Use train_test_split to randomly split the data into training and testing sets
# test_size=0.2 means 20% of the data is used for the test set, and 80% for the training set
# random_state=42 is used for reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
# max_iter=200 specifies the maximum number of iterations for the solver to converge
model = LogisticRegression(max_iter=200)

# Train the logistic regression model using the training data
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
# accuracy_score compares the actual target values in the test set with the predictions
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy to see how well our model performed
print(f'Accuracy: {accuracy}')