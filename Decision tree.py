import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("govt.csv")

# Split the data into features and target
X = data.drop("Employment Status", axis=1)
y = data["Employment Status"]

# Convert categorical data to numerical data
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

# Create a decision tree classifier/model
dtc = DecisionTreeClassifier(random_state=42) #random_state=42, we ensure that the random number
#generator is initialized with the same seed each time the code is run, which ensures that the results are reproducible. The choice of 42 as the seed value is arbitrary, as any integer value can be used.

# Train the classifier on the training data
dtc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dtc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("---------------------------------------------------")
print("Accuracy of Decision tree model:", accuracy)
print("---------------------------------------------------")