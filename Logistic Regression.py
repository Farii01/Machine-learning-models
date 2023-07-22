import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("govt.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Split the dataset into features and target
X = df.drop("Employment Status", axis=1)#axis 1 is the target attribute, oita baad diye baki gula mane feature is selected
y = df["Employment Status"] #target

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Create a logistic regression model
lr_model = LogisticRegression()

# Train the model on the training data
lr_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lr_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("---------------------------------------------------------")
print("Accuracy of Logistic Regression model:", accuracy)
print("---------------------------------------------------------")