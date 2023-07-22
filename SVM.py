import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read in the dataset
df = pd.read_csv("govt.csv")

# drop rows with missing values in the target variable
df.dropna(subset=['Employment Status'], inplace=True)


# split data into training and testing sets
X = df.drop(['Employment Status'], axis=1) #feature
y = df['Employment Status'] #target

X = pd.get_dummies(X)#convert to categorical
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16) #The test_size parameter specifies
#the proportion of the data to use for testing, while random_state sets the random seed for reproducibility.

# fit the model and make predictions
model = SVC()
model.fit(X_train, y_train) #The fit() method is then called on the
#classifier object to train it on the training data (X_train and y_train).
y_pred = model.predict(X_test) #prediction on x test set

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred) #accuracy by comparing prediction and test set
print("-----------------------------------------")
print("Accuracy of SVM model:", accuracy)
print("-----------------------------------------")
