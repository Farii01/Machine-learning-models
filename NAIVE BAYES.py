import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading the dataset
df = pd.read_csv("govt.csv")
#preprocessed
df.dropna(subset=['Employment Status'], inplace=True)

X = df.drop(['Employment Status'], axis=1)
y = df['Employment Status']

#Numerical value conversion
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# fit the model and make predictions
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("---------------------------------------------------")
print("Accuracy of Decision tree model:", accuracy)
print("---------------------------------------------------")
