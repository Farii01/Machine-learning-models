import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read in the dataset
df = pd.read_csv("govt.csv")

df.dropna(subset=['Employment Status'], inplace=True) #The inplace parameter specifies whether to modify the
#original DataFrame or return a new DataFrame with the missing values dropped.

X = df.drop(['Employment Status'], axis=1)
y = df['Employment Status']

#transform
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# fit the model and make predictions
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("---------------------------------------------------")
print("Accuracy of KNN model:", accuracy)
print("---------------------------------------------------")