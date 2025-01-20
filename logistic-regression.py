import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/in22417145/PycharmProjects/Logistic Regression - Breast Cancer/data.csv")
print(data.head())
print(data.info())
print(data.describe())

sns.heatmap(data.isnull())
plt.show()

#drop null column "Unnamed: 32" and irrelevant column "id"
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

#Set values to 0 if value of column diagnosis is M, else set it to 0.
print(data["diagnosis"].unique())
data["diagnosis"] = [1 if i == "M" else 0 for i in data["diagnosis"]]

#to change the datattype of diagnosis from integer to a categorical. copy=False ensures it modifies the original dataframe.
data["diagnosis"] = data["diagnosis"].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")
plt.show()

#Divide the data into target(output) and predictors(input)
#Here we want to predict the diagnosis(target) value based on the different inputs(predictors)

#target variable
y = data["diagnosis"]

#predictor variable
X = data.drop(["diagnosis"], axis = 1)

# Normalize the input predictors to have the uniform units
from sklearn.preprocessing import StandardScaler

# 1. create a scaler object
scaler = StandardScaler()

# 2. fit the scaler to the data and transform it
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# 3. Split the data into training and testing datasets
# It returns 4 data and are assigned as below. 
# The parameters are predictor, target, test size and random state (to have the same data used when repeating the process)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,random_state=12)

# 4. Train the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 5. Predict the target on test data
y_predict = lr.predict(X_test)
print(f"Predicted data {y_predict}")

# 6. Evaluate the model
# parameters are actual value and predicted values
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy of the model {accuracy: .2f}")

# 7. More Statistical metrics
'''
Precision: The ratio of correctly predicted positive observations to the total predicted positives. It answers the question, 
"Of all the positive predictions, how many were actually correct?"
Recall: The ratio of correctly predicted positive observations to all the actual positives. It answers the question, 
"Of all the actual positives, how many were correctly predicted?"
The F1-score ranges from 0 to 1, where 1 indicates perfect precision and recall, and 0 indicates the worst performance.
'''
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


