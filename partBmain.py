import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split    
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#1. Select the training data.csv and convert it to a pandas data frame, the header parameter can be set to None as the given data.csv does not contain a header
tariningdata = pd.read_csv('TrainingDataMulti.csv')
testingdata = pd.read_csv('TestingDataMulti.csv')

tariningdata.fillna(0, inplace=True)

# 2. divide the dataset into training set and testing set
X = tariningdata.iloc[:, :128]
y = tariningdata.iloc[:, 128]

# 3.Use the train_test_split module in scikit-learn to split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# 4.In order to eliminate the influence caused by the large data difference, StandardScaler is used to standardize and highlight the characteristics of the data set.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. n_estimators indicates the number of trees in the random forest, random_state indicates random mode parameters.
# In scikit-learn, the random forest module RandomForestClassifier is used to train the training data.
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 6. After the model is trained, use the test data set to make predictions. The classification of the test set is essentially to predict the dependent variable y through the test set data, that is, the class in the table.
y_pred = classifier.predict(X_test)
print("X_test:", X_test)
print("y_pred:", y_pred)

# 7.Finally, the classifier that comes with scikit-learn is used to output the prediction results, including confusion matrix, Classification Report, and Accuracy.
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:", )
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)


test_data = pd.read_csv('TestingDataMulti.csv', header=None)
X_test = test_data.iloc[:, :128]
#9. Using the RandomForestClassifier test model with specific hyperparameters to adjust and fit the training data to predict the unknown labels of the test dataX_test = testingdata.iloc[:, :128]

model=RandomForestClassifier()

# 10.train the model
model.fit(X,y)
# 11. Prediction with test set
y_pred = model.predict(X_test)
test_data[128] = y_pred
test_data.to_csv('TestingResultsMulti.csv', index=False)


















