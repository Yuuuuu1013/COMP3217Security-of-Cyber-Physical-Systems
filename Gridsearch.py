import pandas as pd 
from sklearn.ensemble import RandomForestClassifier     #Random Forest for Classification
from sklearn.model_selection import train_test_split          # divide training set and test set
from sklearn.metrics import accuracy_score   
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV #Use GridSearchCV to find the optimal parameters


# Select the training data.csv and convert it to a pandas data frame, the header parameter can be set to None as the given data.csv does not contain a header
tariningdata = pd.read_csv('TrainingDataBinary.csv')
testingdata = pd.read_csv('TestingDataBinary.csv')

#  divide the dataset into training set and testing set
X = tariningdata.iloc[:, :128]
y = tariningdata.iloc[:, 128]

# Use the train_test_split module in scikit-learn to split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    
# In order to eliminate the influence caused by the large data difference, StandardScaler is used to standardize and highlight the characteristics of the data set.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {'n_estimators': [10, 30, 50], 'max_features': [2, 4, 6, 8,10],'max_depth': [10, 30, 60],
    'min_samples_split': [2, 10, 20]} 
 

forest_reg = RandomForestClassifier(random_state=42)
# training 5 times
tree_reg = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
tree_reg.fit(X_train, y_train)
y_pr = tree_reg.predict(X_test)
a=pd.DataFrame()

a['pred']=list(y_pr)
a['yest']=list(y_test)
score=accuracy_score(y_pr,y_test)
y_pred_proba=tree_reg.predict_proba(X_test)
b=pd.DataFrame(y_pred_proba,columns=['label=0','label=1'])
print('Score：', score)
tree_reg.best_params_
