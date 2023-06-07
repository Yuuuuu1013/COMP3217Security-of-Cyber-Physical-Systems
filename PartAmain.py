import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier   
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler  

#1. Select the training data.csv and convert it to a pandas data frame, the header parameter can be set to None as the given data.csv does not contain a header
tariningdata = pd.read_csv('TrainingDataBinary.csv')
testingdata = pd.read_csv('TestingDataBinary.csv')

# 2. divide the dataset into training set and testing set
X = tariningdata.iloc[:, :128]
y = tariningdata.iloc[:, 128]

# 3.Use the train_test_split module in scikit-learn to split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    
# 4.In order to eliminate the influence caused by the large data difference, StandardScaler is used to standardize and highlight the characteristics of the data set.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Random Forest All Hyperparameters
#sklearn.ensemble.RandomForestClassifier (n_estimators=100, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                        # min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                         #min_impurity_split=None, class_weight=None, random_state=None, bootstrap=True, oob_score=False, 
                                         #n_jobs=None, verbose=0, warm_start=False)

def random_forestclassifacation():
    model=RandomForestClassifier(max_depth=20,n_estimators=50,min_samples_leaf=1,min_samples_split=2,)
    # Training model
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('RFC Prediction accuracy is: {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
    print('RFC Precision_score is：{:.4}%'.format(precision_score(y_test,y_pred)*100))
    print('RFC recall_score is ：{:.4}%'.format(recall_score(y_test,y_pred)*100))
    print('RFC f1_score is：',f1_score(y_test,y_pred))
    print('RFC cohen_kappa_score is：',cohen_kappa_score(y_test,y_pred))
    print('RFC classification_report is：','\n',classification_report(y_test,y_pred))

def svm_classifacation():
 model=SVC()
 model.fit(X_train,y_train)
 y_pred = model.predict(X_test)
 print('SVM Prediction accuracy is: {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
 print('SVM Precision_score is：{:.4}%'.format(precision_score(y_test,y_pred)*100))
 print('SVM recall_score is ：{:.4}%'.format(recall_score(y_test,y_pred)*100))
 print('SVM f1_score is：',f1_score(y_test,y_pred))
 print('SVM cohen_kappa_score is：',cohen_kappa_score(y_test,y_pred))
 print('SVM classification_report is：','\n',classification_report(y_test,y_pred))
      
def knn():
 knn = KNeighborsClassifier(n_neighbors=2,p=2,metric="minkowski")
 knn.fit(X_train,y_train)
 y_pred = knn.predict(X_test)
 from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
 print('KNN Prediction accuracy is: {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
 print('KNN Precision_score is：{:.4}%'.format(precision_score(y_test,y_pred)*100))
 print('KNN recall_score is ：{:.4}%'.format(recall_score(y_test,y_pred)*100))
 print('KNN f1_score is：',f1_score(y_test,y_pred))
 print('KNN cohen_kappa_score is：',cohen_kappa_score(y_test,y_pred))
 print('KNN classification_report is：','\n',classification_report(y_test,y_pred))

def decisiontree_classifacation():
    model=DecisionTreeClassifier()
    # Training model
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
    print('DCT Prediction accuracy is: {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
    print('DCT Precision_score is：{:.4}%'.format(precision_score(y_test,y_pred)*100))
    print('DCT recall_score is ：{:.4}%'.format(recall_score(y_test,y_pred)*100))
    print('DCT f1_score is：',f1_score(y_test,y_pred))
    print('DCT cohen_kappa_score is：',cohen_kappa_score(y_test,y_pred))
    print('DCT classification_report is：','\n',classification_report(y_test,y_pred))

def Logistic_Regression():
   model=LogisticRegression()
    # Training model
   model.fit(X_train,y_train)
   y_pred = model.predict(X_test)
   print('LGR Prediction accuracy is: {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
   print('LGR Precision_score is：{:.4}%'.format(precision_score(y_test,y_pred)*100))
   print('LGR recall_score is ：{:.4}%'.format(recall_score(y_test,y_pred)*100))
   print('LGR f1_score is：',f1_score(y_test,y_pred))
   print('LGR cohen_kappa_score is：',cohen_kappa_score(y_test,y_pred))
   print('LGR classification_report is：','\n',classification_report(y_test,y_pred))
   
   
   
   
random_forestclassifacation() #~98.25%

#svm_classifacation() #~90.58%
#knn() #~95.25%
#decisiontree_classifacation() #~95.5%
#Logistic_Regression() #~89.08%

# 5. Load Testing Data
X_test = testingdata.iloc[:, :128]
# Ensure the feature names match
X_test.columns = X.columns

#6. Using the RandomForestClassifier test model with specific hyperparameters to adjust and fit the training data to predict the unknown labels of the test dataX_test = testingdata.iloc[:, :128]
model=RandomForestClassifier(max_depth=30,n_estimators=50,min_samples_leaf=1,min_samples_split=2) #After find bestparameters can got 98.99%
# 7.train the model
model.fit(X,y)
# 8. Prediction with test set
y_pred = model.predict(X_test)

# 9. Add the predicted results to the test set DataFrame and save as a new csv file
X_test['Predicted'] = y_pred
X_test.to_csv('TestingResultsBinary.csv', index=False)
