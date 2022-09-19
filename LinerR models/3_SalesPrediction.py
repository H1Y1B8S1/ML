import numpy as np 
import pandas as pd 

dataset = pd.read_csv('C:\SINH\☆Work☆\ML\CSV_files\sales_data.csv')

print(dataset.shape)
print(dataset.head(5))

# saperating variables - dependent and independent
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
# print(X.shape)
# print(Y.shape)
# print(X)


# spliting data for Training and Testing..
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)


### Feature Scalings
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train,Y_train)


#Testing

age = int(input('Enter new customor age:'))
sal = int(input('Enter new customor salary:'))
newCust = [[age,sal]]
result = model.predict(sc.transform(newCust))
print(result)
if result == 1:
    print("Customr will buy")
else:
    print("customre won't buy")


# Prediction(Testing) for all test data
y_pred = model.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))


# Confusion Matrix [[TP,FN],[FP,TN]] -->True,positve,False,Negative
# C = (TP + TN) / (TP +TN + FP + FN)
#Evaluating Model - CONFUSION MATRIX

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test, y_pred)   
print('confusion matrix')
print(cm)

print(f"Accuracy of the model: {accuracy_score(Y_test, y_pred)*100}%")