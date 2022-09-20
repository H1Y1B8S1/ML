import numpy as np 
import pandas as pd

dataset = pd.read_csv("C:\SINH\☆Work☆\ML\CSV_files\Titanicsurvival.csv")
# print(dataset.count())
# print(dataset.head(5))

dataset['Age'].fillna(dataset['Age'].mode()[0],inplace=True)
print(dataset.count())

#mapping Sex to 0 or 1
origenalD = set(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1}).astype(int)
print(dataset.head())

#segregate data into X(input/Independentvariable) & Y(output/dependentvariable)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#splitting dataset into train & test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=20)

#Training
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)

# Predicting wheather Person survived or not 

# pclassNo = int(input("ENter person's Pclass number:"))
# gender = int(input("ENter person's Gender 0-female 1-male:"))
# age = int(input("ENter person's Age:"))
# fare = float(input("ENter person's Fare:"))
# person = [[pclassNo,gender,age,fare]]
# result = model.predict(person)
# print(result)


# if result == 1:
#     print('person might be survived')
# else:
#     print('person might not survived')


Y_pred = model.predict(X_test)
# print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix:")
# print(cm)
print("Accuricy of the Model:{0}%".format(accuracy_score(Y_test,Y_pred)*100))