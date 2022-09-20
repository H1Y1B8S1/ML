import pandas as pd
import numpy as np

dataset = pd.read_csv('C:\SINH\☆Work☆\ML\CSV_files\salary.csv')

# print(dataset.shape)
# print(dataset.head(5))

#mapping salary data to binary value

income_set =set(dataset['income'])
dataset['income']=dataset['income'].map({'<=50K': 0,'>50K':1}).astype(int)
print(dataset.head)

#segregate data into X(input/Independentvariable) & Y(output/dependentvariable)
X = dataset.iloc[:,:-1].values
#X
Y = dataset.iloc[:,-1].values
#Y

#splitting dataset into train & test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test,=train_test_split(X,Y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#finding the best K-value

error=[]
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#calculating error for K value b/w 1 and 40
for i in range(1,40):
    model =KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    pred_i =model.predict(X_test)
    error.append(np.mean(pred_i !=Y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='*',markerfacecolor='blue',markersize=10)
plt.title("Error Rate K value")
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#Training
from sklearn.neighbors import KNeighborsClassifier
model =KNeighborsClassifier(n_neighbors=16,metric='minkowski',p=2)
model.fit(X_train,Y_train)

#predicting wheather new customer with Age & Salary will Buy or Not
age=int(input("Enter New Employee's Age: "))
edu=int(input("Enter New Employee's Education: "))
cg=int(input("Enter New Employee's Captital Grain: "))
wh=int(input("Enter New Employee's Hours per week: "))
newEMp=[[age,edu,cg,wh]]
result = model.predict(sc.transform(newEMp))
print(result)

if result==1:
    print("Employee might get Salary above 50K")
else:
    print("Employee might get Salary below 50K")

Y_pred = model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuricy of the Model:{0}%".format(accuracy_score(Y_test,Y_pred)*100))