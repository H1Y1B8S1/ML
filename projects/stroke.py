import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('C:\SINH\☆Work☆\Projects\ml\stroke.csv')
# print(df)
# print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

import seaborn as sns
sns.set()
print(df['gender'].value_counts())

#Making a count plot for gender column
sns.countplot(x ='gender', data = df)

#Making a count plot for ever_married column
print(df['ever_married'].value_counts())
sns.countplot(x='ever_married', data = df)
plt.show()


#Making a count plot for work_type column
print(df['work_type'].value_counts())
sns.countplot(x='work_type', data = df)
# plt.show()

#Making a count plot for Residence_type column
df['Residence_type'].value_counts()
sns.countplot(x='Residence_type', data = df)
# plt.show()

#Making a count plot for smoking_status column
print(df['smoking_status'].value_counts())
sns.countplot(x='smoking_status', data = df)
# plt.show()

print(df['hypertension'].value_counts())
#0 represents No Hypertension
#1 represents Hypertension
#Making a count plot for hypertension column
sns.countplot(x='hypertension', data = df)
# plt.show()

print(df['heart_disease'].value_counts())
#0 represents No Heart Disease
#1 represents Heart Disease
#Making a count plot for heart_disease column
sns.countplot(x='heart_disease', data = df)
# plt.show()

#Making a count plot for stroke column
print(df['stroke'].value_counts())
#0 represents No Stroke
#1 represents Stroke
sns.countplot(x='stroke', data = df)
# plt.show()

#Showing stroke and no stroke genderwise
sns.countplot(x='gender', hue ='stroke', data = df)
# plt.show()

#Showing hypertension and no hypertension genderwise
sns.countplot(x='gender', hue ='hypertension', data = df)
plt.show()

#Showing heart disease and no heart disease genderwise
sns.countplot(x='gender', hue ='heart_disease', data = df)
plt.show()

df = df.drop(columns = ['ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis = 1)
df.replace({'gender' : {'Male' : 0 , 'Female' : 1 , 'Other' : 2}}, inplace = True)

#Seperating the data and labels
X = df.drop(columns = ['gender', 'hypertension' , 'heart_disease', 'stroke'], axis = 1)
Y_hypertension = df['hypertension']
Y_heartdisease = df['heart_disease']
Y_stroke = df['stroke']
print(X)

#Data standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
standard = scaler.transform(X)
X = standard
print(X)
print(Y_hypertension)
print(Y_heartdisease)
print(Y_stroke)

## SPLIT DATA IN TEST AND TRAIN FOR HYPERTENSION PREDICTION

#Train,Test,Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_hypertension, test_size = 0.2, stratify = Y_hypertension, random_state = 2)
from sklearn import svm
model = svm.SVC(kernel = 'linear')

#Training the SVM Model
model.fit(X_train, Y_train)

#Finding the accuracy score on train dataset
from sklearn.metrics import accuracy_score
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(train_data_accuracy)

#Finding the accuracy score on test dataset
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(test_data_accuracy)

# MODEL EVALUATION FOR HYPERTENSION PREDICTION

data = (44679, 44.0, 85.28, 26.2)
#Converting to numpy array
data = np.array(data).reshape((len(data),1))
data = scaler.transform(data)

prediction = model.predict(data)

if(prediction[0] == 1):
    print('Hypertension')
else:
    print(' No Hypertension')

# SPLIT DATA IN TEST AND TRAIN FOR HEART DISEASE PREDICTION
#Train,Test,Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_heartdisease, test_size = 0.2, stratify = Y_heartdisease, random_state = 2)
model = svm.SVC(kernel = 'linear')

#Training the SVM Model
model.fit(X_train, Y_train)

#Finding the accuracy score on train dataset
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(train_data_accuracy)


#Finding the accuracy score on test dataset
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(test_data_accuracy)

# MODEL EVALUATION FOR HEART DISEASE PREDICTION
data = (44679, 44.0, 85.28, 26.2)
#Converting to numpy array
data = np.array(data).reshape((len(data),1))
data = scaler.transform(data)

prediction = model.predict(data)

if(prediction[0] == 1):
    print('Heart Disease')
else:
    print(' No Heart Disease')


# SPLIT DATA IN TEST AND TRAIN FOR STROKE PREDICTION
#Train,Test,Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_stroke, test_size = 0.2, stratify = Y_stroke, random_state = 2)
model = svm.SVC(kernel = 'linear')
model.fit(X_train, Y_train)

#Finding the accuracy score on train dataset
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(train_data_accuracy)

#Finding the accuracy score on test dataset
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(test_data_accuracy)

# MODEL EVALUATION FOR STROKE PREDICTION
data = (44679, 44.0, 85.28, 26.2)
#Converting to numpy array
data = np.array(data).reshape((len(data),1))
data = scaler.transform(data)

prediction = model.predict(data)

if(prediction[0] == 1):
    print('Stroke')
else:
    print(' No Stroke')