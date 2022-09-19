import pandas as pd 
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('C:\SINH\☆Work☆\ML\CSV_files\Exam_marks.csv')

print(dataset.shape)
print(dataset.head(5))

dataset["hours"].fillna(dataset['hours'].median(),inplace=True)
# print(dataset.to_string())

#segregate Dataset info Input X & Output Y

x = dataset.iloc[:,:-1].values
print(x.shape)

y = dataset.iloc[:,-1].values

# Training Dataset using Liner Refression
model = LinearRegression()
model.fit(x,y)


a = [[9.2,20,0]]
PredictemodelResult = model.predict(a)
print("Predicted Result:",PredictemodelResult)