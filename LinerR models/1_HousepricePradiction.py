import tkinter as tk
from tkinter import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\SINH\☆Work☆\ML\CSV_files\House_data.csv')
# print(dataset.shape)
# print(dataset.head(5))

plt.xlabel("Area")
plt.ylabel('Price')
plt.scatter(dataset.area,dataset.price,color = 'red', marker = '*')
# plt.show()

x = dataset.drop('price',axis='columns')
y = dataset.price

model = LinearRegression()
model.fit(x,y)


windos = tk.Tk()
windos.title("Infidata price pradiction")
windos.geometry('750x250')
windos.configure(bg='yellow')
global text2
text2 = tk.Entry(windos)
text2.grid(row=4,column=0)

l1 = tk.Label(windos,text="Area",bg='#fff044',)
l1.grid(row=0,column=0)

def Print():
    i = int(text2.get())
    # i = int(input("enter the area size"))
    l = [[i]]
    PredictemodelResult = model.predict(l)
    # print(PredictemodelResult)
    l1 = tk.Label(windos,text=PredictemodelResult,font='bold,32',)
    l1.grid(row=3,column=2)

b1 = tk.Button(windos,text='Search',width=8,bg='#f0f044',font='bold,32',command=Print)
b1.grid(row=1,column=1)


windos.mainloop()