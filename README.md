# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Display the values predicted using scatter plot and predict.

3.Plot the graph according to the given input.

4.End the program 

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sachin.C
RegisterNumber:  212222230125
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:

## df.head()
![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/669e3180-8b33-4b78-a3da-67bc8623b721)

## df.tail()
![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/af86f6a5-9ba9-4b4f-9472-e88b2fec9199)

## Array values of X
![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/a5fe67dc-b9ac-4a34-ba67-3ab3d6495cad)

## Array values of Y
![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/6f65b548-8c84-4181-8ccf-8ca6206614ec)

## Values of Y Prediction
![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/86585e0e-910b-4a26-969e-4c2d6d5554ea)

## Array values of Y Test
![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/7abaa21a-b848-4336-a64d-a255a17d9105)

## TRAINING SET GRAPH
![Screenshot 2023-08-24 091301](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/5541802c-a5a1-46c7-850d-38d7c376d76c)

## TEST SET GRAPH
![Screenshot 2023-08-24 091316](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/f3e722b9-d843-4ace-9b93-30127f94770e)

![image](https://github.com/Sachin-vlr/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497666/69bb99d6-f574-43a4-9ff4-600b7e2fcb50)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
