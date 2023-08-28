# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sneha Basyal M
RegisterNumber: 212222240101
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

# segregating data to variables
x = df.iloc[:,:-1].values
x
Y = df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split 
x_train,x_test,Y_train,Y_test=train_test_split(x,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,Y_train)
Y_pred = regressor.predict(x_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(x_train,Y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,Y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## df.head()
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML1.png)

## df.tail()
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML2.png)

## Array values of X
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML3.png)

## Array values of Y
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML4.png)

## Array values of Y prediction
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML5.png)

## Array values of Y test
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML6.png)

## Training set graph
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML7.png)

## Test set graph
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML8.png)

## Values for MSE,MAE,RMSE
![Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored-1](ML9.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
