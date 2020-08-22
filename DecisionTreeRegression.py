# Decision Tree Regression

""" The goal is to predict the salary of an employee given the experience, corresponding to particular level by
deploying Decision Tree Regression"""


##importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

##fitting the desision tree model to the data
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

##predicting our desired value salary
y_pred = regressor.predict(pd.DataFrame([6.5]))

##ploting the model
plt.scatter(x,y, color = 'red')
plt.plot(x,regressor.predict(x), color = 'blue')
plt.title('level vs salary with decision tree regression model')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()

## the above plot shows overfitting because, since decision tree model is not continous, it take the interval and
##it just joins the points. So to avoid this take high resolution look.

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('level vs salary using Decision Tree Regression model')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()











