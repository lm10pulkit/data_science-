import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)"""

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)


## polynomial

from sklearn.preprocessing import PolynomialFeatures

poly_reg= PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

## linear regression

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X),color='blue')
plt.title('Truth')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()


## polynomial regression

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

