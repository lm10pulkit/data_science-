#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset

dataset=pd.read_csv('Churn_Modelling.csv')

X= dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x_1=LabelEncoder()

X[:,1]=labelencoder_x_1.fit_transform(X[:,1])

labelencoder_x_2=LabelEncoder()

X[:,2]=labelencoder_x_2.fit_transform(X[:,2])


onehotencoder = OneHotEncoder(categorical_features=[1])

X= onehotencoder.fit_transform(X).toarray()

X =X[:,1:]


from sklearn.model_selection import train_test_split

X_train,X_test , Y_train , Y_test= train_test_split(X,Y,test_size=0.2)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train= sc.fit_transform(X_test)

X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing nueral network


classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, Y_test, batch_size = 10, nb_epoch = 100)
# Fitting our model
classifier.fit(X_train, Y_test, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

print(cm)