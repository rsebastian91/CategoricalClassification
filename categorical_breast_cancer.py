# -*- coding: utf-8 -*-
"""

@author: robin
"""

import ML_module as ML
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.close('all')

# Load data
data = pd.read_csv("breast-cancer.csv")
dataset = data.values

print('Shape of the dataset: ' + str(dataset.shape))

X= dataset[:,:-1]
Y= dataset[:,-1]

X=X.astype(str)
Y=Y.reshape((len(Y),))

# Split data
x_train, x_valid, y_train, y_valid = train_test_split(X,Y, test_size=0.33, random_state=1)


MLobj=ML.Categorical(x_train,y_train,x_valid,y_valid)

# Explore data
MLobj.check_data()


#Encoding
MLobj.data_encoding(encoding='ordinal')

num_categories =2
MLobj.target_encoding(num_categories,encoding='label')


#################################################
#Creating model
from tensorflow.keras.models import Sequential
x_train=MLobj.x_train
y_train=MLobj.y_train

x_valid=MLobj.x_valid
y_valid=MLobj.y_valid


print("Creating model")
model = Sequential()

#Inpul layer
from tensorflow.keras.layers import Dense

model.add(Dense(units=10, activation='relu', input_shape=(x_train.shape[1],)))


#Output layer
model.add(Dense(units = 1, activation='sigmoid'))

#Model summary
model.summary()

#Model compiling
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training model
nb_epochs=100
history = model.fit(
    x_train, y_train, epochs=nb_epochs, batch_size=16, verbose=2, validation_data=(x_valid, y_valid)
)


acc = [element * 100 for element in history.history['accuracy']]
val_acc = [element * 100 for element in history.history['val_accuracy']]
loss = history.history['loss']
val_loss = history.history['val_loss']

##################################################
#
#
##plot accuracy and loss
MLobj.plot_acc_and_loss(acc,val_acc,loss,val_loss)

