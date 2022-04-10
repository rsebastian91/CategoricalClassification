# -*- coding: utf-8 -*-
"""

@author: robin
"""

import ML_module as ML


# Load data
import pandas as pd

train_df = pd.read_csv('asl_data/sign_mnist_train.csv')
valid_df = pd.read_csv('asl_data/sign_mnist_test.csv')

# Split between train and validation sets
y_train = train_df['label'].values
y_valid = valid_df['label'].values
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values


MLobj=ML.Classification(x_train,y_train,x_valid,y_valid)

# Explore data
MLobj.check_data()


#Data preparation for training
MLobj.data_preparation(flaten=False,normalise=True)


#Target encoding
num_categories =25
MLobj.target_encoding(num_categories,encoding='binarymartix')


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

model.add(Dense(units=512, activation='relu', input_shape=(x_train.shape[1],)))

#Hidden layer
model.add(Dense(units = 512, activation='relu'))

#Output layer
model.add(Dense(units = num_categories, activation='softmax'))

#Model summary
model.summary()

#Model compiling
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

#Training model
nb_epochs=20
history = model.fit(
    x_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(x_valid, y_valid)
)


acc = [element * 100 for element in history.history['accuracy']]
val_acc = [element * 100 for element in history.history['val_accuracy']]
loss = history.history['loss']
val_loss = history.history['val_loss']

#################################################


#plot accuracy and loss
MLobj.plot_acc_and_loss(acc,val_acc,loss,val_loss)

