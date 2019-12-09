import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import datetime
import json
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from plot_keras_history import plot_history


# Pandas read CSV
sf_train = pd.read_csv('C:/Users/iqbal/Desktop/skidikipapap/data/p5_training_data.csv')

# Correlation Matrix for target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# Drop unnecessary columns
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# Pandas read CSV
sf_train = pd.read_csv('C:/Users/iqbal/Desktop/skidikipapap/data/p5_training_data.csv')

# Correlation Matrix for target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# Drop unnecessary columns
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# Pandas read Validation CSV
sf_val = pd.read_csv('C:/Users/iqbal/Desktop/skidikipapap/data/p5_val_data.csv')

# Drop unnecessary columns
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

# Get Pandas array value (Convert to NumPy array)
train_data = sf_train.values
val_data = sf_val.values

# Use columns 2 to last as Input
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Use columns 1 as Output/Target (One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )

# Create Network
inputs = Input(shape=(16,))
h_layer = Dense(10, activation='sigmoid')(inputs)

# Softmax Activation for Multiclass Classification
outputs = Dense(3, activation='softmax')(h_layer)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)

# Compile the model with Cross Entropy Loss
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and use validation data
history = model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=1, validation_data=(val_x, val_y)).history
pd.DataFrame(history).to_csv("historydota2.csv")

# lets assume `model` is main model 
#model_json = model.to_json()
#with open("model_in_json.json", "w") as json_file:
#    json.dump(model_json, json_file)

#model.save_weights("model_weights.h5")
model.save_weights('weight.h5')


# Predict all Validation data
predict = model.predict(val_x)

plot_model(model,show_shapes=True, expand_nested=True,to_file='model.png')

# Visualize Prediction
df = pd.DataFrame(predict)
df.columns = ['Strength', 'Agility', 'Intelligent' ]
df.index = val_data[:,0]
print(df)

plot_history(history, style='-', path='singleton', single_graphs=True, side=12, graphs_per_row=1)
#plot_history(history)
plt.show()