# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:55:35 2020

@author: brand
"""

# import packages
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# import data
(train_pics, train_answers), (test_pics, test_answers) = mnist.load_data()

# transform training data
train_pics = train_pics.reshape((60000,28,28,1))
train_pics = train_pics.astype('float32') / 255
train_answers = to_categorical(train_answers)

# transform testing data
test_pics = test_pics.reshape((10000,28,28,1))
test_pics = test_pics.astype('float32') / 255
test_answers = to_categorical(test_answers)

# build neural network layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.summary()

# build neural network output layers
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_pics, train_answers, epochs=5, batch_size=128)

# baseline test accuracy measure
test_loss, test_accuracy = model.evaluate(test_pics, test_answers)
test_accuracy

# walk through an example
which_num_img = 10
plt.imshow(-np.array(train_pics[which_num_img], dtype='float').reshape((28, 28)), cmap='gray')
example_num = np.expand_dims(train_pics[which_num_img], axis=0)

# extract output from the top 8 layers
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# generate activations
activations = activation_model.predict(example_num)

# retreive activations
first_layer_activations = activations[0]
print(first_layer_activations.shape)

# plot the first 4 channels of the activation of the first layer
for i in range(1,5):
    plt.subplot(2, 2, i)
    plt.axis('off')
    plt.imshow(first_layer_activations[0, :, :, i])

# plot the first 4 channels of the activation of the second layer
second_layer_activations = activations[1]
    
for i in range(1,5):
    plt.subplot(2, 2, i)
    plt.axis('off')
    plt.imshow(second_layer_activations[0, :, :, i]) 

# make a prediction
prediction = model.predict_classes(example_num)
print(prediction)

probability = model.predict_proba(example_num)
print(probability)

y_axis = np.arange(10).astype('str')
x_axis = probability[0][:] #np.vectorize(probability)
plt.barh(y_axis, x_axis)

