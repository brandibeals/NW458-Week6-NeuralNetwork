# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:55:35 2020
Author: Brandi Beals
Description: Developed for course MSDS 458 at Northwestern University
"""

######################################
# IMPORT PACKAGES AND DATA
######################################

# import packages
#from tensorflow.keras import layers
#from tensorflow.keras import models
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import data
(train_pics, train_answers), (test_pics, test_answers) = mnist.load_data()

######################################
# PREPARE DATA
######################################

# transform training data
train_pics = train_pics.reshape((60000,28,28,1))
train_pics = train_pics.astype('float32') / 255
train_answers = to_categorical(train_answers)

# transform testing data
test_pics = test_pics.reshape((10000,28,28,1))
test_pics = test_pics.astype('float32') / 255
test_answers = to_categorical(test_answers)

######################################
# BUILD NEURAL NETWORK
######################################

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
#model.fit(train_pics, train_answers, epochs=5, batch_size=128)

######################################
# SETUP MODEL MONITORING
######################################

# tensorboard callback (https://keras.io/callbacks/#tensorboard)
callbacks = TensorBoard(
    log_dir='tensorboard_log_dir',
    histogram_freq=1,
    write_graph=True,
    write_grads=True,
    write_images=True
)

model.fit(train_pics, train_answers, 
          epochs=5, 
          batch_size=128, 
          validation_split=0.2, 
          verbose=1, 
          callbacks=[callbacks])

# in anaconda prompt change directory to working directory
# type "tensorboard --logdir=./tensorboard_log_dir"

#plot_model(model, show_shapes=True, to_file='model.png')

######################################
# COLLECT PERFORMANCE INFORMATION
######################################

# baseline test accuracy measure
test_loss, test_accuracy = model.evaluate(test_pics, test_answers)
print(test_accuracy)

# extract output from layers
layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# find low probability records
max_prob = []
record_id = []

for i in range(0,60000):
    record_id.append(i)
    num = np.expand_dims(train_pics[i], axis=0)
    pred = model.predict_classes(num)
    prob = model.predict_proba(num)
    a = prob[0][:]
    max_prob.append(max(a))
    #print(max(a))
    #print(pred[0])

#print(max_prob)
#print(record_id)

plt.barh(record_id, max_prob)

# create dataframe of max_prob records
d = {'RecordID':record_id, 'MaximumProbability':max_prob}
df = pd.DataFrame(d)
sns.distplot(df['MaximumProbability'], bins=9)
sns.distplot(df['MaximumProbability'], hist=True, kde=False, rug=False, bins=9)
df_sort = df.sort_values(by=['MaximumProbability'])
#df_sort.head(25)
df_sort[df_sort.MaximumProbability <= 0.3]

######################################
# UNDERSTAND LAYER DETAILS
######################################

# walk through an specific example
n = 59101
plt.imshow(-np.array(train_pics[n], dtype='float').reshape((28, 28)), cmap='gray')
n = np.expand_dims(train_pics[n], axis=0)
p = model.predict_proba(n)
y_axis = np.arange(10).astype('str')
x_axis = p[0][:] #np.vectorize(probability)
plt.barh(y_axis, x_axis)

# walk through an example
#which_num_img = 59101
#plt.imshow(-np.array(train_pics[which_num_img], dtype='float').reshape((28, 28)), cmap='gray')
#example_num = np.expand_dims(train_pics[which_num_img], axis=0)

# make a prediction
#prediction = model.predict_classes(n)
#print(prediction)
#
#probability = model.predict_proba(n)
#print(probability)

# generate activations
activations = activation_model.predict(n)

# retreive activations
first_layer_activations = activations[0]
#print(first_layer_activations.shape)

# plot the channels of the activation of the first layer
for i in range(0,32):
    plt.subplot(4, 8, i+1)
    plt.axis('off')
    plt.imshow(first_layer_activations[0, :, :, i])

# plot the channels of the activation of the second layer
second_layer_activations = activations[1]
    
for i in range(0, 32):
    plt.subplot(4, 8, i+1)
    plt.axis('off')
    plt.imshow(second_layer_activations[0, :, :, i]) 

# plot the channels of the activation of the third layer
third_layer_activations = activations[2]
    
for i in range(0, 64):
    plt.subplot(8, 8, i+1)
    plt.axis('off')
    plt.imshow(third_layer_activations[0, :, :, i]) 

######################################
# UNDERSTAND FILTER DETAILS
######################################

# retrieve weights from the second hidden layer
filters, biases = model.layers[0].get_weights()

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot filters
f = []
for i in range(0,32):
	# get the filter
	f.append(filters[:, :, :, i])

# plot each channel separately
n_filters, ix = 6, 1
for j in range(0,32):
    # specify subplot and turn of axis
    ax = plt.subplot(4, 8, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    # plot filter channel in grayscale
    c = f[j]
    c = np.reshape(c, (3,3))
    plt.imshow(c, cmap='gray')
    ix += 1

# show the figure
plt.show()

######################################
# BUILD NEURAL NETWORK USING SEPARABLE
######################################

# convert CNN to depthwise separable convolution
model2 = models.Sequential()
model2.add(layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model2.add(layers.MaxPooling2D((2,2)))
model2.add(layers.SeparableConv2D(64, (3,3), activation='relu'))
model2.add(layers.MaxPooling2D((2,2)))
model2.add(layers.SeparableConv2D(64, (3,3), activation='relu'))
model2.summary()

# build neural network output layers
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.25))
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(10, activation='softmax'))
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(train_pics, train_answers, epochs=5, batch_size=128)

# baseline test accuracy measure
test_loss2, test_accuracy2 = model2.evaluate(test_pics, test_answers)
print(test_accuracy2)

# extract output from layers
layer_outputs2 = [layer.output for layer in model2.layers]
activation_model2 = models.Model(inputs=model2.input, outputs=layer_outputs2)

# find low probability records
max_prob2 = []
record_id2 = []

for i in range(0,60000):
    record_id2.append(i)
    num2 = np.expand_dims(train_pics[i], axis=0)
    pred2 = model2.predict_classes(num2)
    prob2 = model2.predict_proba(num2)
    a2 = prob2[0][:]
    max_prob2.append(max(a2))

# create dataframe of max_prob records
d2 = {'RecordID':record_id2, 'MaximumProbability':max_prob2}
df2 = pd.DataFrame(d2)
sns.distplot(df2['MaximumProbability'], hist=True, kde=False, rug=False, bins=9)
df_sort2 = df2.sort_values(by=['MaximumProbability'])
#df_sort.head(25)
df_sort2[df_sort2.MaximumProbability <= 0.5]

# walk through an specific example
n2 = 47759
plt.imshow(-np.array(train_pics[n2], dtype='float').reshape((28, 28)), cmap='gray')
n2 = np.expand_dims(train_pics[n2], axis=0)
p2 = model2.predict_proba(n2)
y_axis2 = np.arange(10).astype('str')
x_axis2 = p2[0][:] #np.vectorize(probability)
plt.barh(y_axis2, x_axis2)




