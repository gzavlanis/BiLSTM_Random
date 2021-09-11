from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import random
from keras.utils.vis_utils import plot_model

# print 30000 random numbers between 0 and 1
for i in range(30000):
    print(random.uniform(0, 1))

# create a random waveform as input
X= np.array([random.uniform(0, 1) for _ in range(30000)])
print(X)
pyplot.plot(X)
pyplot.show()
Y= X # define output, same with input
print(Y)
pyplot.plot(Y)
pyplot.show()

# convert the arrays to tensors
X= tf.convert_to_tensor(X, dtype= tf.float32)
Y= tf.convert_to_tensor(Y, dtype= tf.float32)
print(X)
print(Y)

n_timesteps= 3 #define the timesteps of the problem

# activation function
def activation(x):
    return tf.maximum(0.01*x, x)

# loss function
def loss(pred, y):
    return tf.reduce_mean(tf.reduce_sum(tf.square(pred- y), axis=1))

# Create the deep BiLSTM network and make a figure of it
model= Sequential()
model.add(Bidirectional(LSTM(32, return_sequences= True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(128, activation= activation)))
model.add(Dense(128, activation= activation))
model.add(Dense(64, activation= activation))
model.add(Dense(1, activation= None))
model.compile(loss= loss, optimizer= Adam(learning_rate=0.01), metrics=[ 'accuracy' ])
print(model.summary())
plot_model(model, to_file='BiLSTM.png', show_shapes=True, show_layer_names=True)

# reshape data for entering the model
X= tf.reshape(X, [10000, n_timesteps, 1])
Y= tf.reshape(Y, [10000, n_timesteps, 1])
print(X)
print(Y)

# train the model and plot results for loss
def train_model(model, n_timesteps):
    hist= model.fit(X, Y, epochs= 50, batch_size= 64)
    loss= hist.history['loss']
    return loss

loss= train_model(model, n_timesteps)
pyplot.plot(loss, label= 'Loss')
pyplot.title('Training loss of the model')
pyplot.xlabel('epochs', fontsize= 18)
pyplot.ylabel('loss', fontsize= 18)
pyplot.grid()
pyplot.legend()
pyplot.show()

# predict the output Y
Yhat= model.predict(Y, verbose= 0)
Yhat= Yhat.reshape(30000, 1)
print(Yhat)

# Reshape X and Y back to normal size
X= tf.reshape(X, [30000, 1])
Y= tf.reshape(Y, [30000, 1])

# Plot real waveform and predicted waveform
pyplot.plot(Y, 'r', label= 'Real waveform')
pyplot.plot(Yhat, 'b', label= 'Predicted waveform')
pyplot.title('Plot of real vs predicted waveforms', fontsize= 16)
pyplot.legend()
pyplot.grid()
pyplot.show()

# create a new sequence and try to predict it
X1= np.array([random.uniform(0, 1) for _ in range(30000)])
X1= tf.convert_to_tensor(X1, dtype= tf.float32)
X1= tf.reshape(X1, [10000, n_timesteps, 1])
Yhat1= model.predict(X1, verbose= 0)
Yhat1= Yhat1.reshape(30000, 1)
print(Yhat1)
X1= tf.reshape(X1, [30000, 1])
pyplot.plot(X1, 'r', label= 'Real waveform')
pyplot.plot(Yhat1, 'b', label= 'Predicted waveform')
pyplot.title('Prediction of a random waveform without training with it', fontsize= 18)
pyplot.legend()
pyplot.grid()
pyplot.show()
