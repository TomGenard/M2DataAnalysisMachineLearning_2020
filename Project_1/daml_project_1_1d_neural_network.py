#============================================================
#  MADE BY : TOM GENARD, SACHA DAUMAS-TSCHOPP
#  DATA ANALYSIS AND MACHINE LEARNING - PROJECT 1
#  STUDY OF THE ISING MODEL IN 1D WITH NEURAL NETWORKS (TENSORFLOW)
#============================================================

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import os

## Sci-kit
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import scipy.sparse as sp
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

np.set_printoptions(threshold=sys.maxsize)

def create_neural_network_keras(input_dim, n_neurons_layer1, n_neurons_layer2, n_neurons_layer3, n_categories, eta, lmbd):
	model = Sequential()
	model.add(Dense(n_neurons_layer1, input_dim=input_dim, activation='relu', kernel_regularizer=l2(lmbd)))
	model.add(Dense(n_neurons_layer2, activation='relu', kernel_regularizer=l2(lmbd)))
	model.add(Dense(n_neurons_layer3, activation='relu', kernel_regularizer=l2(lmbd)))
	model.add(Dense(n_categories, activation='softmax'))

	print("Full Model Architecture :")
	model.summary()
	
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae', 'acc'])
	
	return model

np.random.seed(123)

import warnings
warnings.filterwarnings('ignore')

### Define Ising model aprams
# System size (length)
L = 40
nb_states = 10000
J_value = 1.0
n_degree = 10 # Order of calculation

# Create 10000 random Ising states
states = np.random.choice([-1, 1], size=(nb_states,L))

def ising_energies(states,L):
	# This function calculates the energies of the states in the nn Ising Hamiltonian
	J = np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1)%L] -= J_value
	# Compute energies
	E = np.einsum('...i,ij,...j->...',states,J,states)

	return E

# Calculate Ising energies
energies = ising_energies(states,L)
energies_values = np.resize(energies,(len(energies),1)) # Transpose line matrix to vector

# Compute the X matrix, which is a vector with each line being the sum of the product
# of the neighboring states
state_sum = np.zeros((nb_states,1),)

j = 0
while ( j < nb_states ) :
	# print("j = ", j)
	k = 0
	state_sum_buffer = 0
	while ( k < L ) :
		if ( k+1 < L ) :
			state_sum_buffer = state_sum_buffer + (states[j][k]*states[j][k+1])
		else :
			state_sum_buffer = state_sum_buffer + (states[j][k]*states[j][0]) # BVK condition, loop

		k = k + 1
	state_sum[j] = state_sum_buffer

	j = j + 1

X = np.zeros((len(state_sum),n_degree+1))

l = 0
while ( l < n_degree + 1 ) :
	X[:,l] = state_sum[:,0]**l
	l = l + 1

#############################################################################################
#
# #   # ##### ####   ###   ####
# #  #  #	  #   # #   # #
# ###   ###   ####  #####  ###
# #  #  #	  #   # #   #	  #
# #   # ##### #   # #   # ####
#
#############################################################################################

# We split out train and test data
X_train, X_test, y_train, y_test = train_test_split(states, energies_values, test_size=0.2)

# We transform the y vectors into categorized matrices
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# We setup the different layers of our neural network
n_neurons_layer1 = 1024
n_neurons_layer2 = 1024
n_neurons_layer3 = 1024
n_categories     = len(y_train[0])
input_dim        = len(X_train[0])

epochs     = 20
batch_size = 32

eta_vals  = [1]
lmbd_vals = np.logspace(-5, 1, 7)

DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# We create empty matrices that will be filled with our accuracy and MSE results
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy  = np.zeros((len(eta_vals), len(lmbd_vals)))

train_loss = np.zeros((len(eta_vals), len(lmbd_vals)))
test_loss  = np.zeros((len(eta_vals), len(lmbd_vals)))

#####################################################################
#                            TRAINING
#####################################################################

es = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 100, min_delta = 1e-2)

for i, eta in enumerate(eta_vals):
	for j, lmbd in enumerate(lmbd_vals):
		# We create the neural network
		DNN = create_neural_network_keras(input_dim, n_neurons_layer1, n_neurons_layer2, n_neurons_layer3, n_categories, eta = eta, lmbd = lmbd)
		# We fit it
		history = DNN.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks=[es], validation_split=0.25)
		# We save the neural network state in an array that will have all of them for later use
		scores = DNN.evaluate(X_train, y_train)
		
		DNN_keras[i][j] = DNN
		
		print("Learning rate = ", eta)
		print("Lambda = ", lmbd)

		y_pred = DNN.predict(X_test)

		# We put the MSE and accuracy in the matrix
		train_loss[i][j] = scores[1]
		train_accuracy[i][j] = scores[2]

		print("MAE: %.3f" % scores[1])
		print("Accuracy: %.3f" % scores[2])

#####################################################################
#                              TEST
#####################################################################

# Compute the accuracy and MSE for the test data
for i in range(len(DNN_keras)) :
	for j in range(len(DNN_keras[i])) :
		scores = DNN_keras[i][j].evaluate(X_test, y_test)
		y_pred = DNN_keras[i][j].predict(X_test)

		test_loss[i][j] = scores[1]
		test_accuracy[i][j] = scores[2]


print(train_accuracy)
print(test_accuracy)

print(train_loss)
print(test_loss)