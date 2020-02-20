#============================================================
#  MADE BY : TOM GENARD, SACHA DAUMAS-TSCHOPP
#  ANALYSIS OF THE PULSAR DATA
#  CLASSIFICATION USING NEURAL NETWORKS AND LOGISTIC REGRESSIONS
#============================================================

import sys
import csv
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
import os
import scipy.sparse as sp

## Sci-kit
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import scipy.sparse as sp
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.tree import export_graphviz

## Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

#np.set_printoptions(threshold=sys.maxsize)

def classify(X, y):
	clf = LogisticRegressionCV(max_iter=1000)
	clf.fit(X, y)
	print("Accuracy score  = ", clf.fit(X, y).score(X, y))
	return clf

def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_neurons_layer3, n_categories, eta, lmbd):
	model = Sequential()
	model.add(Dense(n_neurons_layer1, input_dim=8, activation='relu', kernel_regularizer=l2(lmbd)))
	model.add(Dense(n_neurons_layer2, activation='relu', kernel_regularizer=l2(lmbd)))
	model.add(Dense(n_neurons_layer3, activation='relu', kernel_regularizer=l2(lmbd)))
	model.add(Dense(n_categories, activation='softmax'))
	
	print("Full Model Architecture :")
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])
	
	return model

data = []
star_data = [] # 8 parameters, 1 label

## READING THE DATA

DATA = pd.read_csv('pulsar_stars.csv')
X = DATA.to_numpy()[:,:-1]
y = DATA.to_numpy()[:,-1]

DATA_corr = DATA.corr().round(2)

DATA_corr.to_csv(r'DATA_corr.csv')

## SETTING UP THE DESIGN AND RESULT MATRIX
len_y = len(y)

y = np.resize(y,(len_y,1))

print(len(X))
print(len(y))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

print(scaler_x.fit(X))
xscale = scaler_x.transform(X)
print(scaler_y.fit(y))
yscale = scaler_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=0.2)

print("What classification mode to use ? (1 for Logistic, 2 for Neural Network, 3 for Decision Tree and 4 for Random Forest")
classification_mode = int(input())

#############################################################################################
#
# #      ###   #### #####  #### ##### #####  ####
# #     #   # #       #   #       #     #   #
# #     #   # # ###   #    ###    #     #   #
# #     #   # #   #   #       #   #     #   #
# #####  ###   #### ##### ####    #   #####  ####
#
#############################################################################################

if ( classification_mode == 1 ) :

	print("== LOGISTIC REGRESSION SELECTED ==")

	########################################################
	#					TRAINING DATA					 #
	########################################################

	print("TRAINING DATA")
	logistic_test = classify(X_train, y_train)

	y_exp_train = logistic_test.predict(X_train).ravel()

	########################################################
	#					  TEST DATA					   #
	########################################################

	y_exp_test = logistic_test.predict(X_test).ravel()

	accuracy_score = 0

	# TRANSFORM THE MATRICES INTO ARRAYS FOR EASIER USE

	y_exp_test = np.asarray(y_exp_test)
	y_test_logistic = np.asarray(y_test)

	# COMPUTE THE ACCURACY OF THE LOGISTIC REGRESSION

	dummy_index = 0
	for j in range (len(y_test_logistic)) :
		#print(y_train[j2], " ", y_exp_train[j2])
		if ( y_test_logistic[j] == y_exp_test[j] ) :
			accuracy_score = accuracy_score + 1
		dummy_index = dummy_index + 1

	accuracy_score = accuracy_score/dummy_index

	print("Accuracy score  = ", accuracy_score)

#############################################################################################
#
# #   # ##### ####   ###   ####
# #  #  #     #   # #   # #
# ###   ###   ####  #####  ###
# #  #  #     #   # #   #     #
# #   # ##### #   # #   # ####
#
#############################################################################################

if ( classification_mode == 2 ) :

	print("== NEURAL NETWORK SELECTED ==")

	# We setup the different layers
	n_neurons_layer1 = 1024
	n_neurons_layer2 = 1024
	n_neurons_layer3 = 1024
	n_categories	 = 2
	# We setup the epochs and batch size
	epochs	 = 1000
	batch_size = 32
	# We have found an idea value, so we set it only for that value
	eta_vals  = [0.001]
	lmbd_vals = [0.00001]
	# We setup the epochs and batch size
	DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
	# We create empty matrices that will receive all of our results, to be printed
	train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
	train_loss = np.zeros((len(eta_vals), len(lmbd_vals)))
	test_accuracy  = np.zeros((len(eta_vals), len(lmbd_vals)))
	test_loss = np.zeros((len(eta_vals), len(lmbd_vals)))

	#####################################################################
	#							TRAINING
	#####################################################################

	es = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 100, min_delta = 1e-2)

	y_train_cat = to_categorical(y_train)

	for i, eta in enumerate(eta_vals):
		for j, lmbd in enumerate(lmbd_vals):
			# We create the neural network
			DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_neurons_layer3, n_categories, eta = eta, lmbd = lmbd)
			# We fit it
			history = DNN.fit(X_train, y_train_cat, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks=[es], validation_split=0.25)
			# We save the neural network state in an array that will have all of them for later use
			DNN_keras[i][j] = DNN
			
			print("Learning rate = ", eta)
			print("Lambda = ", lmbd)

			# We put the MSE and accuracy in the matrix
			train_accuracy[i][j] = history.history['acc'][-1]
			train_loss[i][j] = history.history['loss'][-1]

			print("Accuracy: %.3f" % history.history['acc'][-1])
			print("MAE: %.3f" % history.history['loss'][-1])

			# Plot training & validation accuracy values
			plt.plot(history.history['acc'])
			plt.plot(history.history['val_acc'])
			plt.title('Model accuracy')
			plt.ylabel('Accuracy')
			plt.xlabel('Epoch')
			plt.legend(['Train', 'Test'], loc='upper left')
			plt.show()

			# Plot training & validation loss values
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.title('Model loss')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.legend(['Train', 'Test'], loc='upper left')
			plt.show()
			
			# Shap display
			import shap
			X = X_train[:500]
			print(X.shape)
			# Initialize js methods for visualization
			shap.initjs()
			# Create an instance of the DeepSHAP which is called DeepExplainer
			explainer = shap.DeepExplainer(DNN, X)
			# Fit the explainer on a subset of the data (you can try all but then gets slower)
			shap_values = explainer.shap_values(X[:10])

			words = DATA.columns[:-1]
			
			x_test_words = np.stack([np.array(words) for i in range(10)])
			shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0], matplotlib=True)
			

			# Serialize model to JSON
			DNN_json = DNN.to_json()
			with open("DNN.json", "w") as json_file:
				json_file.write(DNN_json)
			# Serialize weights to HDF5
			DNN.save_weights("DNN.h5")
			print("Saved model to disk")
			
	#####################################################################
	#							  TEST
	#####################################################################

	# Change our output matrix into a categorized matrix
	y_test_cat = to_categorical(y_test)
	# Compute the accuracy and MSE for the test data
	for i in range(len(DNN_keras)) :
		for j in range(len(DNN_keras[i])) :
			scores = DNN_keras[i][j].evaluate(X_test, y_test_cat)
			y_pred = DNN_keras[i][j].predict(X_test)

			confusion_matrix = metrics.confusion_matrix(y_test_cat.argmax(axis=1), y_pred.argmax(axis=1))

			print(confusion_matrix)

			test_accuracy[i,j] = scores[1]
			test_loss[i,j] = scores[2]

	print(train_accuracy)
	print(test_accuracy)
	print()
	print(train_loss)
	print(test_loss)
	np.savetxt("trainacc.csv", train_accuracy)
	np.savetxt("testacc.csv" , test_accuracy)

#############################################################################################
#
# ####  #####  #### #####  #### #####  ###  #   #     ##### ####  ##### #####
# #   # #     #       #   #       #   #   # ##  #       #   #   # #     #
# #   # ###   #       #    ###    #   #   # # # #       #   ####  ###   ###
# #   # #     #       #       #   #   #   # #  ##       #   #   # #     #
# ####  #####  #### ##### ####  #####  ###  #   #       #   #   # ##### #####
#
#############################################################################################

if ( classification_mode == 3 ) :

	print("== DECISION TREE SELECTED ==")
	# We setup the regressor for different depths, and we fit it
	regr_1 = DecisionTreeRegressor(max_depth=2)
	regr_2 = DecisionTreeRegressor(max_depth=5)
	regr_4 = DecisionTreeRegressor(max_depth=8)
	regr_3 = DecisionTreeRegressor(max_depth=11)
	regr_1.fit(X_train, y_train)
	regr_2.fit(X_train, y_train)
	regr_3.fit(X_train, y_train)
	regr_4.fit(X_train, y_train)
	# We compute the test data results
	y_test_1 = regr_1.predict(X_test)
	y_test_2 = regr_2.predict(X_test)
	y_test_3 = regr_3.predict(X_test)
	y_test_4 = regr_4.predict(X_test)
	# And the train data
	y_train_1 = regr_1.predict(X_train)
	y_train_2 = regr_2.predict(X_train)
	y_train_3 = regr_3.predict(X_train)
	y_train_4 = regr_4.predict(X_train)

	# Here, we transform our real number into a boolean 0 or 1 response
	y_test_1[y_test_1<0.5] = 0
	y_test_1[y_test_1>=0.5] = 1

	y_test_2[y_test_2<0.5] = 0
	y_test_2[y_test_2>=0.5] = 1

	y_test_3[y_test_3<0.5] = 0
	y_test_3[y_test_3>=0.5] = 1

	y_test_4[y_test_4<0.5] = 0
	y_test_4[y_test_4>=0.5] = 1

	y_train_1[y_train_1<0.5] = 0
	y_train_1[y_train_1>=0.5] = 1

	y_train_2[y_train_2<0.5] = 0
	y_train_2[y_train_2>=0.5] = 1

	y_train_3[y_train_3<0.5] = 0
	y_train_3[y_train_3>=0.5] = 1

	y_train_4[y_train_4<0.5] = 0
	y_train_4[y_train_4>=0.5] = 1

	# We compute the accuracy
	mse_train_1 = accuracy_score(y_train, y_train_1)
	mse_train_2 = accuracy_score(y_train, y_train_2)
	mse_train_3 = accuracy_score(y_train, y_train_3)
	mse_train_4 = accuracy_score(y_train, y_train_4)

	print("Accuracy train 1 (2) = ", mse_train_1)
	print("Accuracy train 2 (5) = ", mse_train_2)
	print("Accuracy train 4 (8) = ", mse_train_4)
	print("Accuracy train 3 (11) = ", mse_train_3)

	mse_test_1 = accuracy_score(y_test, y_test_1)
	mse_test_2 = accuracy_score(y_test, y_test_2)
	mse_test_3 = accuracy_score(y_test, y_test_3)
	mse_test_4 = accuracy_score(y_test, y_test_4)

	print("Accuracy test 1 (2) = ", mse_test_1)
	print("Accuracy test 2 (5) = ", mse_test_2)
	print("Accuracy test 4 (8) = ", mse_test_4)
	print("Accuracy test 3 (11) = ", mse_test_3)

	# And we print the confusion matrix
	confusion_matrix_dt = metrics.confusion_matrix(y_test, y_test_2)
	print("Confusion matrix :")
	print(confusion_matrix_dt)

#############################################################################################
#
# ####   ###  #   # ####   ###  #   #     #####  ###  ####  #####  #### #####
# #   # #   # ##  # #   # #   # ## ##     #     #   # #   # #     #       #
# ####  ##### # # # #   # #   # # # #     ###   #   # ####  ###    ###    #
# #   # #   # #  ## #   # #   # #   #     #     #   # #   # #         #   #
# #   # #   # #   # ####   ###  #   #     #      ###  #   # ##### ####    #
#
#############################################################################################

if ( classification_mode == 4 ) :

	print("== RANDOM FOREST SELECTED ==")
	# We setup the classifiers
	rf_1 = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=1000)
	rf_2 = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1000)
	rf_3 = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=1000)
	rf_4 = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=1000)
	rf_fit_1 = rf_1.fit(X_train, y_train)
	rf_fit_2 = rf_2.fit(X_train, y_train)
	rf_fit_3 = rf_3.fit(X_train, y_train)
	rf_fit_4 = rf_4.fit(X_train, y_train)
	# We compute the test data results
	y_rf_test_1 = rf_fit_1.predict(X_test)
	y_rf_test_2 = rf_fit_2.predict(X_test)
	y_rf_test_3 = rf_fit_3.predict(X_test)
	y_rf_test_4 = rf_fit_4.predict(X_test)
	# And the train data
	y_rf_train_1 = rf_fit_1.predict(X_train)
	y_rf_train_2 = rf_fit_2.predict(X_train)
	y_rf_train_3 = rf_fit_3.predict(X_train)
	y_rf_train_4 = rf_fit_4.predict(X_train)

	# Same as for the decision tree
	y_rf_train_1[y_rf_train_1<0.5] = 0
	y_rf_train_1[y_rf_train_1>=0.5] = 1

	y_rf_train_2[y_rf_train_2<0.5] = 0
	y_rf_train_2[y_rf_train_2>=0.5] = 1

	y_rf_train_3[y_rf_train_3<0.5] = 0
	y_rf_train_3[y_rf_train_3>=0.5] = 1

	y_rf_train_4[y_rf_train_4<0.5] = 0
	y_rf_train_4[y_rf_train_4>=0.5] = 1

	y_rf_test_1[y_rf_test_1<0.5] = 0
	y_rf_test_1[y_rf_test_1>=0.5] = 1

	y_rf_test_2[y_rf_test_2<0.5] = 0
	y_rf_test_2[y_rf_test_2>=0.5] = 1

	y_rf_test_3[y_rf_test_3<0.5] = 0
	y_rf_test_3[y_rf_test_3>=0.5] = 1

	y_rf_test_4[y_rf_test_4<0.5] = 0
	y_rf_test_4[y_rf_test_4>=0.5] = 1

	# We compute the accuracy
	acc_train_1 = accuracy_score(y_train, y_rf_train_1)
	acc_train_2 = accuracy_score(y_train, y_rf_train_2)
	acc_train_3 = accuracy_score(y_train, y_rf_train_3)
	acc_train_4 = accuracy_score(y_train, y_rf_train_4)

	print("Accuracy train 1 (5) = ", acc_train_1)
	print("Accuracy train 2 (10) = ", acc_train_2)
	print("Accuracy train 3 (15) = ", acc_train_3)
	print("Accuracy train 4 (20) = ", acc_train_4)

	acc_test_1 = accuracy_score(y_test, y_rf_test_1)
	acc_test_2 = accuracy_score(y_test, y_rf_test_2)
	acc_test_3 = accuracy_score(y_test, y_rf_test_3)
	acc_test_4 = accuracy_score(y_test, y_rf_test_4)

	print("Accuracy test 1 (5) = ", acc_test_1)
	print("Accuracy test 2 (10) = ", acc_test_2)
	print("Accuracy test 3 (15) = ", acc_test_3)
	print("Accuracy test 4 (20) = ", acc_test_4)
	
	confusion_matrix_rf = metrics.confusion_matrix(y_test, y_rf_test_4)
	print("Confusion matrix :")
	print(confusion_matrix_rf)