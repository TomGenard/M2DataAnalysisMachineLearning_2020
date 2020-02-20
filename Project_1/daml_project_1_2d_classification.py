#============================================================
#  MADE BY : TOM GENARD, SACHA DAUMAS-TSCHOPP
#  DATA ANALYSIS AND MACHINE LEARNING - PROJECT 1
#  CLASSIFICATION OF THE 2D ISING RESULTS - SIMPLE MODEL
#============================================================

import sys
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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Print the whole matrix instead of a reduced version
np.set_printoptions(threshold=sys.maxsize)

def read_t(t=0.25,root="./IsingData/"):
	if t > 0.:
		data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
	else:
		data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
	return np.unpackbits(data).astype(int).reshape(-1,1600)


def classify(X, y):
	clf = LogisticRegressionCV()
	clf.fit(X, y)
	print("Regression score = ", clf.fit(X, y).score(X, y))
	return clf

stack = []

labels = np.zeros((16,10000),)
o = 0
for i,t in enumerate(np.arange(0.25,4.01,0.25)): # np.arange(0.25,4.01,0.25) gives all numbers between 0.25 and 4.01 separated by 0.25
	y = np.ones(10000,dtype=int)
	if t > 2.25:
		y*=0
	stack.append(y)

	for p in range(len(y)) :
		labels[o][p] = y[p]

	o = o + 1


stack = []
for t in np.arange(0.25,4.01,0.25):
	stack.append(read_t(t))

X = np.vstack(stack)
pickle.dump(np.packbits(X), open('Ising2DFM_reSample_L40_T=All.pkl','wb'))

sum_per_T = []

x_reshaped_matrix = np.zeros((16,1),)
n = 0

labels_exp = np.zeros((16,10000),)

enable_logistic = input("Compute with homogeneity (0) or with full matrices (1) ? ")
# Homogeneity means using the calculated homogeneity factor to compute the logistic regression
# Matrices means using the configuration matrix as our X to compute the logistic regression

# COMPUTE THE HOMOGENEITY FACTOR

if ( enable_logistic == "0" ) :
	for i in range(16):
		config_total = 0
		for m in range(10000) :
			x = X[n].reshape(40,40) # Matrix containing all of the data for one configuration

			sum_local = 0

			# For each particle in a system, adds 1 for an up spin and removes 1 for a down spin
			for j in range(len(x)) :
				for k in range(len(x[j])) :
					if ( x[j][k] == 0 ) :
						sum_local = sum_local - 1
					else :
						sum_local = sum_local + 1

			if ( sum_local < 0 ) :
				sum_local = -1 * sum_local

			sum_local = sum_local / 16.
			config_total = config_total + sum_local

			###########################################################
			# COARSE CLASSIFICATION
			###########################################################

			if ( enable_logistic == "0" ) :
				if ( sum_local/100. > 0.5 ) : # If homogeneity above 50%, we have an ordered phase
					labels_exp[i][m] = 1
				else : # Otherwise, we have an disordered phase
					labels_exp[i][m] = 0

			###########################################################
			###########################################################

			n = n + 1 # All of the data are in a single line

		config_total = config_total / 10000.
		print("T = ", np.arange(0.25,4.01,0.25)[i])
		print("Homogenous factor : {0} %".format(config_total))

		sum_per_T.append(config_total)

###########################################################
# LOGISTIC CLASSIFICATION
###########################################################

if ( enable_logistic == "1" ) :
	labels_line = labels.reshape(160000,-1)

	logistic_test = classify(X, labels_line)

	labels_exp = logistic_test.predict(X).ravel()

labels_exp = labels_exp.reshape(16,10000)


# Compute the accuracy
accuracy_score = []

for i2 in range(len(labels)) :
	accuracy_score_local = 0
	dummy_index = 0
	for j2 in range (len(labels[i2])) :
		if ( labels[i2][j2] == labels_exp[i2][j2] ) :
			accuracy_score_local = accuracy_score_local + 1
		dummy_index = dummy_index + 1

	accuracy_score.append(accuracy_score_local/dummy_index)

T_index = 0.25
for i3 in range(len(accuracy_score)) :
	print("For T = {0}, accuracy score : {1}".format(T_index, accuracy_score[i3]))
	T_index = T_index + 0.25