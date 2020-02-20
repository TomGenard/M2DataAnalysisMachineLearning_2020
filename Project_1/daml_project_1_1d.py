#============================================================
#  MADE BY : TOM GENARD, SACHA DAUMAS-TSCHOPP
#  DATA ANALYSIS AND MACHINE LEARNING - PROJECT 1
#  STUDY OF THE ISING MODEL IN 1D
#============================================================

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os

## Sci-kit
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import scipy.sparse as sp
from sklearn.utils import resample

np.random.seed(123)

import warnings
# Comment this to turn on warnings
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

#######################################################################
#
#  ###  #      ####
# #   # #     #
# #   # #      ###
# #   # #         #
#  ###  ##### ####
#
#######################################################################

print("======  OLS  ======")

regressor_ols = LinearRegression().fit(X, energies_values)
print("Beta matrix = ", regressor_ols.coef_)

#######################################################################
# MSE AND R2 TIME
#######################################################################

E_predicted = np.dot(X, regressor_ols.coef_.transpose())

MSE = metrics.mean_squared_error(energies_values, E_predicted)
print("  OLS - MSE coefficients = ", MSE)

print()

#######################################################################
#
# ####  ##### ####   #### #####
# #   #   #   #   # #     #
# ####    #   #   # # ### ###
# #   #   #   #   # #   # #
# #   # ##### ####   #### #####
#
#######################################################################

print("====== RIDGE ======")

lambda_val = 0.0001

regressor_ridge = Ridge(alpha=lambda_val,fit_intercept=False).fit(X, energies_values)
print("Beta matrix = ", regressor_ridge.coef_)

#######################################################################
# MSE AND R2 TIME
#######################################################################

E_predicted = np.dot(X, regressor_ridge.coef_.transpose())

MSE = metrics.mean_squared_error(energies_values, E_predicted)
print("RIDGE - MSE coefficients = ", MSE)

print()

#######################################################################
#
# #      ###   ####  ####  ###
# #     #   # #     #     #   #
# #     #####  ###   ###  #   #
# #     #   #     #     # #   #
# ##### #   # ####  ####   ###
#
#######################################################################

print("====== LASSO ======")

regressor_lasso = Lasso(alpha=lambda_val,fit_intercept=False).fit(X, energies_values)
print("Beta matrix = ", regressor_lasso.coef_)

#######################################################################
# MSE AND R2 TIME
#######################################################################

E_predicted = np.dot(X, regressor_lasso.coef_.transpose())

MSE = metrics.mean_squared_error(energies_values, E_predicted)
print("LASSO - MSE coefficients = ", MSE)