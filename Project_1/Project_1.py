import sys
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
#============================================================
#============================================================

#Define the seed
np.random.seed(12)
#Comment this to turn on warnings
warnings.filterwarnings('ignore')
### define Ising model aprams
# system size
L=40
#polynom variable
polynom = 10
#int(input("Order of complexity (Please enter a number) : "))
# create 10000 random Ising states
nb_imput = 10000
states = np.random.choice([-1, 1], size=(nb_imput,L))
states_values = states[:,0]
def ising_energies(states,L):
#This function calculates the energies of the states in the nn Ising Hamiltonian
    J = np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
        # compute energies withthe einstein summation
        E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

# calculate Ising energies
energies = ising_energies(states,L)
energies_vector = np.resize(energies, (len(energies), 1))
# print('ENERGIES')
# print(energies_vector)

big_func = make_pipeline(PolynomialFeatures(degree=polynom), LinearRegression(fit_intercept=False))

sum_states = np.zeros((nb_imput,1),)
k = 0
while k < nb_imput:
    i = 0
    sum_states_value = 0 
    X = np.zeros((len(energies),k))
    while (i < L):
        if (i + 1 >= L):
            sum_states_value += states[k][i] * states[k][0]
        else:
            sum_states_value += states[k][i] * states[k][i+1]
        i = i + 1
    sum_states[k] = sum_states_value
    k = k + 1

X = np.zeros((len(states),polynom))
for j in range(polynom):
    X[:,j] = sum_states[:,0]**j

<<<<<<< HEAD
X_train, X_test, Y_train, Y_test = train_test_split(X, energies_vector, test_size=0.2)
=======
Data = [states, energies]
X_train = Data[0][:nb_imput]
Y_train = Data[1][:nb_imput]
X_test  = Data[0][:3*nb_imput//2]
Y_test  = Data[1][:3*nb_imput//2]
>>>>>>> 7bd68f96f650829dc839689bbc22f1178ca4eff7
# print('======= X =======')
# print(X)
#=============================================================
#                            OLS
#=============================================================
# regressor = LinearRegression(fit_intercept = False)
# regressor.fit(X_, energies_vector)
# print('======= REGRESSOR COEF =======')
# print(regressor.coef_)
regressor_train = LinearRegression(fit_intercept = False)
regressor_train.fit(X_train, Y_train)
print('======= REGRESSOR COEF TRAIN=======')
print(regressor_train.coef_)

regressor_test = LinearRegression(fit_intercept = False)
regressor_test.fit(X_test, Y_test)
print('======= REGRESSOR COEF TEST=======')
print(regressor_test.coef_)

#=============================================================
#                         MSE OLS
#=============================================================                          
OLS_tilda_train = np.dot(X_train, regressor_train.coef_.transpose())
MSE_function_OLS_train = metrics.mean_squared_error(Y_train, OLS_tilda_train)
print('====== MSE OLS TRAIN ======')
print(MSE_function_OLS_train)

OLS_tilda_test = np.dot(X_test, regressor_test.coef_.transpose())
MSE_function_OLS_test = metrics.mean_squared_error(Y_test, OLS_tilda_test)
print('====== MSE OLS TEST ======')
print(MSE_function_OLS_test)
#=============================================================
<<<<<<< HEAD
#                         R_square OLS
#=============================================================
r_square_train = regressor_train.score(X_train, Y_train)
print('======R_square_train======')
print(r_square_train)

r_square_test = regressor_test.score(X_test, Y_test)
print('======R_square_test======')
print(r_square_test)
print('======================================================')
=======
#                         R_square_train OLS
#=============================================================
r_square_train = metrics.r2_score(energies, Y_train)
print('======R_square_train======')
print(r_square_train)

r_square_test = metrics.r2_score(energies, Y_test)
print('======R_square_test======')
print(r_square_test)
>>>>>>> 7bd68f96f650829dc839689bbc22f1178ca4eff7

#=============================================================
#                        RIDGE METHOD
#=============================================================
lamb = 10000

<<<<<<< HEAD
Ridge_regressor_train = Ridge(lamb, fit_intercept = False)
Ridge_regressor_train.fit(X_train, Y_train)
=======
Ridge_regressor = Ridge(lamb, fit_intercept = False)
Ridge_regressor.fit(X, energies_vector)
>>>>>>> 7bd68f96f650829dc839689bbc22f1178ca4eff7
print('=======  RIDGE METHOD  =======')
print(Ridge_regressor_train.coef_)

Ridge_regressor_test = Ridge(lamb, fit_intercept = False)
Ridge_regressor_test.fit(X_test, Y_test)
print('=======  RIDGE METHOD  =======')
print(Ridge_regressor_test.coef_)


 #=============================================================
 #                          MSE RIDGE
 #=============================================================                          
<<<<<<< HEAD
RIDGE_tilda_train = np.dot(X_train, Ridge_regressor_train.coef_.transpose())
MSE_function_RIDGE_train = metrics.mean_squared_error(Y_train, RIDGE_tilda_train)
print('====== MSE RIDGE_TRAIN  ======')
print(MSE_function_RIDGE_train)

RIDGE_tilda_test = np.dot(X_test, Ridge_regressor_test.coef_.transpose())
MSE_function_RIDGE_test = metrics.mean_squared_error(Y_test, RIDGE_tilda_test)
print('====== MSE RIDGE_TEST  ======')
print(MSE_function_RIDGE_test)

#=============================================================
#                         R_square RIDGE
#=============================================================
r_square_train_Ridge = Ridge_regressor_train.score(X_train, Y_train)
print('======R_square_train======')
print(r_square_train_Ridge)

r_square_test_Ridge = Ridge_regressor_test.score(X_test, Y_test)
print('======R_square_test======')
print(r_square_test_Ridge)
print('======================================================')

=======
RIDGE_tilda = np.dot(X, Ridge_regressor.coef_.transpose())
MSE_function_RIDGE = metrics.mean_squared_error(energies_vector, RIDGE_tilda)
print('====== MSE RIDGE  ======')
print(MSE_function_RIDGE)
>>>>>>> 7bd68f96f650829dc839689bbc22f1178ca4eff7
#=============================================================
#                        LASSO METHOD
#=============================================================

Lasso_regressor_train = linear_model.Lasso(lamb, fit_intercept = False)
Lasso_regressor_train.fit(X_train, Y_train)
print('=======      LASSO     =======')
print(Lasso_regressor_train.coef_)

Lasso_regressor_test = linear_model.Lasso(lamb, fit_intercept = False)
Lasso_regressor_test.fit(X_test, Y_test)
print('=======      LASSO     =======')
print(Lasso_regressor_test.coef_)

 #=============================================================
 #                         MSE LASSO
 #=============================================================

<<<<<<< HEAD
LASSO_tilda_train = np.dot(X_train, Lasso_regressor_train.coef_.transpose())
MSE_function_LASSO_train = metrics.mean_squared_error(Y_train, LASSO_tilda_train)
print('====== MSE LASSO TRAIN======')
print(MSE_function_LASSO_train)

LASSO_tilda_test = np.dot(X_test, Lasso_regressor_test.coef_.transpose())
MSE_function_LASSO_test = metrics.mean_squared_error(Y_test, LASSO_tilda_test)
print('====== MSE LASSO TEST======')
print(MSE_function_LASSO_test)
#=============================================================
#                         R_square LASSO
#=============================================================
r_square_train_Lasso = Lasso_regressor_train.score(X_train, Y_train)
print('======R_square_train_Lasso======')
print(r_square_train_Lasso)

r_square_test_Lasso = Lasso_regressor_test.score(X_test, Y_test)
print('======R_square_test_Lasso======')
print(r_square_test_Lasso)
print('======================================================')
=======
LASSO_tilda = np.dot(X, Lasso_regressor.coef_.transpose())
MSE_function_LASSO = metrics.mean_squared_error(energies_vector, LASSO_tilda)
print('====== MSE LASSO ======')
print(MSE_function_LASSO)

print('LAMBDA')
print(lamb)
