#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:36:00 2020

@author: maxim_anisimov
"""

#%% Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

"""
Student Number: 527144
"""
column_number = '44'

#%% Functions 

# use A.dot(b) when b may be a scalar
# how to deal with scalars??? inversion, transpose, multiplication

def multiv_norm_mean(y, X, b, B):
    mean = (X.T@X + B.I).I @ (X.T@y + (B.I).dot(b))
    return(mean)

def multiv_norm_variance(X, B, Sigma_squared):
    Var = Sigma_squared * (X.T@X + B.I).I
    return(Var)

# Settings
np.random.seed(1997)
#setwd('/Users/maxim_anisimov/Desktop/ERASMUS/Studying/1 module/Bayesian Econometrics/HA/')

#%% Open data
data_path = '/Users/maxim_anisimov/Desktop/ERASMUS/Studying/1 module/Bayesian Econometrics/HA/data/'

returns = pd.read_excel(data_path + 'returns.xls')
Jan = pd.read_excel(data_path + 'Jan.xls')
rf = pd.read_excel(data_path + 'rf.xls')
rm = pd.read_excel(data_path +'rm.xls')

df = pd.DataFrame({'const': np.ones(len(rm)),
                   '3M yield change': rf[rf.columns[2]],
                   'S&P 500 log return': rm[rm.columns[2]],
                   'Jan': Jan['Jan']})

#%% Data Preprocessing
""""
Model: rt =β0 +β1rtf +β2rtm +β3Jant +εt

Prior
β|σ^2 ∼ N(0, σ^2*I_4),
p(σ^2) ∝ σ^{−2}.
"""

# Data 
#y = np.asmatrix(np.random.randn(500, 1))
#X = np.asmatrix(np.random.randn(500, 4))

y = np.asmatrix(returns[returns.columns[int(column_number)+1]]).T
X = np.asmatrix(df)

b = np.matrix([[0],[0],[0],[0]]) # mean beta prior
B = np.asmatrix(np.diag([1,1,1,1])) # prior proportional matrix
Time = y.shape[0]
N = X.shape[1]

# Gibbs sampler parameters
n_sim = 10**5 # number of simulations (size of sample if final posterior sample)
n_burn = 10**3 # burn-in size
k = 10 # thin value
"""
Then the number of ALL simulations is n_sim*k + n_burn
"""

beta_initial = np.matrix([[0],[0],[0],[0]]) # initial beta values for the draws in the first simulation

#%%
##################
# SIMULATIONS ####
##################

# matrix to save draws
draw_matrix = np.empty((n_burn+n_sim*k, 5))
draw_matrix[:] = np.nan 
colnames_draw_matrix = ['sigma_sqrd', 'const', '3M Yield Change', 'SP500 Return', 'Jan']

# Prepare mean of the prior
beta_mean = multiv_norm_mean(y, X, b, B)

# create n_sim*k + n_burn draws
for n_draw in range(n_sim*k+n_burn):
    
    print("Draw #" + str(n_draw))
    if n_draw == 0:
        #first draw -> use initial betas
        beta_prev = beta_initial
    else:
        # there were previous draws -> take them to simulate sigma^2
        beta_prev = np.asmatrix(draw_matrix[n_draw-1, 1:5]).T
  
    ### draw sigma_sqrd conditional on betas and y
    RSS_prev = (y - X@beta_prev).T @ (y - X@beta_prev)  # construct residual sum of squares
    mu_IG = ( RSS_prev + (b-beta_prev).T @ np.linalg.inv(B) @ (b-beta_prev) )[0,0]
    chisq_rv = np.random.chisquare(df=Time+k, size=1)[0] # random chi-squared rv with T+k degrees of freedom
    sigma_sqrd_draw = mu_IG/chisq_rv
  
    # draw betas conditional on sigma_squared_draw and beta
    beta_var = multiv_norm_variance(X=X, B=B, Sigma_squared=sigma_sqrd_draw)
    beta_draw = np.random.multivariate_normal(mean=beta_mean.T.tolist()[0], cov=beta_var, size=1)[0]
  
    current_draw = np.append(sigma_sqrd_draw, beta_draw)
    #print(current_draw)
  
    draw_matrix[n_draw] = current_draw
  

draw_matrix = draw_matrix[n_burn:(n_burn+n_sim*k)] # discard burn-in sample
# Is correlation of draws present?
#for (col_name in colnames(draw.matrix)){
#  acf(draw.matrix[,col_name], main=col_name)
#}
#"Significant first AC in sigma squared"

#%% Thinning
draw_matrix =  draw_matrix[::k]
# check whether thinning helps
#for (col_name in colnames(draw.matrix)){
#  acf(draw.matrix[,col_name], main=col_name)
#}
# Now, no significant spikes at all

#%% Traceplots
for col_num in range(draw_matrix.shape[1]):
  plt.plot(draw_matrix[:,col_num])
  plt.title(colnames_draw_matrix[col_num])
  plt.show();

#%% Posterior Results
post_mean = np.mean(draw_matrix, axis=0)
print('Posterior means:', post_mean)
# posterior population variance with correction
post_var = np.diag(np.var(draw_matrix, axis=0)) * (draw_matrix.shape[0]-1)/draw_matrix.shape[0]
print('Posterior variance:', post_var)

# Posterior Odd
print('Posterior Odd of the 3M Yield Change:',
      np.sum(draw_matrix[:, 2] > 0) / np.sum(draw_matrix[:, 2] < 0))

# Posterior Distribution of 3M Yield Change
plt.hist(draw_matrix[:,2], density=True, bins=100)
plt.axvline(post_mean[2], color='k', ls='--')
plt.title('Posterior Distribution of 3M Yield Change, %');
