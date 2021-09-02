# If there is an error in training or validation using the trained model, this is probably because max lengths of Pluecker vectors in the trained model and the newly loaded data are different. Therefore, one needs to re-train the model, or only load a set of Pluecker coordinates with max length = max length of the trained model.

#Import library


import sqlite3
from ast import literal_eval
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from math import gcd
import itertools
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf
import keras
from keras.layers import Dense,Activation,Dropout
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,Flatten
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.utils import plot_model, to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import layers
from keras.optimizers import Adam, Adamax

from fractions import Fraction

#Some definitions:

def acc(test,pred,maxdiff):
    count=0
    for i in range (len(pred)):
        if abs(pred[i]-test[i])<=maxdiff:
            count=count+1
    return (count/len(pred))


def listgcd(l):
    d=l[0]
    for i in range (1,len(l)):
        d=gcd(d,l[i])
    return d

def appendgcd(l,tlen):
    l0=l;
    ts=list(itertools.combinations(l,tlen));
    for a in ts:
        l0.append(listgcd(a))
    return l0



################################################################################################
# Pluecker coords as input:
# load trained model:
reg=load('Plk_DualVol_MLPReg.joblib')



# machine learning:

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,dual_volume FROM plucker ORDER BY RANDOM() LIMIT 800000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['dual_volume'].to_list()

X=X1
Y0=[];
Y=[];
for a in range (0,len(Y1)):
    Y0.append(Fraction(Y1[a]))
    Y.append(Y0[a].numerator/Y0[a].denominator)

X=pad_sequences(X, padding='post',value=0)




X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))
print(acc(Y_test,Y_pred,5))


# save trained model:
dump(reg,'Plk_DualVol_MLPReg.joblib')



################################################################################################
# augmentation with gcd(length-1)


reg=load('gcdLen-1_DualVol_MLPReg.joblib')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,dual_volume FROM plucker ORDER BY RANDOM() LIMIT 800000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['dual_volume'].to_list()

X=X1
Y0=[];
Y=[];
for a in range (0,len(Y1)):
    Y0.append(Fraction(Y1[a]))
    Y.append(Y0[a].numerator/Y0[a].denominator)



X0=[];
for a in range (len(X1)):
    x0=appendgcd(X1[a],len(X1[a])-1)
    X0.append(x0)
        
X=X0;
X=pad_sequences(X, padding='post',value=0)



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)


Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))
print(acc(Y_test,Y_pred,5))


dump(reg,'gcdLen-1_DualVol_MLPReg.joblib')



################################################################################################
# fix plucker length
# ex length = 10


reg=load('Plk10_DualVol_MLPReg.joblib')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,dual_volume, plucker_len FROM plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 50000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['dual_volume'].to_list()

X=X1
Y0=[];
Y=[];
for a in range (0,len(Y1)):
    Y0.append(Fraction(Y1[a]))
    Y.append(Y0[a].numerator/Y0[a].denominator)

X=pad_sequences(X, padding='post',value=0)


print(len(X))
print(min(Y))
print(max(Y))



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)


Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))
print(acc(Y_test,Y_pred,5))


dump(reg,'Plk10_DualVol_MLPReg.joblib')






reg=load('plk10gcdLen-1_DualVol_MLPReg.joblib')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,dual_volume, plucker_len FROM plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 50000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['dual_volume'].to_list()

X=X1
Y0=[];
Y=[];
for a in range (0,len(Y1)):
    Y0.append(Fraction(Y1[a]))
    Y.append(Y0[a].numerator/Y0[a].denominator)
    
X0=[];
for a in range (len(X1)):
    x0=appendgcd(X1[a],len(X1[a])-1)
    X0.append(x0)
        
X=X0;
X=pad_sequences(X, padding='post',value=0)



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))
print(acc(Y_test,Y_pred,5))


dump(reg,'plk10gcdLen-1_DualVol_MLPReg.joblib')





