# If there is an error in training or validation using the trained model, this is probably because max lengths of Pluecker vectors in the trained model and the newly loaded data are different. Therefore, one needs to re-train the model, or only load a set of Pluecker coordinates with max length = max length of the trained model.

# Import library


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
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow
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


# Some definitions

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
# Volumes


with sqlite3.connect("dim_2_plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume,plucker_len FROM dim_2_plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 24000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

#X1=pad_sequences(X1, padding='post',value=0)



X1_train, X1_test, Y1_train, Y1_test = train_test_split(    X1, Y1, test_size=0.8,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X1_train, Y1_train)

Y1_pred = reg.predict(X1_test)
print(mean_absolute_error(Y1_pred, Y))

print(acc(Y1_test,Y1_pred,0.5))
print(acc(Y1_test,Y1_pred,1))
print(acc(Y1_test,Y1_pred,2))
print(acc(Y1_test,Y1_pred,3))
print(acc(Y1_test,Y1_pred,4))



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume,plucker_len FROM plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 24000", db)

X = df['plucker'].transform(literal_eval)
X=X.to_list()
Y=df['volume'].to_list()

#X=pad_sequences(X, padding='post',value=0)




X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.8,
    shuffle=True)

reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))




# If we do not further train any 3d data, and only use the model for 2d to predict 3d ones:

with sqlite3.connect("dim_2_plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume,plucker_len FROM dim_2_plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 24000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

#X1=pad_sequences(X1, padding='post',value=0)




X1_train, X1_test, Y1_train, Y1_test = train_test_split(    X1, Y1, test_size=0.8,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X1_train, Y1_train)

Y1_pred = reg.predict(X1_test)
print(mean_absolute_error(Y1_pred, Y))

print(acc(Y1_test,Y1_pred,0.5))
print(acc(Y1_test,Y1_pred,1))
print(acc(Y1_test,Y1_pred,2))
print(acc(Y1_test,Y1_pred,3))
print(acc(Y1_test,Y1_pred,4))



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume,plucker_len FROM plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 24000", db)

X = df['plucker'].transform(literal_eval)
X=X.to_list()
Y=df['volume'].to_list()

#X=pad_sequences(X, padding='post',value=0)


X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=23999,
    shuffle=True)

reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))


# Varying length



with sqlite3.connect("dim_2_plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume FROM dim_2_plucker ORDER BY RANDOM() LIMIT 30000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X1=pad_sequences(X1, padding='post',value=0)




len(X1[0])



X1_train, X1_test, Y1_train, Y1_test = train_test_split(    X1, Y1, test_size=0.5,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X1_train, Y1_train)

Y1_pred = reg.predict(X1_test)
print(mean_absolute_error(Y1_pred, Y))

print(acc(Y1_test,Y1_pred,0.5))
print(acc(Y1_test,Y1_pred,1))
print(acc(Y1_test,Y1_pred,2))
print(acc(Y1_test,Y1_pred,3))
print(acc(Y1_test,Y1_pred,4))



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume,plucker_len FROM plucker WHERE plucker_len<=45 ORDER BY RANDOM() LIMIT 40000", db)

X = df['plucker'].transform(literal_eval)
X=X.to_list()
Y=df['volume'].to_list()

X=pad_sequences(X, padding='post',value=0)



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.75,
    shuffle=True)

reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))


# all pad to length 560


with sqlite3.connect("dim_2_plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume FROM dim_2_plucker ORDER BY RANDOM() LIMIT 30000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X11=[];
for a in X1:
    a=a+[0]*(560-len(a))
    X11.append(a)
X1=X11




X1_train, X1_test, Y1_train, Y1_test = train_test_split(    X1, Y1, test_size=0.5,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X1_train, Y1_train)

Y1_pred = reg.predict(X1_test)
print(mean_absolute_error(Y1_pred, Y))

print(acc(Y1_test,Y1_pred,0.5))
print(acc(Y1_test,Y1_pred,1))
print(acc(Y1_test,Y1_pred,2))
print(acc(Y1_test,Y1_pred,3))
print(acc(Y1_test,Y1_pred,4))



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume,plucker_len FROM plucker ORDER BY RANDOM() LIMIT 40000", db)

X = df['plucker'].transform(literal_eval)
X=X.to_list()
Y=df['volume'].to_list()

X0=[];
for a in X:
    a=a+[0]*(560-len(a))
    X0.append(a)
X=X0



X1_train, X1_test, Y1_train, Y1_test = train_test_split(    X1, Y1, test_size=0.75,
    shuffle=True)

# MLP regressor:
reg.fit(X1_train, Y1_train)

Y1_pred = reg.predict(X1_test)
print(mean_absolute_error(Y1_pred, Y))

print(acc(Y1_test,Y1_pred,0.5))
print(acc(Y1_test,Y1_pred,1))
print(acc(Y1_test,Y1_pred,2))
print(acc(Y1_test,Y1_pred,3))
print(acc(Y1_test,Y1_pred,4))


################################################################################################
# Dual volumes


with sqlite3.connect("dim_2_plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,dual_volume,plucker_len FROM dim_2_plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 24000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y11=df['dual_volume'].to_list()

Y10=[];
Y1=[];
for a in range (0,len(Y11)):
    Y10.append(Fraction(Y11[a]))
    Y1.append(Y10[a].numerator/Y10[a].denominator)

#X1=pad_sequences(X1, padding='post',value=0)



X1_train, X1_test, Y1_train, Y1_test = train_test_split(    X1, Y1, test_size=0.8,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X1_train, Y1_train)

Y1_pred = reg.predict(X1_test)
print(mean_absolute_error(Y1_pred, Y))

print(acc(Y1_test,Y1_pred,0.5))
print(acc(Y1_test,Y1_pred,1))
print(acc(Y1_test,Y1_pred,2))
print(acc(Y1_test,Y1_pred,3))
print(acc(Y1_test,Y1_pred,4))



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,dual_volume,plucker_len FROM plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 24000", db)

X = df['plucker'].transform(literal_eval)
X=X.to_list()
Y1=df['dual_volume'].to_list()

Y0=[];
Y=[];
for a in range (0,len(Y1)):
    Y0.append(Fraction(Y1[a]))
    Y.append(Y0[a].numerator/Y0[a].denominator)

#X=pad_sequences(X, padding='post',value=0)



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.8,
    shuffle=True)

reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))
print(acc(Y_test,Y_pred,5))


# Even for higher training percentage
# Remember to re-train reg from 2d every time



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
print(mean_absolute_error(Y_pred, Y))

print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))
print(acc(Y_test,Y_pred,5))





