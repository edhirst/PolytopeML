# If there is an error in training or validation using the trained model, this is probably because max lengths of Pluecker vectors in the trained model and the newly loaded data are different. Therefore, one needs to re-train the model, or only load a set of Pluecker coordinates with max length = max length of the trained model.

#Import library

import sqlite3
from ast import literal_eval
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
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


#Some definitions

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
#Pluecker coords as input:
#load trained models:
reg=load('Plk_Vol_MLPReg.joblib')
model=keras.models.load_model('Plk_Vol_CNN')



# machine learning:
with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume FROM plucker ORDER BY RANDOM() LIMIT 800000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X=X1
Y=Y1

X=pad_sequences(X, padding='post',value=0)




X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)




Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))



#save trained model:
dump(reg,'Plk_Vol_MLPReg.joblib')


#CNN:

trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);



model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()





model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))




predictY = model.predict(validateX,verbose = 1)[:,0]



print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))


#save trained model:
model.save('Plk_Vol_CNN')


################################################################################################
#augmentation with gcd(length-1):

reg=load('gcdLen-1_Vol_MLPReg.joblib')
model=keras.models.load_model('gcdLen-1_Vol_CNN')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume FROM plucker ORDER BY RANDOM() LIMIT 800000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X=X1
Y=Y1


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



print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))


dump(reg,'gcdLen-1_Vol_MLPReg.joblib')


#CNN:

trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);



model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()



model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))



predictY = model.predict(validateX,verbose = 1)[:,0]



print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))



model.save('gcdLen-1_Vol_CNN')


################################################################################################
# fix plucker length
# ex1 length = 10



reg=load('Plk10_Vol_MLPReg.joblib')
model=keras.models.load_model('Plk10_Vol_CNN')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume, plucker_len FROM plucker WHERE plucker_len=10 ORDER BY RANDOM() LIMIT 5000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X=X1
Y=Y1




X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))



dump(reg,'Plk10_Vol_MLPReg.joblib')


#CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))


# In[33]:


predictY = model.predict(validateX,verbose = 1)[:,0]
print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))



model.save('Plk10_Vol_CNN')



reg=load('plk10gcdLen-1_Vol_MLPReg.joblib')
model=keras.models.load_model('plk10gcdLen-1_Vol_CNN')



X0=[];
for a in range (len(X1)):
    x0=appendgcd(X1[a],len(X1[a])-1)
    X0.append(x0)
        
X=X0;



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)




Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))





dump(reg,'plk10gcdLen-1_Vol_MLPReg.joblib')


#CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))




predictY = model.predict(validateX,verbose = 1)[:,0]
print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))




model.save('plk10gcdLen-1_Vol_CNN')

################################################################################################
# ex2 length = 35

reg=load('Plk35_Vol_MLPReg.joblib')
model=keras.models.load_model('Plk35_Vol_CNN')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume, plucker_len FROM plucker WHERE plucker_len=35 ORDER BY RANDOM() LIMIT 100000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X=X1
Y=Y1


X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))



dump(reg,'Plk35_Vol_MLPReg.joblib')


#CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))



predictY = model.predict(validateX,verbose = 1)[:,0]
print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))


# In[64]:


model.save('Plk35_Vol_CNN')



reg=load('plk35gcdLen-1_Vol_MLPReg.joblib')
model=keras.models.load_model('plk35gcdLen-1_Vol_CNN')




X0=[];
for a in range (len(X1)):
    x0=appendgcd(X1[a],len(X1[a])-1)
    X0.append(x0)
        
X=X0;



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))




dump(reg,'plk35gcdLen-1_Vol_MLPReg.joblib')


#CNN:

trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))



predictY = model.predict(validateX,verbose = 1)[:,0]
print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))



model.save('plk35gcdLen-1_Vol_CNN')

################################################################################################
# ex3 length = 56


reg=load('Plk56_Vol_MLPReg.joblib')
model=keras.models.load_model('Plk56_Vol_CNN')





with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,volume, plucker_len FROM plucker WHERE plucker_len=56 ORDER BY RANDOM() LIMIT 100000", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['volume'].to_list()

X=X1
Y=Y1




X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)




Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))




dump(reg,'Plk56_Vol_MLPReg.joblib')


#CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))




predictY = model.predict(validateX,verbose = 1)[:,0]
print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))


model.save('Plk56_Vol_CNN')



reg=load('plk56gcdLen-1_Vol_MLPReg.joblib')
model=keras.models.load_model('plk56gcdLen-1_Vol_CNN')



X0=[];
for a in range (len(X1)):
    x0=appendgcd(X1[a],len(X1[a])-1)
    X0.append(x0)
        
X=X0;



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# MLP regressor:
reg = MLPRegressor(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
reg.fit(X_train, Y_train)



Y_pred = reg.predict(X_test)
print(acc(Y_test,Y_pred,0.5))
print(acc(Y_test,Y_pred,1))
print(acc(Y_test,Y_pred,2))
print(acc(Y_test,Y_pred,3))
print(acc(Y_test,Y_pred,4))


dump(reg,'plk56gcdLen-1_Vol_MLPReg.joblib')


#CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
trainingX = np.array([X[a] for a in trainingindex]);
trainingY = np.array([Y[a] for a in trainingindex]);
validateX = np.array([X[a] for a in validateindex]);
validateY = np.array([Y[a] for a in validateindex]);

trainingX0=[]
validateX0=[]
for i in range (0,len(trainingX)):
    trainingX0.append(np.expand_dims(trainingX[i],axis=0))
trainingX=np.array([trainingX0[a] for a in range (0,len(trainingX0))]);
for i in range (0,len(validateX)):
    validateX0.append(np.expand_dims(validateX[i],axis=0))
validateX=np.array([validateX0[a] for a in range (0,len(validateX0))]);

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(1))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))



predictY = model.predict(validateX,verbose = 1)[:,0]
print(acc(validateY,predictY,0.5))
print(acc(validateY,predictY,1))
print(acc(validateY,predictY,2))
print(acc(validateY,predictY,3))
print(acc(validateY,predictY,4))


model.save('plk35gcdLen-1_Vol_CNN')




