# If there is an error in training or validation using the trained model, this is probably because max lengths of Pluecker vectors in the trained model and the newly loaded data are different. Therefore, one needs to re-train the model, or only load a set of Pluecker coordinates with max length = max length of the trained model.

# Import library


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
# Pluecker coords as input:
# load trained model:
clf=load('Plk_Ref_RandomForest.joblib')
#clf=load('Plk_Ref_MLPClf.joblib')      #(uncomment this line to load model for MLP)
model=keras.models.load_model('Plk_Ref_CNN')


# machine learning:

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive FROM plucker WHERE is_reflexive=1 ORDER BY RANDOM() LIMIT 42943", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['is_reflexive'].to_list()

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive FROM plucker WHERE is_reflexive=0 ORDER BY RANDOM() LIMIT 42943", db)
    
X0 = df['plucker'].transform(literal_eval)
X0=X0.to_list()
Y0=df['is_reflexive'].to_list()

X=X1+X0
Y=Y1+Y0

X=pad_sequences(X, padding='post',value=0)




X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# random forest classifier. n_estimators random trees of max_depth:
clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')



# save trained model:
dump(clf,'Plk_Ref_RandomForest.joblib')


# MLP classifier:
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')


# save trained model:
dump(clf,'Plk_Ref_MLPClf.joblib')



# CNN classifier:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
X_train = np.array([X[a] for a in trainingindex]);
Y_train = np.array([Y[a] for a in trainingindex]);
X_test = np.array([X[a] for a in validateindex]);
Y_test = np.array([Y[a] for a in validateindex]);

trainingX = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
validateX  = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
Y_train_re = np.reshape(Y_train,(Y_train.shape[0],1))
Y_test_re = np.reshape(Y_test,(Y_test.shape[0],1))

trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')


model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()


model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))



predictY = model.predict(validateX,verbose = 1)



predictYround=np.around(predictY)
n=0;
for a in range(len(validateY)):
    if (predictYround[a]==validateY[a]).all():
        n+=1;
n/len(validateY)



# save trained model:
model.save('Plk_Ref_CNN')


################################################################################################
# One-hot encoding

clf=load('OneHot_Ref_RandomForest.joblib')
#clf=load('OneHot_Ref_MLPClf.joblib')
model=keras.models.load_model('OneHot_Ref_CNN')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive FROM plucker WHERE is_reflexive=1 ORDER BY RANDOM() LIMIT 42943", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['is_reflexive'].to_list()

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive FROM plucker WHERE is_reflexive=0 ORDER BY RANDOM() LIMIT 42943", db)
    
X0 = df['plucker'].transform(literal_eval)
X0=X0.to_list()
Y0=df['is_reflexive'].to_list()

X=X1+X0
Y=Y1+Y0

Xmin=min([item for sublist in X for item in sublist])
Xmax=max([item for sublist in X for item in sublist])

X0=np.array([np.array(X[a]) for a in range (0,len(X))]);




X=[];
for a in range (0,len(X0)):
    X.append(tf.keras.utils.to_categorical(X0[a]-Xmin, num_classes=Xmax-Xmin+1, dtype='float32').tolist())

X=pad_sequences(X, padding='post',value=[0]*(Xmax-Xmin+1))


nsamples, nx, ny = X.shape
X_re = X.reshape((nsamples,nx*ny))

X_train, X_test, Y_train, Y_test = train_test_split(    X_re, Y, test_size=0.1,
    shuffle=True)

# Random forest:
clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')



dump(clf,'OneHot_Ref_RandomForest.joblib')


# MLP:
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')




dump(clf,'OneHot_Ref_MLPClf.joblib')



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

trainingY=tf.keras.utils.to_categorical(trainingY, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(validateY, num_classes=None, dtype='float32')


# CNN:
model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv2D(32, kernel_size=(3, 1),activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()


model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))


predictY = model.predict(validateX,verbose = 1)


predictYround=np.around(predictY)
n=0;
for a in range(len(validateY)):
    if (predictYround[a]==validateY[a]).all():
        n+=1;
n/len(validateY)



model.save('OneHot_Ref_CNN')

################################################################################################
# augmentation with gcd(length-1)


clf=load('gcdLen-1_Ref_RandomForest.joblib')
#clf=load('gcdLen-1_Ref_MLPClf.joblib')
model=keras.models.load_model('gcdLen-1_Ref_CNN')



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive FROM plucker WHERE is_reflexive=1 ORDER BY RANDOM() LIMIT 42943", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['is_reflexive'].to_list()

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive FROM plucker WHERE is_reflexive=0 ORDER BY RANDOM() LIMIT 42943", db)
    
X0 = df['plucker'].transform(literal_eval)
X0=X0.to_list()
Y0=df['is_reflexive'].to_list()

XX=X1+X0
Y=Y1+Y0



X0=[];
for a in range (len(XX)):
    x0=appendgcd(XX[a],len(XX[a])-1)
    X0.append(x0)
        
X=X0;
X=pad_sequences(X, padding='post',value=0)



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# Random forest:
clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')



dump(clf,'gcdLen-1_Ref_RandomForest.joblib')



# MLP:
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')


dump(clf,'gcdLen-1_Ref_MLPClf.joblib')



# CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
X_train = np.array([X[a] for a in trainingindex]);
Y_train = np.array([Y[a] for a in trainingindex]);
X_test = np.array([X[a] for a in validateindex]);
Y_test = np.array([Y[a] for a in validateindex]);

trainingX = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
validateX  = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
Y_train_re = np.reshape(Y_train,(Y_train.shape[0],1))
Y_test_re = np.reshape(Y_test,(Y_test.shape[0],1))

trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')



model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()




model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))



predictY = model.predict(validateX,verbose = 1)


predictYround=np.around(predictY)
n=0;
for a in range(len(validateY)):
    if (predictYround[a]==validateY[a]).all():
        n+=1;
n/len(validateY)


model.save('gcdLen-1_Ref_CNN')


################################################################################################
# fixed length
# ex length = 10


with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive,plucker_len FROM plucker WHERE is_reflexive=1 AND plucker_len=10 ORDER BY RANDOM() LIMIT 2410", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['is_reflexive'].to_list()

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive,plucker_len FROM plucker WHERE is_reflexive=0 AND plucker_len=10 ORDER BY RANDOM() LIMIT 2410", db)
    
X0 = df['plucker'].transform(literal_eval)
X0=X0.to_list()
Y0=df['is_reflexive'].to_list()

X=X1+X0
Y=Y1+Y0



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# Random forest:
clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')


# MLP:
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')



# CNN:
trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
X_train = np.array([X[a] for a in trainingindex]);
Y_train = np.array([Y[a] for a in trainingindex]);
X_test = np.array([X[a] for a in validateindex]);
Y_test = np.array([Y[a] for a in validateindex]);

trainingX = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
validateX  = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
Y_train_re = np.reshape(Y_train,(Y_train.shape[0],1))
Y_test_re = np.reshape(Y_test,(Y_test.shape[0],1))

trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))

predictY = model.predict(validateX,verbose = 1)

predictYround=np.around(predictY)
n=0;
for a in range(len(validateY)):
    if (predictYround[a]==validateY[a]).all():
        n+=1;
n/len(validateY)



with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive,plucker_len FROM plucker WHERE is_reflexive=1 AND plucker_len=10 ORDER BY RANDOM() LIMIT 2410", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['is_reflexive'].to_list()

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive,plucker_len FROM plucker WHERE is_reflexive=0 AND plucker_len=10 ORDER BY RANDOM() LIMIT 2410", db)
    
X0 = df['plucker'].transform(literal_eval)
X0=X0.to_list()
Y0=df['is_reflexive'].to_list()

X=X1+X0
Y=Y1+Y0

Xmin=min([item for sublist in X for item in sublist])
Xmax=max([item for sublist in X for item in sublist])

X0=np.array([np.array(X[a]) for a in range (0,len(X))]);

X=[];
for a in range (0,len(X0)):
    X.append(tf.keras.utils.to_categorical(X0[a]-Xmin, num_classes=Xmax-Xmin+1, dtype='float32').tolist())



nsamples, nx, ny = X.shape
X_re = X.reshape((nsamples,nx*ny))

X_train, X_test, Y_train, Y_test = train_test_split(    X_re, Y, test_size=0.1,
    shuffle=True)

clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')




clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')


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

trainingY=tf.keras.utils.to_categorical(trainingY, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(validateY, num_classes=None, dtype='float32')

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv2D(32, kernel_size=(3, 1),activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))

predictY = model.predict(validateX,verbose = 1)

predictYround=np.around(predictY)
n=0;
for a in range(len(validateY)):
    if (predictYround[a]==validateY[a]).all():
        n+=1;
n/len(validateY)


with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive,plucker_len FROM plucker WHERE is_reflexive=1 AND plucker_len=10 ORDER BY RANDOM() LIMIT 2410", db)

X1 = df['plucker'].transform(literal_eval)
X1=X1.to_list()
Y1=df['is_reflexive'].to_list()

with sqlite3.connect("plucker.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT plucker,is_reflexive,plucker_len FROM plucker WHERE is_reflexive=0 AND plucker_len=10 ORDER BY RANDOM() LIMIT 2410", db)
    
X0 = df['plucker'].transform(literal_eval)
X0=X0.to_list()
Y0=df['is_reflexive'].to_list()

XX=X1+X0
Y=Y1+Y0

X0=[];
for a in range (len(XX)):
    x0=appendgcd(XX[a],len(XX[a])-1)
    X0.append(x0)
        
X=X0;
#X=pad_sequences(X, padding='post',value=0)



X_train, X_test, Y_train, Y_test = train_test_split(    X, Y, test_size=0.1,
    shuffle=True)

# random forest classifier. n_estimators random trees of max_depth:
clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')



clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1,hidden_layer_sizes=(100,))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(f'accuracy score: {accuracy_score(Y_pred, Y_test)}')



trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) );
validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));
X_train = np.array([X[a] for a in trainingindex]);
Y_train = np.array([Y[a] for a in trainingindex]);
X_test = np.array([X[a] for a in validateindex]);
Y_test = np.array([Y[a] for a in validateindex]);

trainingX = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
validateX  = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
Y_train_re = np.reshape(Y_train,(Y_train.shape[0],1))
Y_test_re = np.reshape(Y_test,(Y_test.shape[0],1))

trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')

model = Sequential()
input_shapeX = trainingX[0].shape
model.add(Conv1D(32, kernel_size=3,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(2,padding='same'))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(trainingX,trainingY,batch_size=16,epochs=100,verbose=1,
validation_data=(validateX,validateY))

predictY = model.predict(validateX,verbose = 1)

predictYround=np.around(predictY)
n=0;
for a in range(len(validateY)):
    if (predictYround[a]==validateY[a]).all():
        n+=1;
n/len(validateY)




