'''
Machine learning degree from Hilbert series using ML for dimension and then
linear regression for degree
Author: Johannes Hofscheier (johannes.hofscheier@nottingham.ac.uk)
Date: 2020-08-15
'''
import sqlite3
from ast import literal_eval
import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error


with sqlite3.connect('data.db') as db:
    c = db.cursor()
    df = pandas.read_sql_query('SELECT hilb, dim, deg FROM data', db)

# cast 'hilb'-column from a column of strings into a column of lists of ints
df['hilb'] = df['hilb'].transform(literal_eval)

hilb_train, hilb_test, dim_train, dim_test, deg_train, deg_test =\
    train_test_split(df['hilb'].to_list(), df['dim'].to_list(),
                     df['deg'].to_list(), test_size=0.25, shuffle=True)

# support vector machine. We use a classifier.
clf_dim = svm.SVC(kernel='linear', gamma='scale')
# normalise hilbert series: divide entries by last entry and remove last entry
clf_dim.fit([[h[i] / h[-1] for i in range(0, len(h) - 1)]
             for h in hilb_train], dim_train)

# use linear regression to get the linear 'degree-functions':
sort_by_dim = np.argsort(dim_train, axis=0)
hilb_train = np.array(hilb_train)[sort_by_dim]
dim_train = np.array(dim_train)[sort_by_dim]
deg_train = np.array(deg_train)[sort_by_dim]

indices_where_dim_inc = []
for i in range(1, len(hilb_train)):
    if dim_train[i] > dim_train[i-1]:
        indices_where_dim_inc.append(i)

hilb_train = np.split(hilb_train, indices_where_dim_inc)
deg_train = np.split(deg_train, indices_where_dim_inc)

regs = [LinearRegression(fit_intercept=False).fit(
    hilb_train[i][:, :i+2], deg_train[i]) for i in range(len(hilb_train))]

for i, r in enumerate(regs):
    print(f'dimension={i}: {r.coef_}')

# need to normalise hilbert series
dim_pred = clf_dim.predict(
    [[h[i] / h[-1] for i in range(0, len(h) - 1)] for h in hilb_test])
deg_pred = [round(regs[dim_pred[i] - 1].predict(\
    [hilb_test[i][:dim_pred[i]+1]])[0], 0) for i in range(len(hilb_test))]
# predictions and metrics:
print(f'mean squared error: {mean_squared_error(deg_test, deg_pred)}')
print(f'accuracy score: {accuracy_score(deg_test, deg_pred)}')
