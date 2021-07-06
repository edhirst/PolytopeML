'''
Machine learning dimension from Hilbert series using a support vector machine
Author: Johannes Hofscheier (johannes.hofscheier@nottingham.ac.uk)
Date: 2020-08-15
'''
import sqlite3
from ast import literal_eval
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error

with sqlite3.connect('data.db') as db:
    c = db.cursor()
    df = pandas.read_sql_query('SELECT hilb, dim FROM data', db)

# cast 'hilb'-column from a column of strings into a column of lists of ints
df['hilb'] = df['hilb'].transform(literal_eval)

# normalisation: divide entries by last entry and remove last entry
df['hilb'] = df['hilb'].transform(lambda h: [h[i] / h[-1]
                                             for i in range(0, len(h) - 1)])

hilb_train, hilb_test, dim_train, dim_test = train_test_split(\
    df['hilb'].to_list(), df['dim'].to_list(), test_size=0.25,
    shuffle=True)

# support vector machine. We use a classifier.
clf_dim = svm.SVC(kernel='linear', gamma='scale')
clf_dim.fit(hilb_train, dim_train)

# predictions and metrics:
dim_pred = clf_dim.predict(hilb_test)
print(f'mean squared error: {mean_squared_error(dim_test, dim_pred)}')
print(f'accuracy score: {accuracy_score(dim_test, dim_pred)}')
