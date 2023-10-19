import random

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lars
import numpy as np

def impute(dataset):
    imputer = SimpleImputer(strategy='median')
    imputer.fit(dataset)
    return imputer.transform(dataset)

df = pd.read_csv("winequalityN.csv")

predictors = df.columns[1:-1]
outcome = ['quality']

X = df[predictors]
Y = df['quality']

X = impute(X)
normalize(X, copy=False)

print("Both types:")
for i in range(10):
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, shuffle=True, test_size=0.3)
    clf = Lars()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    print(f"score = {clf.score(test_X, test_Y)}")
    #print(f"mse = {np.sqrt(mean_squared_error(test_Y, predictions))}")

red = df.loc[df['type'] == 'red']
red_X = red[predictors]
red_Y = red['quality']

red_X = impute(red_X)
normalize(red_X, copy=False)

print("Red:")
for i in range(10):
    train_X, test_X, train_Y, test_Y = train_test_split(red_X, red_Y, shuffle=True, test_size=0.3)
    clf = Lars()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    print(f"score = {clf.score(test_X, test_Y)}")
    #print(f"mse = {np.sqrt(mean_squared_error(test_Y, predictions))}")

white = df.loc[df['type'] == 'white']
white_X = white[predictors]
white_Y = white['quality']

white_X = impute(white_X)
normalize(white_X, copy=False)

print("White:")
for i in range(10):
    train_X, test_X, train_Y, test_Y = train_test_split(white_X, white_Y, shuffle=True, test_size=0.3)
    clf = Lars()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    print(f"score = {clf.score(test_X, test_Y)}")
    #print(f"mse = {np.sqrt(mean_squared_error(test_Y, predictions))}")

print("Final test:")
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, shuffle=True, test_size=0.3)
clf = Lars()
clf.fit(train_X, train_Y)
predictions = clf.predict(test_X)
print(f"score = {clf.score(test_X, test_Y)}")
#print(f"mse = {np.sqrt(mean_squared_error(test_Y, predictions))}")
for i in range(10):
    index = random.randint(0, X.shape[0])
    sample_X = X[index]
    sample_Y = Y[index]
    prediction = clf.predict([sample_X])
    print(f"Prediction: {round(prediction[0], 5)}, actual value: {round(sample_Y, 5)}")