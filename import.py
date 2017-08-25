import csv
import numpy as np

reader = csv.reader(open("train.csv", "r"), delimiter=",")
print(next(reader)) # skip headers

n = 5
m = 0

temp_X = []
temp_Y = []

male_avg_age = 0.
males_with_age = 0
female_avg_age = 0.
females_with_age = 0

for row in reader:

    pclass = float(row[2])

    if row[4] == 'male':
        psex = 0.
    else:
        psex = 1.

    if row[5] != '':
        age = float(row[5])
        if psex == 0.0:
            males_with_age += 1
            male_avg_age += age
        else:
            females_with_age += 1
            female_avg_age += age

    else:
        age = -1.0 - psex

    fare = float(row[9])

    if row[11] == 'C':
        port = 1.
    elif row[11] == 'Q':
        port = 2.
    else:
        port = 3

    temp_X.append([pclass, psex, age, fare, port])
    temp_Y.append(row[1])
    m = m + 1

male_avg_age /= males_with_age
female_avg_age /= females_with_age

X = np.zeros((m,n))
Y = np.zeros((m,1))

for i in range(m):
    X[i] = temp_X[i]
    if X[i][2] == -1.0:
        X[i][2] = male_avg_age
    elif X[i][2] == -2.0:
        X[i][2] = female_avg_age
    Y[i] = temp_Y[i]

print("Male avg age: {}".format(male_avg_age))
print("Female avg age: {}".format(female_avg_age))

#print(temp_X)
print(X)
#print(Y)
