import csv
import numpy as np
from sklearn.linear_model import LogisticRegression

def import_data( csvfile, hasY ):
    reader = csv.reader(open(csvfile, "r"), delimiter=",")
    print(next(reader)) # skip headers

    n = 4
    m = 0

    temp_X = []
    temp_Y = []

    if hasY:
        classCol = 2
    else:
        classCol = 1

    sexCol = classCol + 2
    ageCol = classCol + 3
    fareCol = classCol + 7
    portCol = classCol + 9

    male_avg_age = 0.
    males_with_age = 0
    female_avg_age = 0.
    females_with_age = 0

    for row in reader:

        # print(row)

        pclass = float(row[classCol])

        if row[sexCol] == 'male':
            psex = 0.
        else:
            psex = 1.

        if row[ageCol] != '':
            age = float(row[ageCol])
            if psex == 0.0:
                males_with_age += 1
                male_avg_age += age
            else:
                females_with_age += 1
                female_avg_age += age
        else:
            age = -1.0 - psex

        # fare = float(row[fareCol])

        if row[portCol] == 'C':
            port = 1.
        elif row[portCol] == 'Q':
            port = 2.
        else:
            port = 3

        # temp_X.append([pclass, psex, age, fare, port])
        temp_X.append([pclass, psex, age, port])
        if hasY:
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
        if hasY:
            Y[i] = temp_Y[i]

    print("Male avg age: {}".format(male_avg_age))
    print("Female avg age: {}".format(female_avg_age))

    Y = np.ravel(Y)

    return X,Y

X,Y = import_data("train.csv", hasY = True)

model = LogisticRegression()
model = model.fit(X, Y)
print(model.score(X, Y))

X_test, Y_test = import_data("test.csv", hasY = False)

pred = model.predict(X_test)

for p in pred:
    print(int(p))
#print(pred)
#print(temp_X)
#print(X)
#print(Y)
