# Predicting Titanic survivors based on passenger data

This is my first application of machine learning techniques. The goal is to
predict whether a passenger survived the sinking of the Titanic based on his
or her passenger information. The dataset comes from Kaggle and is used as
an introduction to machine learning competitions.

The main purpose of this exercise is for me to learn about ML. For this reason,
the code and this document are organized in steps, each one adding a new
technique or optimization to improve the overall model. I'm just starting
with ML, so some of (or all) the solutions are probably naive.

As development environment I'm using Python and some of its popular libraries
for data science:

```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

## 1. Getting the data

The training data is in a CSV file easily readable with Pandas:

```
data = pd.read_csv('train.csv')
```

The describe() method gives an overview of the data:

```
data.describe()
```

| | PassengerId | Survived | Pclass | Age | SibSp | Parch | Fare |
|-|-------------|----------|--------|-----|-------|-------|------|
| count | 891.000000 | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean | 446.000000 | 0.383838 | 2.308642 | 29.699118 | 0.523008 | 0.381594 | 32.204208 |
| std | 257.353842 | 0.486592 | 0.836071 | 14.526497 | 1.102743 | 0.806057 | 49.693429 |
| min | 1.000000 | 0.000000 | 1.000000 | 0.420000 | 0.000000 | 0.000000 | 0.000000 |
| 25% | 223.500000 | 0.000000 | 2.000000 | 20.125000 | 0.000000 | 0.000000 | 7.910400 |
| 50% | 446.000000 | 0.000000 | 3.000000 | 28.000000 | 0.000000 | 0.000000 |  14.454200 |
| 75% | 668.500000 | 1.000000 | 3.000000 | 38.000000 | 1.000000 | 0.000000 |  31.000000 |
| max | 891.000000 | 1.000000 | 3.000000 | 80.000000 | 8.000000 | 6.000000 | 512.329200 |

The survival rate (the "mean" in the "Survived" column above) is 0.38, so by
just predicting that everybody dies I would be right 62% of the times.
Hopefully the algorithm will do better than this.

## 2. Initial features and logistic regression

To start, I will use age, sex, pclass and fare as features for the logistic
regression algorithm. "Sex" has values "male" and "female" that need to be
converted to numeric values.

```
data['Sex'] = data['Sex'].map({'male':1,'female':0})
```

The age for some of the passengers is missing, so I will fill it with the
average age. This is a bit crude, but I'll improve it later.

```
data['Age'].fillna(data['Age'].mean(), inplace=True)
```

For the model, I will start with a Logistic Regression, using the
implementation from Scikit. The training accuracy turns out to be 0.79.

After training the model, I generate the predictions based on the "test.csv"
file and export them to a CSV file ("step1_results.csv") that I then upload to Kaggle. The resulting score is 74.641.

## 3. Adding more features

To improve the accuracy, I add more features to the training data. SibSp, Parch
and Embarked look like they could help. Training the model with these gives me
a training accuracy of 0.81 and the Kaggle score is 0.76077. There are more
features to be added but they need a bit of processing, so I'll first try
something else. Out of curiosity, I added PassengerId to the features and the
Kaggle score lowered to 0.74, which is not surprising given that PassengerId
shouldn't have anything to do with surival rate and only causes the model to
overfit.
