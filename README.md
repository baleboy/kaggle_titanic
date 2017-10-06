# Predicting Titanic survivors using machine learning

A machine learning web application that uses historical passenger data to predict
whether someone would have survived the sinking of the Titanic.

This started as my attempt to solve the Kaggle challenge ["Titanic: machine
learning from disaster"](https://www.kaggle.com/c/titanic), but it ended up
becoming an experiment to create an end-to-end machine learning application.

You can try the finished app [here](https://titanic-kaggle.herokuapp.com/).

## How it Works

The goal of the application is to predict whether a hypothetical passenger
(possibly based on the user's personal details) would have survived the sinking
of the Titanic. It is composed of a training module that learns the relation
between various passenger details and their survival outcome, and a web application
that uses the trained model to make a prediction on new passenger information.

All the modules (well, except the web page) are written in Python, and the web
application runs on Flask.

## Training

The script `train.py` processes the data, trains a few models and picks the
best one to generate the prediction for the target set. The model and other
helper functions needed to infer the data are saved in a pickle file.
The script `predict.py` loads the trained model and helper functions and uses
them to predict the results on a given dataset. This is mainly used to generate
the Kaggle submission (the best score I could achieve was **0.78947**).

There are plenty of resources on how to organize a machine learning project for
this particular problem, so I won't go into too many details. I did try to figure
things out on my own as much as possible. Below is a summary of the main techniques
I used.

### Data Processing

The features I picked for the model are:

* **Title** - extracted from the passenger name. Interestingly, the model seems
to be a lot more sensitive to title than gender.
* **Age** -  I filled the missing values with the average age for a given Title.
The procedure is not very robust because it assumes that there is at least one
age value per title, but luckily that is the case.
* **Cabin Deck** - extracted from the cabin number. Missing cabin numbers are given
the value 'U' - Unspecified.
* **Fare** - Missing fare values are imputed from the average fare per passenger class
* **Embarked** - I filled the missing values with the most common port of embarkation
('C' - Cherbourg).
* **Passenger class, SibSp, Parch** - untouched. I saw a lot of notebooks where
SibSp and Parch were combined into a "Family size" feature, but I decided that
the model should be able to do it by itself.

Numeric features are scaled, and categoric values are turned into "dummies", i.e.
one column per category with value 0 or 1.

The mean and variance of the data is calculated only on the training set, to avoid making any
sort of assumption on the test set (which should anyway follow the same distribution).
This is why the trained scalers are saved to disk during training.

## Model Selection

The script goes through several of the models that come with Sklearn, using default
parameters. The training set is split into a training set (60% of the samples)
and a cross-validation set, and the models are compared based on the score on the CV set.

The best model according to this comparison is the Support Vector Machine
(`SVM.SVC` model in Sklearn), with a training score of 0.84 and a CV score: 0.82.
 Other models (e.g. RandomForest) achieve a better score on the training set, but
 do worse on the test set because of overfitting.

## Web application

The file `app.py` is a Flask application that implementes a REST api
to interrogate the model, and a static web page that makes requests to the server.
It can be run locally in Python, in a Docker container via the included Dockerfile,
or deployed to Heroku. Some of the parameters (e.g. fare, cabin number) are
hard-coded as it would be difficult/not useful for a user to guess them,
but a better solution would be to pick them randomly among the values in the training
set.
