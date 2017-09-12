# Predicting Titanic survivors based on passenger data

This is my first application of machine learning techniques. The goal is to
predict whether a passenger survived the sinking of the Titanic based on his
or her passenger information. The dataset comes from [Kaggle](https://www.kaggle.com/c/titanic) and is used as
an introduction to machine learning competitions.

The best Kaggle score I have achieved is **0.78947**, which is not great (but at least better than predicting that no-one survived, which gives a score of ~0.62). Note that you can cheat and look up the survival status from one of the Titanic passenger databases, which explains the very high scores in the Kaggle leaderboard. But a score of 0.82 should be achievable with a model.

There are plenty of iPython notebooks out there that walk you through the data processing steps, so I won't go too much into details (you can check the code for that). I did try to come up with solutions by myself before looking them up, and in most cases I did.

## Implementation

The script `titanic.py` processes the data, trains a few models and picks the best one to generate the prediction for the target set.

### Data Processing

The features I picked for the model are:

* **Title** - extracted from the passenger name. There are some exotic titles with only one representative, it may be a good idea to squash them in broader categories but I chose not to do it. Then again, perhaps a Count has more chances to survive than a Mr.? Of course one should analyze the data to decide that.
* **Age** - there are a lot of missing values for age, and it looks like it would be important to predict the survival. To better guess the age, I filled the missing values with the average age for a given Title. The procedure is not very robust because it assumes that there is at least one age value per title, but luckily that is the case.
* **Cabin Deck** - extracted from the cabin number. I'm not sure how useful this is because there are a majority of missing values (which are given the value 'U' - Unspecified), and I do get the same score without this feature, but it's easy to add and it may help future models.
* **Fare** - there are a few fare values missing that are imputed from the average fare per passenger class
* **Embarked** - Mainly untouched, but I filled the missing values with the most common port of embarkation.
* **Passenger class, SibSp, Parch** - untouched. I saw a lot of notebooks where SibSp and Parch were combined into a "Family size" feature, but I would expect the model to be able to do it by itself. Naive?

Numeric features are scaled, and categoric values are turned into "dummies", i.e. one column per category with value 0 or 1. This seems to be a requirement for most of the models in Sklearn.

## Model Selection

The script goes through several of the models that come with Sklearn, using default parameters. For each model, a variant with quadratic polynomial features is also tried. The training set is split into a training set (60% of the samples) and a cross-validation set, and the models are compared based on the score on the CV set.

The best model according to this comparison is the Support Vector Machine (`SVM.SVC` model in Sklearn) with polynomial features, with a training score of 0.84 and a CV score: 0.82. Since the two scores are close, there doesn't seem to be a problem of overfitting (as is for example the case with the RandomForest model).
seems to be the best fit.

## Conclusion

I'm sure my approach to this problem is simplistic, as is demonstrated by the fact that I can't increase the score beyond 0.79. I should at least try different model parameters, and do a more systematic analysis of the training data.  
