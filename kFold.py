from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

iris = load_iris()
print("Iris labels: \n{}".format(iris.target)) #Prints the target labels of the dataset (0, 1, and 2 represent different flower types).
logreg = LogisticRegression() #Creates a Logistic Regression model.

kfold = KFold(n_splits=3) #splits dataset into 3 parts for cross-validation (no shuffling)
print("Cross-Validation scores: \n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold))) #Trains and tests the model 3 times and prints the accuracy scores.

kfold = KFold(n_splits=3, shuffle=True, random_state=0) #Splits the data into 3 parts after shuffling the data, random_state=0 makes the result the same every time.
print("Cross-Validation scores: \n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold))) #Again prints accuracy scores, but this time using shuffled data.

'''Iris dataset → flower data
Logistic Regression → prediction model
KFold → divides data into 3 parts
cross_val_score → checks accuracy
Shuffling → gives more reliable results'''
