# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score

# Load Iris dataset
#data = load_iris()
#load breast cancer data
data = load_breast_cancer()
X = data.data
y = data.target

# Binarize the features (convert continuous to 0/1)
binarizer = Binarizer()
X_binary = binarizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.3, random_state=42)

# Initialize Bernoulli Naive Bayes classifier
bernoulli_nb = BernoulliNB()

# Train the classifier
bernoulli_nb.fit(X_train, y_train)

# Predict the labels for test data
y_pred = bernoulli_nb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#output-Accuracy: 0.28888888888888886 (iris dataset)
            #Accuracy: 0.631578947368421 (breast cancer data)
