import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()   #Loads the breast cancer dataset.
X = data.data #Assigns the feature matrix (input variables) to X.
y = data.target #Assigns the target labels to y. Binary classification: 0 = malignant 1 = benign


#Splits the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#test_size=0.2 - 20% of data is used for testing. 80% used for training.
#random_state=42 - Ensures reproducible results.
#stratify=y - Keeps class proportions the same in train and test sets. Important for imbalanced datasets.

model = LogisticRegression(max_iter=5000) #Creates a Logistic Regression model. Allows up to 5000 iterations for convergence.
model.fit(X_train, y_train) #Trains the model using training data. The model learns relationships between features and labels.

#Predictions and confusion matrix
y_pred = model.predict(X_test)  #Uses the trained model to predict labels for test data. Outputs predicted class labels (0 or 1).
cm = confusion_matrix(y_test, y_pred) #Computes confusion matrix comparing: True labels (y_test) & Predicted labels (y_pred)

print("Confusion Matrix: \n", cm)

tn, fp, fn, tp = cm.ravel()
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

#Another way to extract tn, tp, fn, fp

print("Another way--")
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

print("\n Extracted Values: ")
print("TP :", TP)
print("TN :", TN)
print("FP :", FP)
print("FN :", FN)


















