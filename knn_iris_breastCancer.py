from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

#load iris dataset
#data = load__iris()
#load breast cancer
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

for k in range(1,6):
    #Instantiate learning model (k= 1 to 5)
    classifier = KNeighborsClassifier(n_neighbors=k)

    #fitting the model
    classifier.fit(X_train, y_train)

    #predicting the test set results
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)*100
    print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')













    
