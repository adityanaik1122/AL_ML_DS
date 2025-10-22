from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
                                                                    

def MarvellousCalculateAccuracyKNN():
    iris = load_iris()
    data = iris.data
    target = iris.target

    X_train, X_test, Y_train,  Y_test = train_test_split(data,target,test_size=0.5,random_state=45)

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train,Y_train)

    preditions = model.predict(X_test)

    Accuracy = accuracy_score(preditions, Y_test)

    print("Accuracy of KNN classifier wuth k = 5 is : ",Accuracy*100)

    model = KNeighborsClassifier(n_neighbors=7)

    model.fit(X_train,Y_train)

    preditions = model.predict(X_test)

    Accuracy = accuracy_score(preditions, Y_test)

    print("Accuracy of KNN classifier wuth k = 7 is : ",Accuracy*100)

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train,Y_train)

    preditions = model.predict(X_test)

    Accuracy = accuracy_score(preditions, Y_test)

    print("Accuracy of KNN classifier wuth k = 3 is : ",Accuracy*100)

def main():
    MarvellousCalculateAccuracyKNN()
   


if __name__ == "__main__":
    main()    