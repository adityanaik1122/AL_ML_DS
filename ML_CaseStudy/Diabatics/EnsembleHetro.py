import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


def main():
    diabetes = pd.read_csv("diabetes.csv")

    x = diabetes.drop(columns = ['Outcome'])
    y = diabetes['Outcome']

    print(x.shape)
    print(y.shape)

    scaler = StandardScaler()
    x_Scaled = scaler.fit_transform(x)


    x_train, x_test, y_train, y_test = train_test_split(x_Scaled,y,test_size=0.2, random_state=42)
 
    log_clf = LogisticRegression() 
    dt_clf =  DecisionTreeClassifier(max_depth=5)
    knn_clf =  KNeighborsClassifier(n_neighbors=7)

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', log_clf),
            ('dt', dt_clf),
            ('knn', knn_clf)
        ],
        voting= 'soft'
        )

    voting_clf.fit(x_train,y_train)

    y_pred = voting_clf.predict(x_test)

    print(accuracy_score(y_test, y_pred) * 100)
    print()

if __name__ == "__main__":
    main()    