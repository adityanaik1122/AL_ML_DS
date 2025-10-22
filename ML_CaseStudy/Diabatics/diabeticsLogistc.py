import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    diabetes = pd.read_csv("diabetes.csv")

    print(diabetes.columns)

    print(diabetes.head())

    print(diabetes.shape)

    x = diabetes.drop(columns = ['Outcome'])
    y = diabetes['Outcome']

    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
 
    logreg  = LogisticRegression().fit(x_train,y_train)

    print("Training accuracy : ")
    print(logreg.score(x_train,y_train))

    print("Testing accuracy : ")
    print(logreg.score(x_test,y_test))

if __name__ == "__main__":
    main()    