import pandas as pd
import numpy as np

from matplotlib.pyplot import figure,show
import seaborn as sns
from seaborn import countplot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix


def MarvellousTitanicLogistc(Datapath):
    df = pd.read_csv(Datapath)

    print("Dataset laoded succesfully.")
    print(df.head())

    print("Dimention of dataset is : ",df.shape)

    df.drop(columns = ['Passengerid', 'zero'], inplace= True)
    print("Dimention of dataset is : ",df.shape)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
    print("Dimention of dataset is : ",df.shape)

    figure()
    target = "Survived"
    countplot(data = df, x = target).set_title("survived vs Non Survided")
    show()

    figure()
    target = "Survived"
    countplot(data = df, x=target, hue = 'Sex').set_title("Based on gender")
    show()

    figure()
    target = "Survived"
    countplot(data = df, x=target, hue = 'Pclass').set_title("Based on gender")
    show()

    figure()
    df['Age'].plot.hist().set_title("Age report")
    show()

    figure()
    df['Fare'].plot.hist().set_title("Fair report")
    show()

def main():
    MarvellousTitanicLogistc("MarvellousTitanicDataset.csv")


if __name__ == "__main__":
    main()