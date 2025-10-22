import pandas as pd
import numpy as np

from matplotlib.pyplot import figure,show
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    #show()

    figure()
    target = "Survived"
    countplot(data = df, x=target, hue = 'Sex').set_title("Based on gender")
    #show()

    figure()
    target = "Survived"
    countplot(data = df, x=target, hue = 'Pclass').set_title("Based on gender")
    #show()

    figure()
    df['Age'].plot.hist().set_title("Age report")
    #show()

    figure()
    df['Fare'].plot.hist().set_title("Fair report")
    #show()

    plt.figure(figsize = (10,6))
    sns.heatmap(df.corr(), annot= True, cmap= 'coolwarm')
    plt.title("Feature corelation ")
    #plt.show()

    x = df.drop(columns=['Survived'])
    y = df['Survived']
    print("Dimentions of target : ",x.shape)
    print("Dimentions of labels : ",y.shape)

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scale,y,test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy is : ",accuracy)
    print("Confusion matrix : ")
    print(cm)

def main():
    MarvellousTitanicLogistc("MarvellousTitanicDataset.csv")


if __name__ == "__main__":
    main()