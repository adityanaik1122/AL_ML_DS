import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def MarvellousLogistic(datasetpath):
    df = pd.read_csv(datasetpath)

    print("Dimention of datafrom : ",df.shape)
    print("Indtial data is : ")
    print(df.head())

    df['Gender'] = df['Gender'].map({'Female' : 0, 'Male' : 1})
    print("Encoded data is : ")
    print(df.head())

    plt.figure(figsize = (8,6))
    sns.scatterplot(data = df, x = 'Height', y = 'Weight', hue = 'Gender', palette='Set1')
    plt.title("Marvellous Weight Height Predictor")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.show()

    X = df[['Height','Weight']]
    Y = df['Gender']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=42)

    model = LogisticRegression()

    model.fit(X_train,Y_train)

    Y_predict = model.predict(X_test)

    accuracy = accuracy_score( Y_test, Y_predict)

    print("accuracy of logistic regression is : ",accuracy * 100)

    Conf_matrix = confusion_matrix(Y_test, Y_predict)

    print("Confusion matrix of logistic regression is : ",Conf_matrix)

    report = classification_report(Y_test, Y_predict)

    print("Classification report of logistic regression is : \n",report)


def main():
    MarvellousLogistic("weight-height.csv")

if __name__ == "__main__":
    main()