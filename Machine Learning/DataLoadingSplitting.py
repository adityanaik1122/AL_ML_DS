import pandas as pd

def main():
    df = pd.read_csv("iris.csv")
    print("Dataset Loaded Successfully")
    print("Dimetion of dataset is : ",df.shape)

    #print(df["variety"])
    #print(df["sepal.length"].head())

    df["variety"] = df["variety"].map({'Setosa' : 0, 'Versicolor' : 1, 'Virginica' : 2})
    print(df["variety"])

    X=df.drop("variety",axis='columns')
    Y= df["variety"]

    print("Independent variable dimention : ",X)
    print("Dependent variable dimention : ",Y)

if __name__ == "__main__":
    main()    