import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def LoadData(file_path):
    df = pd.read_csv(file_path)
    print("dataset gets loaded in memory successfully")
    return df


def GetInformation(df):
    print("information about the loaded dataset is :")
    print("shape of datset : ",df.shape)
    print("columns : ",df.columns)
    print("Missing values : ",df.isnull().sum())

def encodedata(df):
    df["variety"] =  df["variety"].map({"Setosa" : 0, "Versicolor" : 1, "Virginica" : 2})    
    return df

def split_feture_target(df):
    X = df.drop('variety', axis=1)
    Y = df["variety"]
    return X,Y  

def split(X,Y,size = 0.2):
    return train_test_split(X,Y,test_size=size)



def main():
    data = LoadData("iris.csv")
    print(data.head())

    GetInformation(data)

    print("Data after encoding")
    data = encodedata(data)
    print(data.head())

    print("splititng feture and labels")
    Independent, Dependent = split_feture_target(data)
    print(Independent.head)
    print(Dependent.head)

    X_train, X_test, Y_train, Y_test = split(Independent, Dependent, 0.3)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    
if __name__ == "__main__":
    main()    