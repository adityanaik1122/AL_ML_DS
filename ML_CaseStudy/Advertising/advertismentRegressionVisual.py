import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def Advertising(Datapath):
    line = "-"* 70

    print(line)
    print("Dataset sample is : ")
    df = pd.read_csv(Datapath)
    print(df.head())
    print(line)

    print("clean the detaset")
    df.drop(columns = ['Unnamed: 0'] , inplace = True)

    print("Updated Dataset is : ")
    print(df.head())
    print(line)

    print("Missing values in each column : ",df.isnull().sum())
    print(line)

    print("statistical summary : ", df.describe())
    
    print("Co-relation matrix : ", df.corr)

    plt.figure(figsize=(10,5) )
    sns.heatmap(df.corr(), annot= True, cmap = 'coolwarm')
    plt.title("Marvellous heatpam")
    plt.show()

    sns.pairplot(df)
    plt.suptitle("pairplot of features",y = 1.02)
    plt.show()


def main():
    Advertising("Advertising.csv")

if __name__ == "__main__":
    main()