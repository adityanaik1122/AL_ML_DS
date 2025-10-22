from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


def main():
    print("Code to demonstrate the concept of WCSS in KMEANS")

    dataset = pd.read_csv("iris.csv")

    X = dataset.iloc[:,[0,1,2,3]].values
    print(X.shape)


    model = KMeans(n_clusters = 3, init='k-means++', n_init = 10, random_state=42)
    y_kmeans = model.fit_predict(X)
    
    print("Values of y_kmeans")
    print(y_kmeans.shape)

    print("Cluster of setosa : 0")
    for i in range(0,10):
        print(X[i], y_kmeans[i])

    print("Cluster of versicolor : 1")
    for i in range(51,61):
        print(X[i], y_kmeans[i])

    print("Cluster of setosa : 2")
    for i in range(101,111):
        print(X[i], y_kmeans[i])        

if __name__ == "__main__":
    main()    