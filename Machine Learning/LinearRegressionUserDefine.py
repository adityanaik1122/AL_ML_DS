import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousPredictor():
    #load the data

    X = [1,2,3,4,5]
    Y= [3,4,2,4,5]

    print("Values of Independent variables :",X)
    print("Values of Dependent variables :",Y)

    mean_x = 0
    mean_y = 0

    XSum = 0
    YSum = 0

    for i in range(len(X)):
        XSum = XSum + X[i]
        YSum = YSum + Y[i]

    mean_x = XSum / len(X)
    mean_y = YSum / len(Y)    

    print("X_Mean is :", +mean_x)
    print("Y_Mean is :", +mean_y)

def main():
    MarvellousPredictor()
    
if __name__ == "__main__":
    main()