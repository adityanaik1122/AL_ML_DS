#import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    
    y_actual =    [10,20,30,40,50]
    y_predicted = [12,18,32,38,52]

    print("Actual values are : ",y_actual)
    print("Predicted values are : ",y_predicted)

    MSE = mean_squared_error(y_actual, y_predicted)
    print("Mean squared error : ",MSE)

    RMSE = np.sqrt(MSE)
    print("Predicted values are : ",RMSE)

if __name__=="__main__":
    main()        