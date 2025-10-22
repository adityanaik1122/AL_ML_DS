import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def MarvellousHeadBrainLinear(Datapath):
    Line = "-"*100
    df = pd.read_csv(Datapath)
    print(Line)
    print("First few records of datasets are : ")
    print(Line)
    print(df.head())

    print(Line)
    print("describe dataset are : ")
    print(df.describe())
    print(Line)

    x = df[['Head Size(cm^3)']]
    y = df[['Brain Weight(grams)']]
    print("Indipendent variables are : HeadSize")
    print("Dependent variables are : BrainWeight")

    print("Total records in dataset : ", x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    print("Dimentions of Training dataset : ",x_train.shape)
    print("Dimentions of Testing dataset : ",x_test.shape)

    model = LinearRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    MSE = mean_squared_error(y_pred,y_test)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)


    print("Visual representation : ")
    plt.figure(figsize=(8,5))
    plt.scatter(x_test, y_test, color = 'blue', label = 'Actual')
    plt.plot(x_test.values. flatten(), y_pred, color = 'red', linewidth = 2, label = "Regrassion")
    plt.xlabel('HeadSize')
    plt.ylabel('Brainweight')
    plt.title('marvellous head brain regression')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(Line)
    print("Result of case study : ")
    print("mean square error is : ",MSE)
    print("root mean square erroe is : ",RMSE)
    print("R square value is : ",R2)
    print("Slope of line(m) : ",model.coef_[0])
    print("Y intercept (c) : ",model.intercept_)


def main():
    MarvellousHeadBrainLinear("MarvellousHeadBrain.csv")

if __name__=="__main__":
    main()        