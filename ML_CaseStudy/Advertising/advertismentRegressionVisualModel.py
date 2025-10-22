import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

def Advertising(Datapath):
    line = "-"* 70

    print(line)
    print("Dataset sample is : ")
    df = pd.read_csv(Datapath)
    print(df.head())
    print(line)

    print("clean the detaset")
    table = df.drop(columns = ['Unnamed: 0'] , inplace = True)

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

    x = df[['TV','radio','newspaper']]
    y = df['sales']

    x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = metrics.r2_score(y_test, y_pred)

    print("Mean squared error : ",MSE)
    print("root mean squared error ",RMSE)
    print("r square is : ",R2)

    print("Model coefficient are : ")
    for col, coef in zip(x.columns, model.coef_):
        print(f"{col} : {coef}")

    print("Y Intercept is ",model.intercept_)   
    plt.figure(figsize=(8,5))
    plt.scatter(y_test,y_pred, color = 'blue')
    plt.xlabel("Actual sales")
    plt.ylabel("predicted sales")
    plt.title("Marvellous Advertisement")
    plt.grid(True)
    plt.show() 

    filename = 'trained_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)


def main():
    Advertising("Advertising.csv")

if __name__ == "__main__":
    main()