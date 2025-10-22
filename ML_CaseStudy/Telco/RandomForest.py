import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Telco-Customer-Churn.csv")

    df.drop(['customerID', 'gender'], axis = 1, inplace=True)

    for col in df.select_dtypes(include = 'object'):
        df[col] = LabelEncoder().fit_transform(df[col])   

    x = df.drop('Churn', axis = 1) 
    y = df['Churn']  

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth= 7, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Accuracu is : ", accuracy_score( y_test, y_pred) * 100)

    importance = pd.Series(model.feature_importances_, index = x.columns)
    importance = importance.sort_values(ascending = False)

    importance.plot(kind = 'bar', figsize=(10,6), title = "Feature Imporatnace")
    plt.show()

if __name__ == "__main__":
    main()    