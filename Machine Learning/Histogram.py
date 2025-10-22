import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("iris.csv")

    plt.hist(df['sepal.length'],bins=10, color = "skyblue", edgecolor = "black")

    plt.xlabel("sepal Length")
    plt.ylabel("frequency")
    plt.title("Marvellous histogram for iris")
    
    plt.show()

if __name__ == "__main__":
    main()    