from sklearn.metrics import confusion_matrix

def main():
    Actual = [1,0,1,1,0,1,0,1,0,1]
    predicted = [1,0,1,0,0,1,1,1,1,1]

    Con_mat = confusion_matrix(Actual,predicted)

    print("Confusion matrix is : ")
    print(Con_mat)

    # Accuracy = (TN + TP) / (TN + TP + FN + FP)
    # TN -2 FP-2
    # 

if __name__ == "__main__":
    main()    