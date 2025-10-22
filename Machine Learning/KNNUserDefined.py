#   A B C D
# X:1,2,3,6
# Y:2,3,1,5

import numpy as mp
import math

def EucDistance(p1,p2):
    Ans = math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)
    return Ans

def MarvellousKNN():
    line = "-"*80
    data = [{'point':'A','x': 1,'y': 2,'label':'Red'},
            {'point':'B','x': 2,'y': 3,'label':'Red'},
            {'point':'C','x': 3,'y': 1,'label':'Blue'},
            {'point':'D','x': 6,'y': 5,'label':'Blue'}]
    print(line)
    print("Training data set :")
    print(line)

    for i in data :
        print(i)
    print(line)

    new_point = {'x' : 3, 'y' : 3}

    #calculate the distance
    for d in data:
        d['distance'] = EucDistance(d, new_point)
    
    print(line)
    print("Calculated distance are : ")
    print(line)

    for d in data:
        print(d)
    print(line)    

    #sort by distance
    sorted_data = sorted(data,key = lambda item : item['distance'])
    print(line)
    print("Sorted data are : ")
    print(line)
    for d in sorted_data:
        print(d)
    print(line)    

    k = 3
    nearest = sorted_data[:k]
    print(line)  
    print("sorted 3 elements are :")
    print(line) 
    for d in nearest:
        print(d)
    print(line)  

    #voting
    votes = {}
    for neighbour in nearest:
        label = neighbour['label']
        votes[label] = votes.get(label,0) + 1

    print(line)
    print("result of voting is :")
    for d in votes:
        print("name :",d, "Value :", votes[d])
    print(line)

    predicted_class = max(votes, key = votes.get) # type: ignore

    print("Predicted class for poinnt(3,3) is : ",predicted_class)
    


def main():
    print("Demonstration of KNN Algorithm")
    MarvellousKNN()
if __name__ == "__main__":
    main()    