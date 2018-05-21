import numpy as np
import pandas as pd
import csv
import random
import math
import operator


trainingSet = []
testSet = []


def letters_to_digit_convert(letters):
    class_numbers=[]
    for i in range(65,91):
        temp=chr(i) #temp will take values from A to Z i.e ASCI(65)=A, ASCI(66)=B....ASCI(91)=Z
        for j in range(0,len(letters)):# Run the loop with number of loops eequals the length of total letters
            element=letters[j]    
            if(element==temp): 
                char_num=i-64 
                class_numbers.append(char_num)    
    return class_numbers

def pick_data(class_numbers):
    from numpy import genfromtxt
    my_data = genfromtxt('C:\\Datamining project1\\HandWrittenLetters.csv', delimiter=',')
    a = np.transpose(my_data)
    b = np.int_(a)
    data = []
    for i in range(len(b)):
        for j in class_numbers:
            if b[i][0] == j:            
                data.append(b[i])            
        np.savetxt('picked_data1.csv', data, fmt='%d', delimiter=",")

train_data=[]
test_data=[]
TestX = []
final=[]
TrainX1=[]
TrainY1=[]
TestX1=[]
TestY1=[]

def test_train_data(training_instances,test_instances):
    
    from numpy import genfromtxt
    number_of_classes = 39
    data = genfromtxt('picked_data1.csv', delimiter=',')
    data = np.int_(data)
    array = np.roll(data,-1,-1)
    n=len(class_numbers)   
    for x in range(0,n):            
            for y in range(0,training_instances):
                temp=y+(x*number_of_classes)
                train_data.append(array[temp])
            for z in range(training_instances,training_instances+test_instances):
                temp2=z+(x*number_of_classes)
                test_data.append(array[temp2])
                
    np.savetxt('train_data.csv', train_data, fmt='%d', delimiter=",")
    np.savetxt('test_data.csv', test_data, fmt='%d', delimiter=",")
    for i in range (0,len(train_data)):
        TrainX1.append(train_data[i][0:320])# attribute vectors
        TrainY1.append(train_data[i][320])#labels
    for i in range (0,len(test_data)):
        TestX1.append(test_data[i][0:320])
        TestY1.append(test_data[i][320])
            
    np.savetxt('TrainX.csv', TrainX1, fmt='%d', delimiter=",") 
    np.savetxt('TrainY.csv', TrainY1, fmt='%d', delimiter=",")
    np.savetxt('TestX.csv', TestX1, fmt='%d', delimiter=",")
    np.savetxt('TestY.csv', TestY1, fmt='%d', delimiter=",")



def split_data_for_test_train(train_instances, test_instances):
    total = train_instances + test_instances
    split = (test_instances / total)
    #print(split)
    data = pd.read_csv('picked_data1.csv', header=None).values
    x = data[:,0]
    y = data[:,1:]
    import sklearn.model_selection
    TrainX, TestX, TrainY, TestY = sklearn.model_selection.train_test_split(x, y, test_size=split, random_state=0)
    return TrainX, TestX, TrainY, TestY

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(test_data, predictions):
	correct = 0
	for x in range(len(test_data)):
		if test_data[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(test_data))) * 100.0
 
    
predictions=[]

def main():
    k = 3
    for x in range(len(test_data)):
        neighbors = getNeighbors(train_data, test_data[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(test_data, predictions)
    print(predictions)
    #print ("Knn method's predicted classes")
    #print (predictions)
    return accuracy/100

abc = []
centroid = []  
support = []
Linear = []

def predictor(TrainY,TrainX,TestY,TestX,accuracy):    
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import svm
    cen=NearestCentroid()    
    SVM=svm.SVC()
    cen.fit(TrainY,TrainX)
    SVM.fit(TrainY,TrainX)
    from sklearn import linear_model
    from sklearn.metrics import r2_score    
    regr = linear_model.LinearRegression()
    regr.fit(TrainY, TrainX)
    val = []
    val = regr.predict(TestY) 
    print("LR Labels")
    print (val)
    abc.append(accu)
    centroid.append(cen.predict(TestY))
    support.append(SVM.predict(TestY))
    Linear.append(r2_score(TestX, val))
    #print("LR method's predicted labels")
    return TestX


################################################################################    
num = int(input("Enter the number of classes that you want to consider: "))
letters = []
print("Enter the elements in order \"Write in Uppercase only\": ")
for i in range(0,num):
    letters.append(input()) 
class_numbers=letters_to_digit_convert(letters) 
pick_data(class_numbers)    
temporary = [] 
acc = []
total = 0
count = int(input("Enter the number of times you want to repeat the split: ")) 
choice=int(input("Do you want to perform prediction press 0: "))
temporary=[] 
ktotal = []
ctotal = []
stotal = []
ltotal = []
array = []
accu = []
cent = sup = lin = knnv = 0
for i in range(count): 
    test_instances=int(input("Enter the number of testing elements: "))
    training_instances=39-test_instances  
    print ("The number of training elements is: ",training_instances)
    TrainX, TestX, TrainY, TestY = split_data_for_test_train(test_instances,training_instances)
    test_train_data(test_instances,training_instances)
    if(choice==0):
        accu = main() 
        TestX = predictor(TrainY,TrainX,TestY,TestX,accu)
        array.append(TestX)
'''   
from sklearn.metrics import accuracy_score
if(choice ==0):
    print("KNN Accuracy")
    for o in range(len(abc)):
        print(abc[i])
        ktotal.append(abc[o])
    print("Knn method's predicted accuracy")
    print(predictions)
    print ("Centroid Accuracy")
    for t in range(len(centroid)): 
        #print (len(centroid), len(array))
        print(accuracy_score(centroid[t], array[t]))
        ctotal.append(accuracy_score(centroid[t], array[t]))
    print("centroid method's predicted labels")
    print (centroid)
    print ("SVM Accuracy")
    for s in range(len(support)):  
        #print (len(support), len(array))
        print(accuracy_score(support[t], array[t]))
        stotal.append(accuracy_score(support[t], array[t]))
    print("SVM method's predicted labels")
    print(support)

    print ("LR Accuracy")
    for l in range(len(Linear)):        
        print (Linear[l])
        ltotal.append(Linear[l])
    print("Actual Labels")
    print(array)
    for m in range(len(ktotal)):
        knnv += ktotal[m]
    print ("Average KNN Accuracy")
    print (knnv/len(ktotal))
    for i in range(len(ctotal)):        
        cent += ctotal[i]
    print ("Average Centroid Accuracy")
    print (cent/len(ctotal))
    for j in range(len(stotal)):        
        sup += stotal[j]
    print ("Average SVM Accuracy")
    print (sup/len(stotal))
    for k in range(len(ltotal)):        
        lin += ltotal[k]
    print ("Average LR Accuracy")
    print(lin/len(ltotal))
    import matplotlib.pyplot as plt
    val1 = 0
    plt.plot(np.zeros_like(ctotal) + val1, ctotal, 'x')
    plt.text(0.0, 0.975, r'')
    plt.xlabel("Centroid Accuracy")
    plt.show()

  '''