# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:42:02 2018

@author: heera
"""

import numpy as np
import pandas as pd
import csv
import random
import math
import operator


trainingSet = []
testSet = []


def pick_data(classes):
    from numpy import genfromtxt
    my_data = genfromtxt('C:\\Datamining project1\\ATNTFaceImages400.csv', delimiter=',')
    a = np.transpose(my_data)
    b = np.int_(a)
    data = []
    for i in range(len(b)):
        for j in classes:
            if b[i][0] == j:
                data.append(b[i])
        np.savetxt('picked_data.csv', data, fmt='%d', delimiter=",")

train_data=[]
test_data=[]
TrainX=[]
TrainY=[]
TestX=[]
TestY=[]
final=[]

def test_train_data(training_instances,test_instances):
    
    from numpy import genfromtxt
    number_of_classes = 10
    data = genfromtxt('picked_data.csv', delimiter=',')
    data = np.int_(data)
    array = np.roll(data,-1,-1)
    #np.savetxt('test_and_train.csv', array, fmt='%d', delimiter=",")
    n=len(classes)   
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
            TrainX.append(train_data[i][0:644])
            TrainY.append(train_data[i][644])
    for i in range (0,len(test_data)):
            TestX.append(test_data[i][0:644])
            TestY.append(test_data[i][644])
            
    np.savetxt('TrainX.csv', TrainX, fmt='%d', delimiter=",") 
    np.savetxt('TrainY.csv', TrainY, fmt='%d', delimiter=",")
    np.savetxt('TestX.csv', TestX, fmt='%d', delimiter=",")
    np.savetxt('TestY.csv', TestY, fmt='%d', delimiter=",")       
            

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
    return accuracy



def predictor(TrainX,TrainY,TestX,TestY,accuracy):    
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import svm
    cen=NearestCentroid()    
    SVM=svm.SVC()
    cen.fit(TrainX,TrainY)
    SVM.fit(TrainX,TrainY)
    from sklearn import linear_model
    from sklearn.metrics import r2_score    
    regr = linear_model.LinearRegression()
    regr.fit(TrainX, TrainY)
    val = []
    val = regr.predict(TestX)
    abc=[]    
    abc.append(predictions)
    abc.append(cen.predict(TestX))
    abc.append(SVM.predict(TestX))
    abc.append(r2_score(TestY, val))
    #print (abc)
    return abc


def validator(): 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import svm
    from sklearn import linear_model
    from sklearn.cross_validation import cross_val_score
    
    KNN = KNeighborsClassifier()
    cen=NearestCentroid()
    SVM=svm.SVC()
    regr = linear_model.LinearRegression()
    attribute=TrainX
    classlabels=TrainY
    average=[]
    #cv=[2,3,5]
    cv = [5]
    k=[]
    c=[]
    s=[]
    l=[]
    for lp in range(0,len(cv)) :
        
        score=cross_val_score(KNN,attribute,classlabels,cv=cv[lp],scoring="accuracy")
        print ('KNN accuracy')
        print (score)
        avg=0
        for ty in range(0,len(score)):
            avg=avg+score[ty]
        avg=avg/len(score)
       
        score1=cross_val_score(cen,attribute,classlabels,cv=cv[lp],scoring="accuracy")
        print ('Centroid accuracy')
        print (score1)
        avg1=0
        for ty in range(0,len(score1)):
            avg1=avg1+score1[ty]
        avg1=avg1/len(score1)
        
        X_digits = np.array(TrainX)
        Y_digits = np.array(TrainY)
        score2 = cross_val_score(regr, X_digits, Y_digits, cv=cv[lp])
        print ('Linear regression accuracy')
        print (score2)
        avg2=0
        for ty in range(0,len(score2)):
            avg2=avg2+score2[ty]
        avg2=avg2/len(score2)
        
        score3=cross_val_score(SVM,attribute,classlabels,cv=cv[lp],scoring="accuracy")
        print('SVM accuracy')
        print (score3)
        avg3=0
        for ty in range(0,len(score3)):
            avg3=avg3+score3[ty]
        avg3=avg3/len(score3)
        
        demoavg=[]
        demoavg.append(avg)
        demoavg.append(avg1)
        demoavg.append(avg3)
        demoavg.append(avg2)        
        average.append(demoavg)

    for i in range(0,len(average)):
        k.append(average[i][0])
        #print (k)
        c.append(average[i][1])
        #print (c)
        s.append(average[i][2])
        #print (s)
        l.append(average[i][3])
    print('Average accuracy of KNN,Centroid,Linear Regression,SVM respectively')    
    print (average)
    import matplotlib.pyplot as plt
    val = 0
    plt.plot(np.zeros_like(average) + val, average, 'x')
    #plt.plot(k)
    #plt.plot(c)
    #plt.plot(s)
    #plt.plot(l)
    plt.text(0.0, 0.975, r'Blue=Knn Red=LR Green=SVM Orange=Centroid')
    plt.xlabel("Cross Validation")
    plt.show()


################################################################################    

num = int(input("Enter the number of classes that you want to consider: "))
classes = []
for i in range(num):
    j = int(input("Enter the "+ str(i+1) + " class you want to consider: "))
    classes.append(j)
pick_data(classes)    
training_instances=int(input("Enter the number of training elements: "))
test_instances=10-training_instances    
test_train_data(training_instances,test_instances)    
choice=int(input("Do you want to perform prediction or cross validation: "))
temporary=[]
if(choice==0):
    accuracy = main() 
    temporary=predictor(TrainX,TrainY,TestX,TestY,accuracy)
    #print (temporary)
    from sklearn.metrics import accuracy_score
    print("\n\n\nThe Accuracy Scores are:")
    for t in range(0,len(temporary)-1):
        print(accuracy_score(temporary[t], TestY))
    print(temporary[3])
    print("for KNN,Centroid, SVM and LR respectively")    
elif(choice==1):
    validator()