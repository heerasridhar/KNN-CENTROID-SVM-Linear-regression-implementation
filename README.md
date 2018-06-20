# K-nearest neighbours-Centroid-Support vector machine-Linear-regression implementation

There are two datasets ATNT-face-image400 and Hand-written-26-letters.

data set: ATNT-face-image400.txt  :
 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=28x23).

Total 40 classes. each class has 10 images. Total 40*10=400 images

data set: Hand-written-26-letters.txt :

Text file. 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=20x16).
Total 26 classes and each class has 39 images. Total 26*39=1014 images.

---------------------------------------------------------------------------
The following tasks were performed:

TASK-1:
Used the data-handler to select "A,B,C,D,E" classes from the hand-written-letter data. 
From this smaller dataset, Generated a training and test data: for each class
using the first 30 images for training and the remaining 9 images for test.
Later performed classification on the generated data using the four classifers.


TASK-2:
On ATNT data, 5-fold cross-validation (CV) was done using  each of the 
four classifiers: KNN, centroid, Linear Regression and SVM.
Found the classification accuracy for each classifier and also the average of these 5 accuracy numbers.

TASK-3:
On handwritten letter data, on 10 classes. Used the data handler to generate training and test data files.
For seven different splits:  (train=5 test=34), (train=10 test=29),  (train=15 test=24) , 
       (train=20 test=19), (train=25 test=24) , (train=30 test=9) ,  (train=35 test=4). 
 On these seven different cases, the centroid classifier was used to compute average test image classification
 accuracy and these 7 average accuracy were plotted on one curve in a figure. 

TASK-4:
RepeatD task-3 for another different 10 classes and the 7 average accuracy were plotted in one curve 
    
TASK-5:
Data handler.

The following subroutine were implemented in the code.

subroutine-1: pickDataClass(filename, class_ids)
 
  filename: char_string specifying the data file to read. For example, 'ATNT_face_image.txt'
  class_ids:  array that contains the classes to be pick. 
  Returns: an multi-dimension array or a file, containing the data (both attribute vectors and class labels) 
           of the selected classes
  We use this subroutine to pick a small part of the data to do experiments. 

 
subroutine-2: splitData2TestTrain(filename, number_per_class,  test_instances)
  filename: char_string specifying the data file to read. This can also be an array containing input data.
  number_per_class: number of data instances in each class (we assume every class has the same number of data instances)
 
  Return/output: Training_attributeVector(trainX), Training_labels(trainY), Test_attributeVectors(testX), Test_labels(testY)
  The data should easily feed into a classifier.

subroutine-3:
   This routine will store (trainX,trainY) into a training data file, 
   and store (testX,testY) into a test data file. The format of these files is determined by 
   
Subroutine-4: "letter_2_digit_convert" that converts a character string to an integer array. 
   For example,letter_2_digit_convert('ACFG') returns array (1, 3, 6, 7). 
   
