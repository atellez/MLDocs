
## Example to learn how to implement a Support Vector Machine (SVM) using Python's Scikit-Learn

## Doing this FIRST requires you to have Scikit-Learn installed on your machine. If you DO NOT have Scikit-Learn installed, please visit the 'How-To-Install _______' section.
## If you do have Scikit-Learn installed (from now on, we'll just refer to it as 'scikit') then let's jump right in.

## We will apply a Support Vector Machine (SVM) for image recognition and in particular, classifying faces using the Olivetti Faces Dataset.
## Lucky for you, scikit ships with the Olivetti dataset so we don't have to spend time loading it in (we will learn to do that in the 'Text Classification Using Naive Bayes Algorithm' chapter!)
## First let's load our dataset and then look at the overall shape of our dataset.
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

## Step 1: Get Data
## You will see that the above 3 imports are usually the very first lines of code for most examples of scikit learn. We are simply importing the 'Big 3' packages that are most common to running ML algorithims in Python.
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
faces.keys()
print faces.DESCR

## As you will read in the description ('DESCR'), this dataset contains 400 images of 40 different people. 
## Also note that the 'keys()' function returns the meta-data about the dataset: images (actual faces), data (4,096 features), target (400 IDs), and DESCR (description of the dataset)
## Each image (face) is a 64 x 64 raw pixel image which means the # of dimensions in our dataset is 4,096 features wide (each pixel is a feature).
## You can confirm the dataset 'topology' by running the following commands:
print faces.images.shape
print faces.data.shape

## Scikit has already identified the  the faces (i.e. an ID that corresponds to a person's name for each of the 400 images); This is referred to as the 'target' and the raw pixels that are input features is the 'data'.
print faces.target.shape
## The answer to the above line of code should be 400 becuase there are 400 images, each of which has an ID that is linked to a person. 
## Suppose you want to look at the first row - that is 4,096 features wide - of the dataset AND the corresponding target class.  You would just type:
print faces.data[0], faces.target[0]

## Step 2: Check data to ensure normalization
## As with most machine learning algorithms, you want to be sure to normalize your dataset to make sure all the features are on the same scale.  We can examine the dataset to see if it's normalized between 0 and 1 by calling some simple Numpy functions:
print np.min(faces.data), np.max(faces.data), np.mean(faces.data)

## Luckily, this data has already been normalized BUT in the event that your data is NOT normalized, you will want to do so by defining a function similar to the one we did in Chapter 1.
normalizedData = (faces.data - np.min(faces.data, 0)) / (np.max(faces.data, 0) + 0.0001)  # 0-1 scaling

## Step 3: Set up our Support Vector Machine to using different kernels
## For any scikit machine learning algorithm, we will always start by calling the respective class from the respective module.
## In our case we will are going to import the SVC (Support Vector Classifier) from the sklearn.svm module as such:
from sklearn.svm import SVC  #Note that this is a classification problem. If this is a regression problem you can use the SVR class

## The most important parameter in the SVC class is the kernel function which is defaulted to the 'rbf' kernel which allows us to model non-linear problems.  To start we will use the 'rbf' kernel and then the 'tanh' kernel.
## Note that there are 3 other kernels we can employ: 'linear', 'polynomial' and 'sigmoid' (tanh)
svc_rbf = SVC(kernel='rbf')  #Recall that the default is rbf
svc_sigmoid = SVC(kernel='sigmoid')
svc_linear = SVC(kernel='linear')

## Step 4: Split data into training and testing
## As always, we need to partition our dataset into a training and testing set.  Lucky for you scikit learn has a nice and convenient function that allows us to do this:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size = 0.2, random_state = 0)   # Note that the data is already shuffled BUT if it was not, you can call for shuffling by stating 'random_state = 33')

## To check that we did this correctly, let's just check a few things using the functions we already know:
print X_train.shape, y_train.shape # Should read: (320, 4096) (320, ). This basically says there are 320 training instances, 4,096 features wide.  Our y_train contains the target values for the 320 data points.

## Step 5: Train our models and score them against the hold out set.
## The basic steps are a) call the ML algorithm class (done) b) run it on the training set, c) predict the testing dataset, and d) look at the confusion matrix
## First, we are going to import the 'metrics' library from scikit which will allow us to score and get the accuracy on our models.
from sklearn import metrics
svc_rbf.fit(X_train, y_train)
svc_sigmoid.fit(X_train, y_train)
svc_linear.fit(X_train, y_train)

print "Training set accuracy for RBF Kernel:"
print svc_rbf.score(X_train, y_train) # I get 12% Accuracy! Yikes!

print "Training set accuracy for Sigmoid Kernel:"
print svc_sigmoid.score(X_train, y_train) # 3% Accuracy...even worse! Is there any hope?!

print "Training set accuracy for Linear Kernel:"
print svc_linear.score(X_train, y_train) # I get 100% Accuracy(!!) Is that even right?! We'll see...

## Whoa, that can't be right huh? 100% Accuracy?! 'Nothing is 100% accurate!', you're thinking. Well, that's the beauty of having a TEST hold out dataset and we are now going to take our trained Support Vector Machine and see how it does against the hold out dataset.
print "Testing set accuracy for RBF Kernel:"
print svc_rbf.score(X_test, y_test) # Yuck..0% accuracy. Put this model on failblog.com

print "Testing set accuracy for Sigmoid Kernel:"
print svc_sigmoid.score(X_test, y_test) # Eeks...also 0%.

print "Testing set accuracy for Linear Kernel:"
print svc_linear.score(X_test, y_test) # ahhh...intersting, I get ~99% Accuracy.

## Indeed, we see the power of Support Vector Machines and in particular, the linear kernel. But let's keep our healthy skepticism and ask a different question:
## Rather than try to classify faces to their IDs, let's see if we can identify those faces that have facial hair from those that do not.
## If you want to see for yourself the entire Olivetti faces pictures, copy+paste this into your web-browser: http://www.cs.nyu.edu/~roweis/data/olivettifaces.gif
## Step 1 is to define the ranges of pictures that of men that have OBVIOUS facial hair (stubble doesn't count!):
facialHair = [
  (60, 69), (100, 109), (130, 139), (160, 169),
  (240, 249), (250, 259), (270, 279), (360, 369)
]

