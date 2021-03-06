
## Example to better understand Nearest Neighbor Algorithm using R-Package: "caret"
## download package from here: http://cran.r-project.org/web/packages/caret/
## it will mean loading about 20 some packages

## Step 1: Get Iris Dataset from R and do some exploratory analysis 
## iris dataset is pre-loaded into R
data(iris)
str(iris)
summary(iris)
## str is short for 'structure' and allows us to examine the data structure that sits behind the dataset: Iris.  This is extremely useful when loading new data.
## Note that this is a classification problem in that we are trying to predict the Iris type: Versicolor, Virginica, or Setosa which is the column 'Species'

## Now let's plot a histogram of that looks at Sepal Length and it's frequency within the dataset
hist(iris$Sepal.Length)

#Suppose we want to see how many classes of Iris we have in our dataset? We can generate a table of our outcome variable, 'Species' as such:
table(iris$Species)
#What do you think this does?
prop.table(table(iris$Species))

## Step 2: Shuffle the Deck
## Notice how in our original dataset (literally type in 'iris' in R) you will see how the first 50 datapoints all correspond to 'Setosa', the next 50 are all 'versicolor' and the final 50 are all 'viriginca'.
## If we didn't shuffle our dataset and let's say we trained a model on the first 100 samples, our model would only learn either setosa or versicolor and not viriginca. We don't want that so before we split into training and testing we need to shuffle our deck of cards - so to speak so that we get a nice mix of all classes.
set.seed(1234)
iris_shuffled <-iris[order(runif(150)),]
iris_shuffled

## The set seed function is a random number generator that starts from a position known as the 'seed', which in our case is the number '1234'. The reason why we need to 'set the seed' is so that if we were to re-run this analysis again, an identical result can be achieved each time.
## The runif(150) command generates a random list of 150 numbers. Why 150 and not 155 or 200? Because our Iris dataset has 150 points and thus we need 150 random numbers.


## Step 3: Normalize Features
## When dealing with data - especially distance functions like Nearest Neighbor, K-Means or PCA - it helps to normalize our data.  We can do that one of two ways in R using functions:
normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
}

## Okay, so what did we just do? The above function is taking a vector called 'X', and for each value in 'X', we are going to subtract it from the minumum value in 'X' and then divide that difference by the range of values in 'X' (range = highest - lowest values)
## Once we create this function we can call it - i.e. use it - anytime we like!
normalize(c(100,200,300,400,500))
iris_normalized <- as.data.frame(lapply(iris_shuffled[1:4],normalize))
iris_normalized$Species <- iris_shuffled$Species
str(iris_normalized)

## Wait...what? So the code above reads like this: Take the first 4 columns in the Iris dataset (which is everything BUT the variable 'Species' which is what we are trying to predict) and treat it like a List that we are going to APPLY (<- get it?! lapply) our created function 'normalize'.  Then, return this list as a data frame that we can run a machine learning algorithm from!
## Notice the difference in datastrucutre when you run these two code snippets:
lapply(iris_shuffled[1:4],normalize)
as.data.frame(lapply(iris_shuffled[1:4],normalize))
## Finally, we append our class variable to our newly transformed data frame (we can only do this BECAUSE we have 2 data frames!)

## Step 4: Split our dataset into Training and Testing
iris_training <- iris_normalized[1:100,]
iris_testing <- iris_normalized[101:150,]

## So our training set will have 100 'mixed' samples of Iris flower and our testing set (AKA the holdout set) will have 50 samples of Iris flowers.
## Because the distribution of Iris flower is balanced - that is, 50 Setosa, 50 Virginca and 50 Versicolor - if we did this correctly we should have the same distribution of Iris flowers in both the training and the testing datasets.
table(iris_training$Species)
table(iris_testing$Species)

## Question: Suppose you wanted to see these as percentages, how would you do that? (hint: use 'prop.table')

## Step 5: Train our algorithm, kNN!
## Whew, finally we are getting to the modeling step. To do this we are first going to call the library(caret) and then start running some models.
## The basic idea is: 1) Train a model using k=Some number. 2) Test this model against the holdout set (iris_testing) using the 'predict' function. 3) Calculate the Error. 4) Repeat Steps 1-3 until a good model is reached!

library(caret)
model_knn3 <- knn3(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, k = 3, data = iris_training)
## Within the package "caret", there is an implementation of the Nearest Neighbor algorithm. In the code above, we are defining our first model which predicts 'Species' using the 4 features of the iris_training dataset. 
## The number of datapoints to look at to determine the class of iris (i.e. k = 3) is 3.  So the idea is what are the 3 nearest classes of iris flower (Euclidian distance) to a given training sample.  And then, cast a vote among the 3 cases.  So if 2/3 points are Versicolor and 1/3 is Setosa, the majority = Veriscolor.
## We can try a few different models varying the # of neighbors, k, and then at the end see which gives us the best result. 
## Finally, in this example there are 4 features but in YOUR DATA you might have 100+ features. You don't want to write all that out in R for defining the model.  Instead, R has a nifty cheat which says 'include all these parameters' (this is represented by a '.' after the '~') without having to write out each parameter individually. 

model_knn5 <- knn3(Species ~ ., k = 5, data = iris_training)
model_knn7 <- knn3(Species ~ ., k = 7, data = iris_training)
model_knn21 <- knn3(Species ~ ., k = 21, data = iris_training)

## Step 6: Predict the classes or Iris on our holdout dataset, iris_testing
## Most algorithms have a 'predict' function that is attached to them that allows us to score a model against a holdout set, which in our case, is the iris_testing dataset.
## We can predict the iris class and output one of 2 things: The class of iris flower ('Setosa', 'Versicolor', 'Virginica') OR the posterior probabilities for each class ('Setosa' = 30%, 'Versicolor' = 50%, 'Viriginca' = 20%, for example)
## This can be accomplished in 2 ways:

knn3_prediction_probs <- predict(model_knn3, iris_testing)
## The above returns the posterior probabilities for each class.
knn3_prediction_probs

knn3_prediction <- predict(model_knn3, iris_testing, type = "class")
knn3_prediction
## This line of code outputs the class that has the highest probability attached to it.
## Now, we predict for the rest of our models using the type="class" prediction.
knn5_prediction <- predict(model_knn5, iris_testing, type = "class")
knn7_prediction <- predict(model_knn7, iris_testing, type = "class")
knn21_prediction <- predict(model_knn21, iris_testing, type = "class")

## Step 7: Evaluate the various models and see which one is the best!
## So now that we have 4 models based on different k-values, and we scored them against our holdout set, it's now time to see how our models did.
## The most common way to evaluate classification problems (which this is) is by generating a confusion matrix which shows where the errors/correct predictions are being made.
## Let's generate one first and then walk through the interpreation of it all:
table(true = iris_testing$Species, predicted = knn3_prediction)

## So we are calling the table function again and now we are specifying that the true values is the 'Species' column in our iris_testing dataset and the predictions are going to come from our knn3_prediction which outputs 1 column of iris types.
## The output should be a 3 x 3 table and what you're looking for is a 'heavy diagonal' which indicates that actual = predicted.  
## Just using this simple model we see that for 3 nearest-neighbors, our model does a pretty good job predicting iris type flowers! Only 2 flowers were incorrectly predicted amongst the holdout set of 50 flowers. Not shabby!

## What are the evaluation metrics for the other models? Do we overfit (i.e. learn our training data too well that we can't predict on the holdout set) when we use too many nearest-neighbors?

