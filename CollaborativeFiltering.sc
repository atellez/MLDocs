
## Example that shows you how to implement a ratings recommendation engine for a dataset THAT IS NOT the moivelens dataset(!!!)
## Please download the files from our dataset branch. For more information refer here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
## About this dataset: Some dude named Cai-Nicolas Zeigler collected a whole-buncha book ratings from the Book-Crossing community in 2004.
## There are 3 files in the dataset: BX-Users.csv (278,858 users), BX-Books.csv (271,379 books), and BX-Ratings.csv (1,149,780 book ratings).

## NOTE: This tutorial requires Apache Spark. If you do not have this installed, STOP, and get this installed by following the instructions on our 'Appendix - Installations' document.

## Step 1: Fire up the spark-shell and load in our dataset.
## In your terminal, fire up spark-shell
cd wherever/you/downloaded/spark
./bin/spark-shell

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
## The first import loads the ALS (Alternating Least Squares) library that we will use to build our recommendation engine.
## The second import loads in a very special RDD type called 'Rating' which is a special form of RDD that requires the follow syntax: (UserID, ProductID, Rating)
## Essentially, 'Rating' is a dataset of triplets that we will feed into the ALS class to use for predicting new books to read for the readers in our book club!