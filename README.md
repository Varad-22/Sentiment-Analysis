
# Sentiment Analysis

## Overview
Sentiment Analysis model is a machine learning model that classifies sentiment of user into three classes, namely positive review, neutral review and negative review. 

## Approach
I decided to train the model using some basic classification model like Random Forest, SVM, Decision Tree, and Naive Bayes. The major problem encountered was large training time.
Using Artificial Neural Network reduced training time, and gave good accuracy on training set.



## Model
The model first extract the data from the dataset and preprocess the data to replace any NaN value found. In this model, it replaces NaN value by "space" charecter. 
The model then clean the dataset by removing all the stopwords from the dataset. This process is achieved by using the "nltk" library in python. Since the set of stopwords pre-stored contains negative words as well, it became necessary to remove these words from list of stopwords. After accomplishing this task, the model then Lemmatizes and Tokenizes the sentences. This is done by using "re" library and "nltk" library. 
The next step is tokenization and creating space array, using the "sci-kit learn" library. 
After splitting data into train and test set, we have to apply One Hot Encoding to the target variable (sentiment) to classify into three different classes. 
The next step is to train the Artificial Neural Network model and predicting the results, and calculating the accuarcy of model.  

## Evaluation
On test set, model performed with accuracy score of 72.46% 