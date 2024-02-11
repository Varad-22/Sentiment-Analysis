# import libraries

import numpy as np 
import pandas as pd 

# load dataset

dataset = pd.read_csv('sentiment_analysis_dataset.csv', index_col=0)

# cleaning

for i in range(27481):
    data=[]
    if str(dataset.values[i][0])=="nan":
        data.append(" ")
        data.append(dataset.values[i][1])
        # print(data)
        dataset.values[i] = data
        
# starting nlp 

import nltk
import re
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf 
tf.keras.utils.set_random_seed(17)

# cleaning dataset

corpus=[]

## removing negative words from stopwords as we have positive and negative sentiments

all_stops = stopwords.words('english')
lst = ['no', 'nor', 'not', "don't", 'should','couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'ain', 'aren', "aren't"]

for i in lst:
    all_stops.remove(i)

for i in range(27481):
    review = re.sub('[^a-zA-Z]', ' ', dataset.values[i][0])
    review = review.lower().split()
    lem = WordNetLemmatizer()
    review = [lem.lemmatize(word) for word in review if word not in set(all_stops)]
    review = " ".join(review)
    corpus.append(review)

# tokenisation and space array with sklearn
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=15200) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1:].values

# train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# encoding the dependent variable 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])])
y_train = np.array(ct.fit_transform(y_train))
y_test = np.array(ct.transform(y_test))

# using tensorFlow

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=5, min_lr=0.01)
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=128, activation='relu', ))
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))
ann.compile(optimizer = tf.keras.optimizers.RMSprop(), loss = 'categorical_crossentropy', metrics = ['accuracy', 'mse'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 2, callbacks=[reduce_lr])
y_pred = ann.predict(X_test)

y_pred_lst = y_pred.tolist()
label = ['negative', 'neutral','positive']
y_prediction = np.array([label[prob.index(max(prob))] for prob in y_pred_lst])
y_prediction = np.reshape(y_prediction, newshape=(len(y_prediction),1))
y_prediction = np.array(ct.transform(y_prediction))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_prediction))