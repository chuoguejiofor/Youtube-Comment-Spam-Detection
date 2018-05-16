# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

df1 = pd.read_csv("~/Downloads/YouTube-Spam-Collection-v1/Youtube01-Psy.csv")
df2 = pd.read_csv("~/Downloads/YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv")
df3 = pd.read_csv("~/Downloads/YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv")
df4 = pd.read_csv("~/Downloads/YouTube-Spam-Collection-v1/Youtube04-Eminem.csv")
df5 = pd.read_csv("~/Downloads/YouTube-Spam-Collection-v1/Youtube05-Shakira.csv")

df1['YOUTUBE_ID'] = '9bZkp7q19f0'
df2['YOUTUBE_ID'] = 'CevxZvSJLk8'
df3['YOUTUBE_ID'] = 'KQ6zr6kCPj8'
df4['YOUTUBE_ID'] = 'uelHwf8o7_U'
df5['YOUTUBE_ID'] = 'pRpeEdMmmQ0'

df = df1.append(df2).append(df3).append(df4).append(df5)

counts = count_vect.fit_transform(df.CONTENT.values)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
targets = df.CLASS.values
classifier.fit(counts, targets)

examples = ['Free Viagra call today!', "I'm going to attend the Linux users group tomorrow.", "Click here for free money", "Like for an iPhone", "I thought Infinity War was amazing"]
example_counts = count_vect.transform(examples)
predictions = classifier.predict(example_counts)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])

pipeline.fit(df.CONTENT.values, df.CLASS.values)
predictions = pipeline.predict(df2.CONTENT.values)
np.mean(predictions == df2.CLASS.values)

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(df.CONTENT, df.CLASS,random_state = 0)
pipeline.fit(train_X, train_y)
predictions = pipeline.predict(val_X)

from sklearn.metrics import accuracy_score
print(accuracy_score(val_y, predictions))