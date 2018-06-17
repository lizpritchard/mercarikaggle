# -*- coding: utf-8 -*-
"""
Name: Liz Pritchard
Date: 6/16/2018
Project: Mercari Price Suggestion Challenge for Kaggle
"""

#%% 

import pandas as pd
import seaborn as sns
import os 
import numpy as np

import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

import re
import string

os.chdir("C:/Users/Liz/Mercari Files")

#%%
# Creating dataframes from the files
train = pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")

#%%
# Distribution of prices - Heavily positively skewed
sns.distplot(train.price)

#%%
# Log-transformed distribution due to positive skew
# Confirmed that distribution looks more normal after transformation
train["logPrice"] = np.log(train.price + 1)
sns.distplot(train.logPrice)

#%%
# Taking every word and turning it into a variable
# Want to tokenize words in description to feed into a model later
# Kaggle user ThyKhueLy included this in their kernal
# https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling

stop = set(stopwords.words('english'))
def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
    
train['tokens'] = train['item_description'].map(tokenize)
test['tokens'] = test['item_description'].map(tokenize)

# To create a check point
train.to_csv("mercariTokenizedData.csv")