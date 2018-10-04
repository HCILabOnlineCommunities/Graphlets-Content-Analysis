# -*- coding: utf-8 -*-
"""
Created on Fri May 12 02:14:33 2017

@author: James
"""

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    #Used if nltk list of stop_words necessary, but TfidfVectorizer handles most stop words
#    return stem_tokens([word for word in nltk.word_tokenize(text.lower().translate(remove_punctuation_map)) if word not in stopwords.words()])
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
