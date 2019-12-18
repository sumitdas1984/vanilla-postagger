'''
description : this is the test module for the postagger
author      : Sumit
date        : 25/12/2018

run instruction: python postagger_train.py
'''

from nltk import word_tokenize
import pickle
from postagger_train import features

# load the model from disk
clf = pickle.load(open('vanilla_postagger_model.pkl', 'rb'))

def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)
 
print(list(pos_tag(word_tokenize('This is my friend, John.'))))
