'''
description : this is the training module for the postagger
author      : Sumit
date        : 25/12/2018

run instruction: python postagger_train.py
'''

import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import pickle

 
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y
 

def train_tagger():
    tagged_sentences = nltk.corpus.treebank.tagged_sents()

    # print tagged_sentences[0]
    print("Tagged sentences: ", len(tagged_sentences))
    print("Tagged words:", len(nltk.corpus.brown.tagged_words()))

    # Split the dataset for training and testing
    cutoff = int(.75 * len(tagged_sentences))
    training_sentences = tagged_sentences[:cutoff]
    test_sentences = tagged_sentences[cutoff:]

    print("Train dataset: ", len(training_sentences))   # 2935
    print("Test dataset: ", len(test_sentences))         # 979

    X, y = transform_to_dataset(training_sentences)

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])

    clf.fit(X[:10000], y[:10000])   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
    print("this is a training run")
    print('Training run completed')

    X_test, y_test = transform_to_dataset(test_sentences)

    print("The calculated accuracy:", clf.score(X_test, y_test))
    print("training accuracy calculated")

    # save the model to disk
    filename = 'vanilla_postagger_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

if __name__ == '__main__':
    train_tagger()
