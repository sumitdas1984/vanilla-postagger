from nltk import word_tokenize
import pickle
from postagger_train import features

# load the model from disk
clf = pickle.load(open('vanilla_postagger_model.sav', 'rb'))

def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)
 
print(list(pos_tag(word_tokenize('This is my friend, John.'))))
