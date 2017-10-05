# sentiment analysis , opinion mining 

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.tokenize import word_tokenize

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode# sentiment analysis , opinion mining 

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from os.path import exists,join
from os import mkdir

import sys
reload(sys)  
sys.setdefaultencoding('cp850')


# posterior = prior * likelih

listOfClassifiers = ['NuSVC_classifier' ,'LinearSVC_classifier','SGDClassifier_classifier','LogisticRegression_classifier','BNB_classifier','MNB_classifier','classifier']

def pickleClassifier(givenClassifier , nameClassifier):
    classifierFile = open(join('pickleClassifiers',nameClassifier)+'.pickle' ,'wb')
    pickle.dump(givenClassifier , classifierFile)
    classifierFile.close()

def loadClassifier(nameClassifier):
    classifierFile = open(join('pickleClassifiers',nameClassifier)+'.pickle' ,'rb')
    classifier = pickle.load(classifierFile)
    classifierFile.close()
    return classifier    

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

short_pos = open("positive.txt" , "r").read()
short_neg = open("negative.txt" , "r").read()

documents = []
all_words = []
allowed_word_types = ['J']

for r in short_pos.split('\n'):
    documents.append((r , 'pos'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append((r , 'neg'))
    words = word_tokenize(r)
    neg = nltk.pos_tag(words)
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev) , category) for (rev , category) in documents]
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]



if not exists('pickleClassifiers'):

    mkdir('pickleClassifiers')

    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    print (" original naive bayes : " , nltk.classify.accuracy(NB_classifier , testing_set)*100)
    NB_classifier.show_most_informative_features(15)
    pickleClassifier(NB_classifier , 'NB_classifier')

    print 'done with classifier 1'

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print (" MNB naive bayes : " , nltk.classify.accuracy(MNB_classifier , testing_set)*100)
    pickleClassifier(MNB_classifier,'MNB_classifier')

    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)
    print (" BNB naive bayes : " , nltk.classify.accuracy(BNB_classifier , testing_set)*100)
    pickleClassifier(BNB_classifier,'BNB_classifier')

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print (" LogisticRegression_classifier naive bayes : " , nltk.classify.accuracy(LogisticRegression_classifier , testing_set)*100)
    pickleClassifier(LogisticRegression_classifier,'LogisticRegression_classifier')

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print (" SGDClassifier_classifier naive bayes : " , nltk.classify.accuracy(SGDClassifier_classifier , testing_set)*100)
    pickleClassifier(SGDClassifier_classifier,'SGDClassifier_classifier')

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print (" LinearSVC_classifier naive bayes : " , nltk.classify.accuracy(LinearSVC_classifier , testing_set)*100)
    pickleClassifier(LinearSVC_classifier,'LinearSVC_classifier')

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print (" NuSVC_classifier naive bayes : " , nltk.classify.accuracy(NuSVC_classifier , testing_set)*100)
    pickleClassifier(NuSVC_classifier,'NuSVC_classifier')


print 'Loading Pickles'
NB_classifier = loadClassifier('NB_classifier')
# NuSVC_classifier = loadClassifier('NuSVC_classifier')
LinearSVC_classifier = loadClassifier('LinearSVC_classifier')
# SGDClassifier_classifier = loadClassifier('SGDClassifier_classifier')
LogisticRegression_classifier = loadClassifier('LogisticRegression_classifier')
MNB_classifier = loadClassifier('MNB_classifier')
BNB_classifier = loadClassifier('BNB_classifier')
# training_set = loadClassifier('training_set')
# testing_set = loadClassifier('testing_set')
print 'Done Loading Pickles'

class voteClassifier(ClassifierI):
    def __init__(self , *classifiers):
        self._classifiers = classifiers

    def classify(self , features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self , features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = float(float(choice_votes) / float(len(votes)))

        return conf



voted_classifiers = voteClassifier(LinearSVC_classifier,LogisticRegression_classifier,BNB_classifier,MNB_classifier,NB_classifier )

def sentiment(text):
    features = find_features(text)

    return voted_classifiers.classify(features) , voted_classifiers.confidence(features)