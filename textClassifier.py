# sentiment analysis , opinion mining 

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
documents = [(list(movie_reviews.words(fileid)) , category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words  = []

for word in movie_reviews.words():
    all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

print word_features

def find_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev) , category) for (rev , category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior * likelihood / evidence  ---- naivebayes

def trainAndStore():
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    save_classifier = open("naiveBayes.pickle" , "wb")
    pickle.dump(classifier , save_classifier)
    save_classifier.close()

def readAndUse():
    classifier_file = open("naiveBayes.pickle" , "rb")
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier

classifier = readAndUse()
print (" original naive bayes : " , nltk.classify.accuracy(classifier , testing_set)*100)
# classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print (" MNB naive bayes : " , nltk.classify.accuracy(MNB_classifier , testing_set)*100)

# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print (" GNB naive bayes : " , nltk.classify.accuracy(GNB_classifier , testing_set)*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print (" BNB naive bayes : " , nltk.classify.accuracy(BNB_classifier , testing_set)*100)


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print (" LogisticRegression_classifier naive bayes : " , nltk.classify.accuracy(LogisticRegression_classifier , testing_set)*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print (" SGDClassifier_classifier naive bayes : " , nltk.classify.accuracy(SGDClassifier_classifier , testing_set)*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print (" SVC_classifier naive bayes : " , nltk.classify.accuracy(SVC_classifier , testing_set)*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print (" LinearSVC_classifier naive bayes : " , nltk.classify.accuracy(LinearSVC_classifier , testing_set)*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print (" NuSVC_classifier naive bayes : " , nltk.classify.accuracy(NuSVC_classifier , testing_set)*100)


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
        conf = float(choice_votes / len(votes))

        return conf



voted_classifiers = voteClassifier(NuSVC_classifier ,LinearSVC_classifier,SGDClassifier_classifier,LogisticRegression_classifier,BNB_classifier,MNB_classifier,classifier )
print (" voted classifier : " , nltk.classify.accuracy(voted_classifiers , testing_set)*100)
print ("classification : " , voted_classifiers.classify(testing_set[0][0]) , "confidence : " , voted_classifiers.confidence(testing_set[0][0])*100)
