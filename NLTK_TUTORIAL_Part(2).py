'''
A very similar operation to stemming is called lemmatizing. The major difference between these is, as you saw earlier,
stemming can often create non-existent words,
whereas lemmas are actual words.

So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma.

Some times you will wind up with a very similar word, but sometimes, you will wind up with a completely different word. Let's see some examples.
import nltk
'''

import nltk
import pickle

from nltk.stem import WordNetLemmatizer

##lemmatizer = WordNetLemmatizer()
##
##print(lemmatizer.lemmatize("long"))
##print(lemmatizer.lemmatize("long", pos = "v"))

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

##sample = gutenberg.raw("bible-kjv.txt")
##
##tok = sent_tokenize(sample)
##
##print(tok[5:15])
##
##WordNet is a lexical database for the English language,
##which was created by Princeton, and is part of the NLTK corpus.
##You can use WordNet alongside the NLTK module to find the meanings of words,
##synonyms, antonyms, and more. Let's cover some examples.
##First, you're going to need to import wordnet:


from nltk.corpus import wordnet
##syns = wordnet.synsets("Dog")
##print(syns[1].name())
##
##print(syns[1].definition())
##
##print(syns[1].examples())
##
##good = wordnet.synset('good.a.01')
##print(good.lemmas()[0].antonyms()[0].name())
##
##
##synonyms = []
##antonyms = []
##
##for syn in wordnet.synsets("good"):
##    for l in syn.lemmas():
##        synonyms.append(l.name())
##        if l.antonyms():
##            antonyms.append(l.antonyms()[0].name())
##
##print(synonyms)
##
##print(set(synonyms))
##print(set(antonyms))
##
##w1 = wordnet.synset('ship.n.01')
##w2 = wordnet.synset('boat.n.01')
##print(w1.wup_similarity(w2))
##
##Now that we're comfortable with NLTK, let's try to tackle text classification.

import random
import scipy
import sklearn
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode



##We're calling our class the VoteClassifier, and we're inheriting from NLTK's ClassifierI. Next,
##we're assigning the list of classifiers that are passed to our class to self._classifiers.
##
##Next, we want to go ahead and create our own classify method. We want to call it classify,
##so that we can invoke .classify later on, like a traditional NLTK classifier would allow.

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes) / len(votes)
        return conf



documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


#random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
##print(all_words.most_common(10))
##print(all_words["stupid"])

Word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in Word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))



featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)





##Training classifiers and machine learning algorithms can take a very long time, especially if you're training against a larger data set. Ours is actually pretty small.
##Can you imagine having to train the classifier every time you wanted to fire it up and use it? What horror!
##Instead, what we can do is use the Pickle module to go ahead and serialize our classifier object, so that all we need to do is load that file in real quick.
##
##So, how do we do this? The first step is to save the object.
##To do this, first you need to import pickle at the top of your script, then, after you have trained with .train() the classifier, you can then call the following lines:


##save_classifier = open("naivebayes.pickle","wb")
##pickle.dump(classifier, save_classifier)
##save_classifier.close()

##This opens up a pickle file, preparing to write in bytes some data. Then, we use pickle.dump() to dump the data.
##The first parameter to pickle.dump() is what are you dumping, the second parameter is where are you dumping it.
##
##After that, we close the file as we're supposed to, and that is that, we now have a pickled, or serialized, object saved in our script's directory!
##
##Next, how would we go about opening and using this classifier? The .pickle file is a serialized object, all we need to do now is read it into memory,
##which will be about as quick as reading any other ordinary file. To do this:

##classifier_f = open("naivebayes.pickle", "rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()

##Here, we do a very similar process. We open the file to read as bytes. Then,
##we use pickle.load() to load the file, and we save the data to the classifier variable.
##Then we close the file, and that is that. We now have the same classifier object as before!

##We've seen by now how easy it can be to use classifiers out of the box, and now we want to try some more!
##The best module for Python to do this with is the Scikit-learn (sklearn) module.



voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)




