# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:43:05 2018

@author: krishnan.S
"""

#To Import the Packages
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from nltk.corpus import wordnet
import random
from nltk.corpus import movie_reviews
from nltk.classify import Classifier 
from nltk.classify.scikitlearn import SKlearnClassifier
from sklearn.linear_model import LogisticRegression

##to create a text
text="The Supreme Court on Tuesday directed authorities not to initiate any ‘action’ against 40 lakh persons left out of the draft Assam National Register of Citizens (NRC) published on July 30.The Bench of Justices Ranjan Gogoi and Rohinton Nariman ordered the government, in consultation with State NRC Coordinator Prateek Hajela, to frame a ‘fair’ standard operating procedure (SOP) to deal with the claims and objections of those who did not find their names in the draft NRC."

##to tokenize a sentence
sent=sent_tokenize(text)
print(sent)

#to tokenize words
word=word_tokenize(text)
print(word)

##to remove the stop words
stop_words=set(stopwords.words("english"))

filtered_sentence=[w for w in word if not w in stop_words]
print(filtered_sentence)

##to get the POS tagging
pos=pos_tag(filtered_sentence)
print(pos)

##STEMMING
ps=PorterStemmer()

for w in filtered_sentence:
    print(ps.stem(w))

#Lemmatizer
lemmatizer=WordNetLemmatizer()
for w in filtered_sentence:
    print(lemmatizer.lemmatize(w))
    
##to load a corpus
sample=gutenberg("bible-kvj.txt")

##to initialize the wordnet
syns=wordnet.synsets("program")

##Text Clasification
documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories() 
           for fileid in movie_reviews.fileids(category)]

##To shuffle the documents to train and test the data
random.shuffle(documents)

#To print and verify the shuffled documents
print(documents[1])

#To  add all the words to the list
all_words= []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
#To convert the list into NLTK Frequency Distribution    
all_words=nltk.FreqDist(all_words)

#To print the most common words 
print(all_words.most_common(10))

#To convert the words as features and selecting the top 1500 words
word_features=list(all_words)[:1500]

#To find the features within the document
def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
        
        return features

#To print the features     
print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets=[(find_features(rev),category) for (rev,category) in documents]

#To split the data into training and testing
training_set=featuresets[:1900]
testing_set=featuresets[1900:]

#To apply NAIVE BAYES Classifier
classifier=nltk.NaiveBayesClassifier.train(training_set)
print("NB accu:", (nltk.classify .accuracy(classifier,testing_set))*100)

#To apply LOGISTIC REGRESSION Classifier
Logisticregression_classifier=SKleanrClassifier(LogistcRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression,testing_set))*100)

#To perform SENIMENT on the Text
def sentiment(text):
    feats=find_features(text)
    return classifier.classify(feats)

print(sentiment("This a very good movie."))

