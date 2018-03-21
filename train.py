
from nltk.corpus import stopwords

from sklearn import svm
from nltk.stem import *
import pickle
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from sklearn.metrics import *
from nltk.stem.porter import *
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import nltk
import argparse
import pandas as pd

# parser = argparse.ArgumentParser()
# parser.add_argument("-lr", "--LogisticRegression", help="LogisticRegression classifier",action="store_true")
# parser.add_argument("-svm", "--SVM_classifier", help="SVM_classifier",action="store_true")
# parser.add_argument("-rf", "--RandomForest_classifier", help="RandomForest_classifier",action="store_true")
# parser.add_argument("-nb", "--naive_bayes", help="naive_bayes classifier",action="store_true")
# args = parser.parse_args()

#filter(lambda a: a not in ['not','no','none'],stopwords.words('english'))
def classifier(clf):
	print("##########Training Classifier##########",end="\n")
	vectorizer = TfidfVectorizer(stop_words=None,decode_error='ignore',min_df=5, max_df = 0.8,use_idf=True,ngram_range=(1,3),norm='l2')
	#vectorizer = CountVectorizer(stop_words=None,decode_error='ignore',min_df=5, max_df = 0.8,ngram_range=(1,3),binary=True)
	#vectorizer=TfidfVectorizer(decode_error='ignore',ngram_range=(1,2),use_idf=True,norm='l2')
	#train_vectors = vectorizer.fit_transform(train_data)
	f=pd.read_csv("./train.csv", encoding = "cp1252")['SentimentText'].values## Tweet file
	g=pd.read_csv("./train.csv", encoding = "cp1252")['Sentiment'].values## labels file 
	#vectorizer = HashingVectorizer(stop_words='english', decode_error='ignore',binary=True,analyzer=tok)
	# print(f)
	x=vectorizer.fit_transform(f)
	y=np.array([int(lines) for lines in g])
	cross=ShuffleSplit(x.shape[0],n_iter=10,test_size=0.2)
	if clf == 0:
		classifier=LogisticRegression(penalty='l2',solver='sag',n_jobs=10)
	elif clf == 1:
		classifier=svm.LinearSVC(loss='hinge')
	elif clf == 2:
		classifier=RandomForestClassifier(n_estimators=10, criterion='entropy',n_jobs=20)
	elif clf == 3:
		classifier=MultinomialNB()
	else:
		classifier=MultinomialNB()
	score = 0
	for s,t in cross:
		classifier.fit(x[s],y[s])
		joblib.dump(classifier,'binary_lin.pkl')
		# print(type(x[t]))
		pred = classifier.predict(x[t])
		# print ("Confusion Matrix")
		# score = confusion_matrix(y[t], pred)
		# print( score)
		# print ("F1 Score")
		# score=f1_score(y[t],pred,pos_label=4)
		# print (score)

		score=accuracy_score(y[t],pred)
	#	for i in xrange(len(t)):
	#		if y[t[i]]!=pred[i]:
	#			print t[i]
	print ("Accuracy: " + str(score*100)+"\r")
	return classifier , vectorizer
