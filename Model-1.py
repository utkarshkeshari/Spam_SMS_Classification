import csv
import random
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')

data = {"text":[], "class":[]}

#Read File
f = open("/home/divyanshu/PycharmProjects/Learn/SMSClassification/SMSSpamCollection", "r")
reader=csv.reader(f,delimiter='\t')

#For each line
for target, value in reader:
    tokens = []
    token = tokenizer.tokenize(value)                   #Tokenize the SMS. This will seperate out each word and token will be a List of words
    for i in token:
        if i not in stop:                               #Remove Stop Words
            tokens.append(i)

	value = " ".join(tokens).decode('cp1252', 'ignore') #Join the remaining words to create a sentence
	data["text"].append(value)
	data["class"].append(target)

f.close()

length = len(data["text"])
sample = random.sample(range(0, length), length)        #Randomize the Data and Target
data["text"] = [data["text"][i] for i in sample]
data["class"] = [data["class"][i] for i in sample]

pipeline = Pipeline([                                   #Putting all the operation in Pipeline. It will first find out
	('vect',  CountVectorizer(ngram_range=(1, 2))),     #bag of words Feature and put the result vect variable
    ('tfidf', TfidfTransformer()),                      #then it will calculates the tf-idf of features in vect
	('clf',  MultinomialNB()) ])                        #and finally it will use classifier MultinomialNB to classify using feature in tfidf

k_fold = KFold(n=len(data["text"]), n_folds=10)         #Split data in 10 Folds

new_data_text = np.asarray(data['text'])                #Conver data in numpy array
new_data_class = np.asarray(data['class'])
scores = []

for train_indices, test_indices in k_fold:              #for Train,Test index in KFold Find out the training and test data

    train_text = new_data_text[train_indices]
    train_y = new_data_class[train_indices]
    test_text = new_data_text[test_indices]
    test_y = new_data_class[test_indices]

    pipeline.fit(train_text, train_y)                   #This will fit the each Fold serially Bag_of_Words, tf-idf, Classifier
    predicted = pipeline.predict(test_text)             #Predict the class on test data

    score = pipeline.score(test_text, test_y)           #Calculates the Accuracy Score
    scores.append(score)                                #Store Accuracy scores

print(metrics.classification_report(test_y, predicted)) #Generate the Classification report. It is Basically confusion Matrix along with some other Info

score = sum(scores) / len(scores)                       #finds the mean accuracy of scores calcualted by pipeline

print "Mean Accuracy: " + str(score)


#=======================================================================================================================

'''
# This is the result when we are using Bag_of_words and tf-idf both to extract feature before going to Classifier

        precision    recall  f1-score   support

        ham       0.99      1.00      0.99      7119
       spam       1.00      0.96      0.98      1953

avg / total       0.99      0.99      0.99      9072

Mean Accuracy: 0.990234229134

#----------------------------------------------------

#This Result is when we use only Bag_of_words Feature

       precision    recall  f1-score   support

        ham       0.99      1.00      0.99      7156
       spam       0.99      0.96      0.98      1916

avg / total       0.99      0.99      0.99      9072

Mean Accuracy: 0.987842393541

'''