# coding: utf-8

# In[1]:

import codecs
import json
import pandas as pd
import json
import codecs
import re
import random
import numpy as np
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin

def read_dataset(path):
  with codecs.open(path, 'r', 'utf-8') as myFile:
    content = myFile.read()
  dataset = json.loads(content)
  return dataset


# In[14]:

###load JSON file
file_name = 'data/pizza_request_dataset.json'
dataset = read_dataset(file_name)
df = pd.read_json(json.dumps(dataset, sort_keys=True, indent=2))

###create random 90-10 split
X = df
print len(X)

rows = random.sample(X.index, int(0.9*len(X)) + 1)
X_train = X.ix[rows]
X_test = X.drop(rows)
y_train = X_train.requester_received_pizza.astype(int)
y_test = X_test.requester_received_pizza.astype(int)

print "Data loading and train-test splits = DONE!"
###

###subsample data
# print "Subsampling"
# subsample = 30
# X_train, y_train, X_test, y_test = X_train[:subsample], y_train[:subsample], X_test[:subsample], y_test[:subsample]


# In[ ]:




# In[3]:

print "part 1"


# In[ ]:

###Model 1 - a) n-grams

### build pipeline; fit train; predict on test
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

##vectorizer arguments blah!

tokenizer=None#word_tokenize
# stop_words=nltk.corpus.stopwords.words("english")#None
ngram_range=(1, 2)
lowercase=True
max_features=500
binary=False
dtype=np.float64

class TextExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        return X[self.vars]  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing

###create unigram vectorizer
uniVect = CountVectorizer(decode_error="ignore",
#                                tokenizer=tokenizer,
#                                stop_words=stop_words,
                               ngram_range=(1,1),
                               lowercase=lowercase,
                               binary=binary,
                               dtype=dtype,
                               max_features=max_features)

###create bigram vect
biVect = CountVectorizer(decode_error="ignore",
#                                tokenizer=tokenizer,
#                                stop_words=stop_words,
                               ngram_range=(2,2),
                               lowercase=lowercase,
                               binary=binary,
                               dtype=dtype,
                               max_features=max_features)

# load custom features and FeatureUnion with Vectorizer
features = []
features.append(('unigram', uniVect))
features.append(('bigram', biVect))
all_features = FeatureUnion(features)

linear_svc = svm.SVC(kernel='linear', probability=True)

###create pipeline
text_clf = Pipeline([
                     ('getText', TextExtractor('request_text_edit_aware')),
                     ('all', all_features),
#                      ('tfidf', TfidfTransformer()),
                    ('clf', linear_svc),
                    ])

# text_clf = Pipeline([
#                      ('vect', vectorizer),
# #                      ('tfidf', TfidfTransformer()),
#                     ('clf', linear_svc),
#                     ])

###fit training data
print "Fitting!"
text_clf = text_clf.fit(X_train, y_train)

### predict on test data
predicted = text_clf.predict(X_test)
np.mean(predicted == y_test)
# print "NO!"
probas = text_clf.predict_proba(X_test)
# print "YO!"
###get performance metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
accuracy = accuracy_score(y_test,predicted)
#         accuracy = np.mean(y_CVtest == predicted)
precision, recall, fscore, sup = precision_recall_fscore_support(y_test, predicted, average='binary', pos_label=0)
print precision, recall, fscore, accuracy# np.mean(predicted == y_test)
from sklearn.metrics import roc_auc_score
# roc_auc = roc_auc_score(y_test, predicted)
roc_auc = roc_auc_score(y_test, probas[:,1])
print "ROC = ", roc_auc


# In[8]:

print "part 2"


# In[ ]:

###Model 2 - a) custom features

### build pipeline; fit train; predict on test
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        return X[self.vars]  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing
    
class StringExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        return X[self.vars].astype(str)  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing
    
class SubredditExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        return X[self.vars].apply(lambda x: ' '.join(x))  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing

##vectorizer arguments blah!
tokenizer=None#word_tokenize
# stop_words=nltk.corpus.stopwords.words("english")#None
ngram_range=(1, 1)
lowercase=True
max_features=10000
binary=False
dtype=np.float64
###create unigram vectorizer
subVect = CountVectorizer(decode_error="ignore",
#                                tokenizer=tokenizer,
#                                stop_words=stop_words,
                               ngram_range=ngram_range,
                               lowercase=lowercase,
                               binary=binary,
                               dtype=dtype,
                               max_features=max_features)

flairVect = CountVectorizer(decode_error="ignore",
#                                tokenizer=tokenizer,
#                                stop_words=stop_words,
                               ngram_range=ngram_range,
                               lowercase=lowercase,
                               binary=binary,
                               dtype=dtype,
                               max_features=100)

# load custom features for FeatureUnion
cols_act = [
'post_was_edited',
'requester_account_age_in_days_at_request',
'requester_account_age_in_days_at_retrieval',
'requester_days_since_first_post_on_raop_at_request',
'requester_days_since_first_post_on_raop_at_retrieval',
'requester_number_of_comments_at_request',
'requester_number_of_comments_at_retrieval',
'requester_number_of_comments_in_raop_at_request',
'requester_number_of_comments_in_raop_at_retrieval',
'requester_number_of_posts_at_request',
'requester_number_of_posts_at_retrieval',
'requester_number_of_posts_on_raop_at_request',
'requester_number_of_posts_on_raop_at_retrieval',
'requester_number_of_subreddits_at_request',
# 'requester_subreddits_at_request',
]

cols_rep = [
'number_of_downvotes_of_request_at_retrieval',
'number_of_upvotes_of_request_at_retrieval',
'requester_upvotes_minus_downvotes_at_request',
'requester_upvotes_minus_downvotes_at_retrieval',###this contains negative values
'requester_upvotes_plus_downvotes_at_request',
'requester_upvotes_plus_downvotes_at_retrieval',
### 'requester_user_flair',
]

get_flair = Pipeline([
                     ('getFlair', StringExtractor('requester_user_flair')),
                     ('counts', flairVect),
                    ])

get_subs = Pipeline([
                     ('getSubs', SubredditExtractor('requester_subreddits_at_request')),
                     ('counts', subVect),
                    ])
##feature union
features = []
features.append(('activity', ColumnExtractor(cols_act)))
features.append(('reputation', ColumnExtractor(cols_rep) ))
features.append(('flair', get_flair))
features.append(('subs', get_subs))

all_features = FeatureUnion(features)

linear_svc = svm.SVC(kernel='linear', probability=True)

###create pipeline
text_clf = Pipeline([
#                      ('getText', TextExtractor('request_text_edit_aware')),
                     ('all', all_features),
#                      ('tfidf', TfidfTransformer()),
#                     ('clf', MultinomialNB(alpha=0.1)),
                    ('clf', linear_svc),
                    ])

###fit training data
print "Fitting!"
text_clf = text_clf.fit(X_train, y_train)

### predict on test data
predicted = text_clf.predict(X_test)
np.mean(predicted == y_test)
# print "NO!"
probas = text_clf.predict_proba(X_test)
# print "YO!"
###get performance metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
accuracy = accuracy_score(y_test,predicted)
#         accuracy = np.mean(y_CVtest == predicted)
precision, recall, fscore, sup = precision_recall_fscore_support(y_test, predicted, average='binary', pos_label=0)
print precision, recall, fscore, accuracy# np.mean(predicted == y_test)
from sklearn.metrics import roc_auc_score
# roc_auc = roc_auc_score(y_test, predicted)
roc_auc = roc_auc_score(y_test, probas[:,1])
print "ROC = ", roc_auc
