# conda install -c conda-forge textblob

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set

sms_data = pd.read_csv("C:/Training/Analytics/Naive_Bayes/Sms_data/sms_raw_NB.csv",encoding = "ISO-8859-1")


# cleaning data 
import re
stop_words = []
with open("C:/Training/Analytics/Naive_Bayes/Solved/stop.txt") as f:
    stop_words = f.read()


# splitting the entire string by giving separator as "\n" to get list of 
# all stop words
stop_words = stop_words.split("\n")


"this is awsome 1231312 $#%$# a i he yu nwj"

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

           
#Passing the data through the cleaning function

sms_data.text = sms_data.text.apply(cleaning_text)

# removing empty rows 
sms_data.shape
sms_data = sms_data.loc[sms_data.text != " ",:]


# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# TfidfTransformer
# Transform a count matrix to a normalized tf or tf-idf representation

# creating a matrix of token counts for the entire text document 

def split_into_words(i):
    return [word for word in i.split(" ")]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

sms_train,sms_test = train_test_split(sms_data,test_size=0.3)


# Preparing sms texts into word count matrix format 
sms_bow = CountVectorizer(analyzer=split_into_words).fit(sms_data.text)


# For all messages
all_sms_matrix = sms_bow.transform(sms_data.text)
all_sms_matrix.shape 
# For training messages
train_smss_matrix = sms_bow.transform(sms_train.text)
train_smss_matrix.shape 



# For testing messages
test_smss_matrix = sms_bow.transform(sms_test.text)
test_smss_matrix.shape # (1668,6661)

####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_smss_matrix,sms_train.type)
train_pred_m = classifier_mb.predict(train_smss_matrix)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) # 98%

test_pred_m = classifier_mb.predict(test_smss_matrix)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) # 96%

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_smss_matrix.toarray(),sms_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_smss_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==sms_train.type) # 90%

test_pred_g = classifier_gb.predict(test_smss_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==sms_test.type) # 83%


#########################################################3

# Learning Term weighting and normalizing on entire smss
tfidf_transformer = TfidfTransformer().fit(all_sms_matrix)

# Preparing TFIDF for train smss
train_tfidf = tfidf_transformer.transform(train_smss_matrix)

train_tfidf.shape # (3891, 6661)

# Preparing TFIDF for test smss
test_tfidf = tfidf_transformer.transform(test_smss_matrix)

test_tfidf.shape #  (1668, 6661)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,sms_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) # 96%

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) # 95%

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),sms_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==sms_train.type) # 91%
test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==sms_test.type) # 85%

# inplace of tfidf we can also use train_smss_matrix and test_smss_matrix instead of term inverse document frequency matrix 

