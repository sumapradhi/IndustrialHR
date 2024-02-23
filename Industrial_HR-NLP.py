#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.linear_model import LogisticRegression
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# Replace the deprecated function with the recommended one
import tensorflow as tf

# Old function
# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import time
import os

import matplotlib.pyplot as plt 
import seaborn as sns


# In[66]:


import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(os.listdir(r"C:\Users\sumat\OneDrive\Documents\IND_HR"))


# Table of Contents: Dataset Preparation: The first step is the Dataset Preparation step which includes the process of loading a dataset and performing basic pre-processing. The dataset is then splitted into train and validation sets. Feature Engineering: The next step is the Feature Engineering in which the raw dataset is transformed into flat features which can be used in a machine learning model. This step also includes the process of creating new features from the existing data. Model Training: The final step is the Model Building step in which a machine learning model is trained on a labelled dataset. Improve Performance of Text Classifier: In this article, we will also look at the different ways to improve the performance of text classifiers.

# # Loading the Training Data

# In[114]:


train = pd.read_csv(r"C:\Users\sumat\OneDrive\Documents\IND_HR\df_final.csv")
train.head(2)


# In[135]:


#Identifying the first Tag for all rows
first_tag = []
for str in train['nic_name']:
    first_tag.append(str.split(';')[0])


# In[136]:


#Number of Records in First_Tag
len(first_tag)


# In[137]:


#Adding a new feature into the training dataset
train['First_Tag'] = first_tag
train.head(2)


# # Unique Value Count for each feature

# In[138]:


#Unique Values for Geographic Locations
train['states'].unique()


# In[139]:


train['districts'].unique()


# In[140]:


#Unique Values for CompanyType
train['division'].unique()


# In[141]:


#Abstracting States,NIC Name,Category from train into a new dataset
train_ana = train[['states','nic_name','First_Tag']]
train_ana.head(2)


# We will now carry out all the Text Preprocessing steps on the 'nic_name' field converting all letters to lower or upper case converting numbers into words or removing numbers removing punctuations, accent marks and other diacritics removing white spaces expanding abbreviations removing stop words, sparse terms, and particular words text canonicalization

# # Convert text to lowercase

# In[142]:


import pandas as pd

train_ana['Tidy_Desc'] = train_ana['nic_name'].str.lower()

train_ana.head(1)


# In[143]:


import re
#Function to remove any additional special characters, if needed.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[125]:


#Facts: Translate function has been changed from Python 2.x to Python 3.x
#It now takes only one output
pattern = r'\d'  # Remove all digits from the text
train_ana['Tidy_Desc'] = train_ana['Tidy_Desc'].apply(lambda x: remove_pattern(x, pattern))


# In[144]:


train_ana.head(1)


# # Removing punctuations, accent marks and other diacritics

# In[145]:


#[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]
train_ana.Tidy_Desc=train_ana.Tidy_Desc.apply(lambda x: x.translate({ord(c):'' for c in "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]"}))


# In[146]:


train_ana.head(1)


# # Remove whitespaces

# In[147]:


train_ana.Tidy_Desc=train_ana.Tidy_Desc.apply(lambda x: x.strip())


# In[148]:


train_ana.head(1)


# Tokenization
# Tokenization is the process of splitting the given text into smaller pieces called tokens.

# # StopWord Removal + Tokenization

# In[149]:


stop_words = stopwords.words('english')
from nltk.tokenize import word_tokenize


# In[150]:


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# In[151]:


clean_sentences  = [remove_stopwords(r.split()) for r in train_ana['Tidy_Desc']]


# In[152]:


train_ana['Tidy_Desc'] = clean_sentences


# In[153]:


train_ana.head(3)


# In[72]:


#Tokenization
tokens = train_ana['Tidy_Desc'].apply(lambda x: x.split())
#Now that we have the removed StopWords and tokenized the nic_name
#We will now subject the tokenized version to removing stop words, sparse terms, and particular words
#In some cases, it’s necessary to remove sparse terms or particular words from texts. 
#This task can be done using stop words removal techniques considering that any group of words can be chosen as the stop words.


# In[154]:


#Stemming
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
Stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming


# Now let’s stitch these tokens back together.

# In[155]:


for i in range(len(Stemmed_tokens)):
    Stemmed_tokens[i] = ' '.join(Stemmed_tokens[i])

train_ana['Tidy_Desc_Stemmed'] = Stemmed_tokens


# In[156]:


train_ana.head(3)


# In[157]:


import nltk
nltk.download('wordnet')


# In[158]:


#Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
Lemmatized_tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(i) for i in x]) # Lemmatizing


# In[159]:


for i in range(len(Lemmatized_tokens)):
    Lemmatized_tokens[i] = ' '.join(Lemmatized_tokens[i])

train_ana['Tidy_Desc_Lemma'] = Lemmatized_tokens


# In[160]:


train_ana.head(3)


# In[161]:


#Before going for Count Vectorization as Feature
#We would like to check the type of Count of Type of Tags

train_ana['First_Tag'].value_counts().head(5)


# Count Vectors as features

# In[162]:


#set(train_ana['First_Tag'])
#Selecting the first 10 labels
T10Tag = train_ana['First_Tag'].value_counts().index.tolist()
T10Tag = T10Tag[:10]
T10Tag


# In[163]:


train_10 = train_ana[train_ana['First_Tag'].isin(T10Tag)]
train_10.shape


# In[164]:


set(train_10['First_Tag'])


# In[165]:


from collections import Counter
Counter(train_10['First_Tag'])


# In[166]:


train_10.columns.tolist()


# In[167]:


train_10_ana = train_10[['Tidy_Desc_Lemma','First_Tag']]


# In[168]:


#pre-processing
import re 
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", "", string)    
    string = re.sub(r"\r", "", string) 
    string = re.sub(r"[0-9]", "digit", string)
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


# In[169]:


#train test split
from sklearn.model_selection import train_test_split
X = []
for i in range(train_10_ana.shape[0]):
    X.append(clean_str(train_10_ana.iloc[i][0]))
y = np.array(train_10_ana["First_Tag"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# In[170]:


#feature engineering and model selection
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[172]:


#pipeline of feature engineering and model
model = Pipeline([('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])


# In[95]:


#paramater selection
from sklearn.model_selection import GridSearchCV
parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
               'tfidf__use_idf': (True, False)}


# In[96]:


gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X, y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)


# In[97]:


#preparing the final pipeline using the selected parameters
model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])


# In[98]:


#fit model with training data
model.fit(X_train, y_train)


# In[99]:


#evaluation on test data
pred = model.predict(X_test)


# In[100]:


model.classes_


# In[101]:


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred, y_test)


# In[102]:


accuracy_score(y_test, pred)


# # Using k-fold cross-validation

# In[103]:


from sklearn.model_selection import cross_val_score


# In[174]:


import re

# Define a function to clean and preprocess the category names
def clean_category_name(category):
    # Convert to lowercase
    category = category.lower()
    # Remove any non-alphanumeric characters except spaces
    category = re.sub(r'[^a-zA-Z\s]', '', category)
    # Remove extra spaces
    category = category.strip()
    # Remove common words like 'of'
    stop_words = set(stopwords.words('english'))
    category = ' '.join([word for word in category.split() if word not in stop_words])
    return category

# Example usage:
categories = {
    'Construction of buildings': 1353,
    'Activities of households as employers of domestic personnel': 1218,
    'Manufacture of furniture': 1146,
    'Water collection, treatment and supply': 1104,
    'Veterinary activities': 1029
}

# Clean and preprocess category names
cleaned_categories = {clean_category_name(category): count for category, count in categories.items()}

print(cleaned_categories)


# In[176]:


# 1. Split Data into Train and Test Sets (already done)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# 2. Define the Model (using the existing pipeline)
model = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))
])


# In[177]:


# 3. Cross-Validation
# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')


# In[178]:


# 4. Average Accuracy
average_accuracy = cv_scores.mean()


# In[179]:


# Print the average accuracy
print("Average Accuracy:", average_accuracy)


# # Checking using Random Forest Classifier

# In[180]:


from sklearn.ensemble import RandomForestClassifier

import re

# Define a function to clean and preprocess the category names
def clean_category_name(category):
    # Convert to lowercase
    category = category.lower()
    # Remove any non-alphanumeric characters except spaces
    category = re.sub(r'[^a-zA-Z\s]', '', category)
    # Remove extra spaces
    category = category.strip()
    # Remove common words like 'of'
    stop_words = set(stopwords.words('english'))
    category = ' '.join([word for word in category.split() if word not in stop_words])
    return category

# Example usage:
categories = {
    'Construction of buildings': 1353,
    'Activities of households as employers of domestic personnel': 1218,
    'Manufacture of furniture': 1146,
    'Water collection, treatment and supply': 1104,
    'Veterinary activities': 1029
}

# Clean and preprocess category names
cleaned_categories = {clean_category_name(category): count for category, count in categories.items()}

# Define class weights based on the class distribution
class_weights = {'Manufacture of furniture': 1,
                 'Water collection, treatment and supply': 1,
                 'Construction of buildings': 1,
                 'Veterinary activities': 1,
                 'Primary education': 1,
                 'Other human health activities': 1,
                 'Activities of households as employers of domestic personnel': 1,
                 'Blank': 0.5,  # Adjust the weight for the "Blank" class
                 'Incomplete description/ Wrongly Classifed': 0.5,  # Adjust the weight for the "Incomplete description/Wrongly Classified" class
                 'Manufacture of tobacco products': 1}

# Define the pipeline with RandomForestClassifier
model_rf = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights))
])

# Fit the model with training data
model_rf.fit(X_train, y_train)

# Evaluate on test data
pred_rf = model_rf.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, pred_rf)
print("Accuracy with Random Forest Classifier after adjusting for class imbalance:", accuracy_rf)


# In[ ]:




