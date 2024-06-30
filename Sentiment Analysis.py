#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
path = r'C:\Users\bappa\Downloads\imdb dataset\IMDB Dataset.csv'
df = pd.read_csv(path)

df.head()


# In[3]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
docs = np.array(['I am Bappa Aditya ' 'I am not a good boy ' 'Bappa studying MCA'])
bag = vect.fit_transform(docs)


# In[4]:


print(vect.vocabulary_)


# In[5]:


print(bag.toarray())


# In[6]:


from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision = 2)
tfidf = TfidfTransformer(use_idf = True, norm='l2', smooth_idf = True)
print(tfidf.fit_transform(bag).toarray())


# In[7]:


import nltk
nltk.download('stopwords')


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(use_idf = True, norm='l2', smooth_idf = True)
y = df.sentiment.values
x = tfidf.fit_transform(df['review'].values.astype('U'))


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)


# ## Pickle is used to store the data of model

# In[ ]:


import pickle
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=300).fit(x_train,y_train)

saved_model = open('saved_model.sav','wb')
pickle.dump(clf,saved_model)
saved_model.close()


# In[ ]:


filename = 'saved_model.sav'
saved_clf = pickle.load(open(filename, 'rb'))
saved_clf.score(x_test,y_test)


# ### Function to preprocess and predict sentiment of new text data

# In[ ]:


def preprocess_and_predict(new_texts):
    # Transform the new texts using the same TF-IDF vectorizer
    new_texts_tfidf = tfidf.transform(new_texts)

    # Predict the sentiment using the loaded model
    predictions = saved_clf.predict(new_texts_tfidf)
    return predictions


# ### Predict the sentiment of the new texts

# In[ ]:


new_texts = [
    "It was perfect, perfect , down to the last minute details",
    "It is waste of money"
]

predictions = preprocess_and_predict(new_texts)

print(f"Predictions: {predictions}")


# In[ ]:




