#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


credits_df = pd.read_csv("~/Desktop/credits.csv")
movies_df = pd.read_csv("~/Desktop/movies.csv")


# In[3]:


credits_df


# In[4]:


movies_df


# In[5]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[6]:


movies_df


# In[7]:


credits_df


# In[8]:


credits_df.head()


# In[9]:


movies_df.head()


# In[10]:


movies_df=movies_df.merge(credits_df,on='title')


# In[11]:


movies_df


# In[12]:


movies_df.shape


# In[13]:


movies_df.info()


# In[14]:


movies_df=movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[15]:


movies_df.head()


# In[16]:


movies_df.info()


# In[17]:


movies_df.isnull().sum()


# In[18]:


movies_df.dropna(inplace=True)


# In[19]:


movies_df.duplicated().sum()


# In[20]:


movies_df.iloc[0].genres


# In[21]:


#improting abstract syntax tree
import ast


# In[22]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[23]:


movies_df['genres']= movies_df['genres'].apply(convert)
movies_df['keywords']= movies_df['keywords'].apply(convert)


# In[24]:


movies_df.head()


# In[25]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
        return L


# In[26]:


movies_df['cast']=movies_df['cast'].apply(convert3)


# In[27]:


movies_df.head()


# In[28]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L


# In[29]:


movies_df['crew']=movies_df['crew'].apply(fetch_director)


# In[30]:


movies_df


# In[31]:


movies_df['overview'][0]


# In[32]:


movies_df['overview']= movies_df['overview'].apply(lambda x:x.split())


# In[35]:


movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
# Modify lambda function to handle None values
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else [])
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else [])


# In[36]:


movies_df


# In[37]:


movies_df['tags']=movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']+movies_df['crew']



# In[38]:


movies_df


# In[39]:


new_df=movies_df[['movie_id','title','tags']]


# In[40]:


new_df


# In[41]:


new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))


# In[42]:


new_df


# In[43]:


new_df['tags'][0]


# In[44]:


new_df['tags']=new_df['tags'].apply(lambda X:X.lower())


# In[45]:


new_df.head()


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000,stop_words='english')


# In[48]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[49]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[50]:


vectors[0]


# In[52]:


len(cv.get_feature_names_out())


# In[53]:


import nltk


# In[54]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[55]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[56]:


new_df['tags']=new_df['tags'].apply(stem)


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity


# In[58]:


cosine_similarity(vectors)


# In[59]:


cosine_similarity(vectors).shape


# In[60]:


similarity = cosine_similarity(vectors)


# In[61]:


similarity[0]


# In[63]:


similarity[0].shape


# In[64]:


sorted(list(enumerate(similarity[0])),reverse = True ,key=lambda x:x[1])[1:6]


# In[67]:


def recommend (movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list =sorted(list(enumerate(distances)),reverse = True ,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[75]:


recommend ('Interstellar')


# In[ ]:




