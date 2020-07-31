#!/usr/bin/env python
# coding: utf-8

# In[30]:


import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
import json
import time
from textblob import TextBlob
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from scipy import spatial
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json
import spacy
en_nlp = spacy.load('en')
import sys


# In[29]:


#!pip install rake-nltk
from textblob.en.np_extractors import FastNPExtractor
#!python -m textblob.download_corpora
npe=FastNPExtractor()


# In[4]:


infersent=torch.load("InferSent/mymodel1.pkl")


# In[5]:


data = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)
train = pd.read_json("data/train-v1.1.json")


# In[6]:


#data["question"][2800]


# In[7]:


ques=[]
ques.append(' '.join(sys.argv[1:]))


# In[8]:


keywords=npe.extract(ques[0])
li=[]
for i in keywords:
  li.append(i.lower())
keywords=li


# In[9]:


l=keywords


# In[143]:


'''with open("/content/drive/My Drive/SQuAD/data/dict_embeddings1.pickle", "rb") as f:
    d1 = pickle.load(f)
with open("/content/drive/My Drive/SQuAD/data/dict_embeddings2.pickle", "rb") as f:
    d2 = pickle.load(f)
dict_emb = dict(d1)
dict_emb.update(d2)
del d1,d2'''


# In[10]:


ques_emb=infersent.encode(ques)
#ques_emb


# In[11]:


keywords=[]
for x in l:
  keywords.extend(x.split(" "))


# In[12]:


contexts=[]
titles=[]
for i in range(train.shape[0]):
  s=""
  topic = train.iloc[i,0]['paragraphs']
  for j in topic:
    s+=j['context']
  contexts.append(s)
  titles.append(train.iloc[i,0]['title'])


# In[13]:


paragraphs = pd.DataFrame(list(zip(contexts, titles)), columns =['Context', 'Title']) 


# In[14]:


li=[]
for i in paragraphs["Title"]:
  i=i.replace("_"," ")
  i=i.lower()
  li.append(i)
paragraphs['Title']=li


# In[15]:


vect=TfidfVectorizer(vocabulary=keywords,strip_accents='unicode')
X = vect.fit_transform(paragraphs["Context"])
Y = vect.fit_transform(paragraphs["Title"])
dfx = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
dfy = pd.DataFrame(Y.toarray(), columns = vect.get_feature_names())
dfx["sum"]=dfx.sum(axis=1)
dfy["sum"]=dfy.sum(axis=1)
dfx["idx"]=dfx.index
dfy["idx"]=dfy.index
dfx=dfx.sort_values(by=["sum"],ascending=False)
dfy=dfy.sort_values(by=["sum"],ascending=False)
dfx["sum"]+=dfy["sum"]
dfx=dfx.sort_values(by=["sum"],ascending=False)


# In[16]:


relevant=[]
for i in range(0,5):
  if dfx.iloc[i]["sum"]>0:
    relevant.append(dfx.iloc[i]["idx"])
  else:
    break


# In[17]:


l=[]
for i in relevant:
  i=int(i)
  l.append(i)
relevant=l


# In[18]:


#relevant


# In[19]:


from rake_nltk import Rake
r=Rake()
r.extract_keywords_from_text(ques[0])
l=r.get_ranked_phrases()
keywords=[]
for x in l:
  keywords.extend(x.split(" "))
vect=TfidfVectorizer(vocabulary=keywords,strip_accents='unicode')


# In[20]:


def process_data(train):
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    train['sent_emb'] = train['sentences'].apply(lambda x: ["haha" if 0 else infersent.encode([item])[0] for item in x])        
    return train   


# In[21]:


def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item,ques_emb))
    return li


# In[22]:


def pred_idx(distances):
    return np.argmin(distances)   


# In[23]:


def predictions(train):    
    train["cosine_sim"] = train.apply(cosine_sim, axis = 1)    
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))    
    return train    


# In[24]:


findat={}
findat["sentence"]=[]
findat["weight"]=[]
for i in relevant[0:3]:
  relcont=[]
  topic = train.iloc[i,0]['paragraphs']
  for j in topic:
    relcont.append(j['context'])
  tfid=vect.fit_transform(relcont)
  mat = pd.DataFrame(tfid.toarray(), columns = vect.get_feature_names())
  mat["sum"]=mat.sum(axis=1)
  mat.sort_values(by=["sum"],ascending=False)
  mat=mat.iloc[0:10]
  cont=[]
  weights=[]
  for j in mat.index:
    cont.append(relcont[j])
    weights.append(mat.loc[j]["sum"])
  cont=pd.DataFrame(list(zip(cont,weights)),columns=["context","weight"])
  cont=process_data(cont)
  predicted=predictions(cont)
  for j in range(0,10):
    predicted["cosine_sim"][j]=predicted["cosine_sim"][j][predicted["pred_idx_cos"][j]]/predicted["weight"][j]
  num=np.argmin(predicted["cosine_sim"])
  findat["sentence"].append(predicted.iloc[num]["sentences"][predicted.iloc[num]["pred_idx_cos"]])
  findat["weight"].append(predicted["cosine_sim"][num]/dfx["sum"][i])


# In[25]:


findf=pd.DataFrame(findat,columns=["sentence","weight"])


# In[26]:


findf=findf.sort_values(by=["weight"])


# In[27]:


print(findf.iloc[0]["sentence"])


# In[ ]:




