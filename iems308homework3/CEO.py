
# coding: utf-8

# In[1]:

#Thomas Cho
#IEMS 308
#Goal: To extract entities and build a classification model
import pandas as pd
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import io
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split 


# In[2]:

#Initialize panda DataFrame
df = pd.DataFrame()


# In[3]:

#Read each text file
#split text into sentences using regular expression. Logic--> sentence ends in !.? and . is not between numbers or letters
#append each sentence as row
for s in ('2013','2014'):
    path = '/Users/Tcho187/Documents/Senior_year/iems308text/%s' % (s)
    for file in os.listdir(path):
        with io.open(os.path.join(path,file),'r',encoding='utf-8',errors='ignore') as infile:
            txt = infile.read()
        sentences = re.split(r'!|\?|\.{3}|\.\D|\.\s', txt)
        df = df.append(sentences,ignore_index=True)


# In[4]:

#Give text a header
df.columns=['text']


# In[5]:

''' RE part
Logic: Find first (& last name) after 'Chief Executive Officer' and its variations. Grab the first instance --> I assume 
that most, if not all sentences will contain at most 1 CEO.
   ''' 
def preprocessor(text):
    text = re.search(r'(?:Chief Executive Officer|CEO|ceo|Chief executive officer)(\s[A-Z]\w+(?:\s[A-Z]\w+|\b))',text)
    if not text:
        text=""
    else:
        text= text.group(1)
    return text


# In[6]:

#clean/ preprocessing

#removes plural words and returns the stem
def stemmer(word):
    [(stem,end)] = re.findall('^(.*ss|.*?)(s)?$',word)
    return stem

#removes non-letter, non-number characters with some exception (. %)
#converts all characters into lower case
#delete stop words 
def clean(text):
    str_list=[]
    letters = re.sub(r'[^A-Za-z0-9%.]',' ',text)
    lower_case = letters.lower().split()
    stop_list = set(('a','an','are','and','are','as','at','be','by','can','do','for','from','has','i','he','in','is','is','its','it','of','on',                 'that','the','to','was','were','will','with','would','yet','you','or',))
    important_words = [word for word in lower_case if not word in stop_list]
    
    important_words= (" ".join( important_words ))
    important_words=important_words.lower().split()
    for word in important_words:
        str_list.append(stemmer(word))
    return " ".join(str_list)


# In[7]:

#label is either "" or CEO name 
df['label']=df['text'].apply(preprocessor)


# In[8]:

#I only care about sentences with CEO names 
df2 = df[df.label != ""]


# In[9]:

label_list=df2['label'].tolist()


# In[10]:

# preprocess each sentence and create new DataFrame
temp_list=[]
for i, row in df2.iterrows():
    cleaned=clean(row['text'])
    temp_list.append(cleaned)
df3= pd.DataFrame({'text':temp_list,'label':label_list})


# In[ ]:

##Alternative solution discussed in document. Didn't really
# temp_string=""
# for i,row in df3.iterrows():
#     if row['label']!="":
#         temp_string=row['label']
#     else:
#         row['label']=temp_string
    


# In[ ]:




# In[ ]:




# In[ ]:

df3 = df3[df3.label != ""]


# In[11]:




# In[12]:

#I'm restricting my features to 1000 max 
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features=1000) 


# In[13]:

#train/split 80/20
msg_train, msg_test, label_train, label_test =     train_test_split(df3['text'], df3['label'], test_size=0.2)


# In[14]:

#tf-idf 
train_data_features_counts = vectorizer.fit_transform(msg_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(train_data_features_counts)
train_data_features=train_data_features_counts.toarray()


# In[15]:

#Naive Bayes Classifier model
train_model = MultinomialNB().fit(train_data_features,label_train )


# In[16]:

#test 
test_data_features_count = vectorizer.transform(msg_test)
X_new_tfidf = tfidf_transformer.transform(test_data_features_count)
test_data_features=test_data_features_count.toarray()


# In[17]:

all_predictions = train_model.predict(test_data_features)


# In[18]:

print all_predictions


# In[19]:

print 'accuracy', accuracy_score(label_test, all_predictions)


# In[ ]:

pd.set_option('expand_frame_repr', False)

print classification_report(label_test, all_predictions)


# In[20]:

msg_train.shape,msg_test.shape


# In[ ]:

ceo_names=df3['label'].drop_duplicates().to_csv('./ceo_names.csv')


# In[ ]:



