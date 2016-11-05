
# coding: utf-8

# In[1]:

import pandas as pd
import random as rd
from IPython.display import display


# In[19]:

# filename="trnsact.csv"
# n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
# s = 100000 #desired sample size
# skip = sorted(rd.sample(xrange(1,n+1),n-s))
# pd.options.display.max_columns = None
# df = pd.read_csv(filename,skiprows=skip,header=None,names=['SKU','Store','Trannum','a','b','Date','c','Quantity','d','e','f','g','Mic','h'])


# In[43]:

iter_csv = pd.read_csv(filename, iterator=True, chunksize=1000,header=None, names=['SKU','Store','Trannum','a','b','Date','c','Quantity','d','e','f','g','Mic','h'])
df = pd.concat([chunk[chunk['Store'] ==4903] for chunk in iter_csv])


# In[44]:

print df.shape


# In[52]:

grouped = df.groupby(['Trannum','Date'])['SKU'].apply(list)


# In[53]:

grouped_frame = grouped.to_frame()


# In[54]:

ds = []
for index, row in grouped_frame.iterrows():
    ds.append(row['SKU'])


# In[120]:

print len(ds)


# In[56]:

def createC1(dataset):
    "Create a list of candidate item sets of size one."
    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    #frozenset because it will be a ket of a dictionary.
    return map(frozenset, c1)


def scanD(dataset, candidates, min_support):
    "Returns all candidates that meets a minimum support level"
    sscnt = {}
    for tid in dataset:
        for can in candidates:
            if can.issubset(tid):
                sscnt.setdefault(can, 0)
                sscnt[can] += 1

    num_items = float(len(dataset))
    retlist = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key] / num_items
        if support >= min_support:
            retlist.insert(0, key)
        support_data[key] = support
    return retlist, support_data


def aprioriGen(freq_sets, k):
    "Generate the joint transactions from candidate sets"
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(freq_sets[i] | freq_sets[j])
    return retList


def apriori(dataset, minsupport=0.05):
    "Generate a list of candidate item sets"
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, support_data = scanD(D, C1, minsupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minsupport)
        support_data.update(supK)
        L.append(Lk)
        k += 1

    return L, support_data


# In[114]:

L, support_data = apriori(ds, minsupport=0.001)


# In[122]:

print L


# In[86]:

print support_data


# In[62]:

def generateRules(L, support_data, min_confidence=0.7):
    """Create the association rules
    L: list of frequent item sets
    support_data: support data for those itemsets
    min_confidence: minimum confidence threshold
    """
    rules = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            print "freqSet", freqSet, 'H1', H1
            if (i > 1):
                rules_from_conseq(freqSet, H1, support_data, rules, min_confidence)
            else:
                calc_confidence(freqSet, H1, support_data, rules, min_confidence)
    return rules


def calc_confidence(freqSet, H, support_data, rules, min_confidence=0.7):
    "Evaluate the rule generated"
    pruned_H = []
    for conseq in H:
        conf = support_data[freqSet] / support_data[freqSet - conseq]
        if conf >= min_confidence:
            print freqSet - conseq, '--->', conseq, 'conf:', conf
            rules.append((freqSet - conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H


def rules_from_conseq(freqSet, H, support_data, rules, min_confidence=0.7):
    "Generate a set of candidate rules"
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calc_confidence(freqSet, Hmp1,  support_data, rules, min_confidence)
        if len(Hmp1) > 1:
            rules_from_conseq(freqSet, Hmp1, support_data, rules, min_confidence)


# In[125]:

print generateRules(L, support_data,min_confidence=0.65)


# In[113]:

print AR


# In[ ]:



