#!/usr/bin/env python
# coding: utf-8




import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer,PorterStemmer,SnowballStemmer, WordNetLemmatizer
import os
import glob
import  numpy as np
import csv
import pickle
from functools import reduce
import itertools
from copy import deepcopy
import math
import time
from sys import getsizeof
from IPython.display import clear_output as CP
from termcolor import colored as col
from dataclasses import dataclass
color={"OR":"blue","XOR":"red"}   #for priniting rules
@dataclass
class Point:   ##to storing rules like a xor b then  Point(x=a,y=b,z="xor")
    x: list
    y: list
    z: str



def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        new_word=re.sub(r"_+"," ",new_word)
        new_words.extend(nltk.word_tokenize(new_word))
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words
def remove_numbers(words):
    return re.sub(r'\d+', '', words)
def removelinks(words):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    return re.sub(regex,"",words)

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words
def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return (stems,lemmas)

#######################
def process(filepath):
    var=0
    txt=filepath
#     with open(filepath,'r',encoding = "ISO-8859-1") as f: # write code-------------------------------
#         for lineno,line in enumerate(f.readlines(),1):    # to read---------------------------------
#             if var:                                       # data------------------------------------
#                 txt+=line
#             else:
#                 if line.startswith("Abstract"):
#                     var=1
    txt=removelinks(txt)
    txt=remove_numbers(txt)
    words = nltk.word_tokenize(txt)
    words = normalize(words)
    words.append(word1) ####################appending context if somehow missed
    words.append(word2)
    return stem_and_lemmatize(words)
def tfidf(m,stemDict,stems):
    print("inside tfidf")
    print("frequency counting.....")
    docFreq={}
    for i in stemDict:
        docFreq[i]=0
    ll=len(stems)
    for idx,data in enumerate(stems):
        print("{}/{}".format(idx,ll))
        CP(wait=True)
        for i in stemDict:
            if i in data:
                docFreq[i]+=1

    print("frequency counting done:-)")
    tfidf={}
    l=len(stems)
    take=int(len(stemDict)*m/100)
    print("new dict will have {} words".format(take))
#     ch=str(input("want more or less words: enter \"yes\" or \"no\""))
#     if ch=='yes':
#            take=int(input("enter number of words you want: "))
#     else:
#            pass
    print("calculating tfidf")
    ll=len(stemDict)
    for idx,i in enumerate(stemDict):
        print("{}/{}".format(idx,ll))
        CP(wait=True)
        val=0
        for docs in stems:
            x=docs.count(i)
            x=x*(round(math.log2(l/docFreq[i]),4))
            if val<x:
                val=x
        tfidf[i]=val
    print("calculating tfidf done")
    tfidf={a:b for a,b in sorted(tfidf.items(),reverse=True,key=lambda item: item[1])}
    keys=list(tfidf.keys())

    with open("./procDatat/tfidf.txt", "wb") as fp:  #<---------------------ADDRESS
        pickle.dump(keys, fp)
    return keys[0:take]
def transGen(stems,newDict,revDict): #<-------------some modification needed, deleted repeated stems
    transactions=[]
    for docs in stems:
        temp=[]
        for words in docs:
            if words in newDict:
                temp.append(revDict[words])
        transactions.append(list(set(temp)))
#     print("transactions",len(transactions[0]))
    return transactions
def revTrans(transactions):
    #reverse transactions( given an item no. it will give all the transactions which include the given item)
    item_trans={}   # contains all the transaction corresponding to every item
    for i,j in enumerate(transactions):
        for it in j:
            if it not in item_trans:
                item_trans[it]=[i]
            else:
                item_trans[it].append(i)
    return item_trans
def readRaw(filepaths=None):
    stemDict=[]
    lemDict=[]
    trans=[]
    if 0:                  ########### change in the code this was written by thinking that CSV
                                    ##                               has already processed data
        with open(filepaths,'r') as csv_file: #Opens the file in read mode
            csv_reader = csv.reader(csv_file) # Making use of reader method for reading the file
        for line in csv_reader: #Iterate through the loop to read line by line
            trans.append(line)
        for i in trans:
            stemDict.extend(i)
            stemDict=list(set(stemDict))
    else:
        textData=[]
        with open(filepaths[0],'r') as csv_file: #Opens the file in read mode
            csv_reader = csv.reader(csv_file)
            for idx,tex in enumerate(csv_reader):
                if idx==0:
                    continue
                if idx==readData:
                    break
                textData.append(tex[6])
        ll=len(textData)
        for idx,tex in enumerate(textData):
            print("{}/{}".format(idx,ll))
            CP(wait=True)
            stems,lemmas  = process(tex)   # 6th column has text
            stemDict.extend(stems)
            stemDict=list(set(stemDict))
            lemDict.extend(lemmas)
            lemDict=list(set(lemDict))
            trans.append(stems)
        ##removing words of one character
        dictlen=len(stemDict)-1
        while(dictlen>=0):
            if(len(stemDict[dictlen])<=1):
                del stemDict[dictlen]
            dictlen-=1


    with open("./procDatat/stemdict.txt", "wb") as fp:  #<---------------------ADDRESS
        pickle.dump(stemDict, fp)

    with open("./procDatat/data.txt", "wb") as fp:  #<---------------------ADDRESS
        pickle.dump(trans, fp)
    with open("./procDatat/lemDict.txt", "wb") as fp:  #<---------------------ADDRESS
        pickle.dump(lemDict, fp)
    print("stemDict and trans has been done inside rawData functions. next is tfidf")
    return (trans,stemDict)


#--------------------------------------------
#some useful functions
# gives the transactions which contains all the input items of "itemsList"

def commonTrans(itemsList,itemTrans):
    if len(itemsList)==1:
        return list(itemTrans[itemsList[0]])
    tran_list=[]
    for i in itemsList:
        tran_list.append(itemTrans[i].copy())
    a=reduce(lambda x,y:x&y,(set(l) for l in tran_list))
    return list(a)
# check if pair of passed sets is fullfiling the requirement
def pairLen(a,b,size,minno,itemTrans):
    temp=a.copy()
    temp.extend(b)
    temp=list(set(temp))
    temp.sort()
    var=len(commonTrans(temp,itemTrans))
    if len(temp)==size and var>=minno:
        return temp
    else:
        return None
# returns sets if given size which satisfies the condition("minno" transactions)
def mul_pair(size,minno,sets,itemset,itemTrans): #reqconf is no which make a set rule if transaction exceeds that
    temp=[]
    if size==1:
        temp.extend([[i] for i in itemset if len(itemTrans[i])>=minno])
    else:
        templist=deepcopy(sets[size-1])
#         print("elements in {}th set are {}".format(size-1,len(templist)))
        n=1
        for i in itertools.combinations(sets[size-1],2):
            var=pairLen(i[0],i[1],size,minno,itemTrans)
            if var:
                if var not in temp:
                    temp.append(deepcopy(var))
                if i[0] in templist:
                    templist.remove(i[0])
                if i[1] in templist:
                    templist.remove(i[1])
            if n%5000 ==0:
                print("{} combination done".format(n))
#                 pass
            n+=1
        sets[size-1]=templist
    return temp
# returns valid sets of all size which satisfies the condition
def freq_itemset(minno,itemset,itemTrans,sizeLimit=1):
    sets={}
    for i in range(1,sizeLimit+1):
        sets[i]=[]
        var=mul_pair(i,minno,sets,itemset,itemTrans)
        sets[i].extend(deepcopy(var))

        if len(sets[i])==0:
            del sets[i]
            break
#         print("size {} is done".format(i))
#         print(sets[i])
    return sets
# returns transactions which contains "dset"
def dcommonTrans(dset,itemTrans): #drule is list of list
    if len(dset)==1:
        return commonTrans(dset[0],itemTrans)
    temp=[]
    for i in dset[:-1]:
        temp.extend(commonTrans(i,itemTrans))
    temp=list(set(temp))
    return temp
# returns the value of parameter which decides the best pair
def bestpairmetric(dset1,dset2,itemTrans):
    var1=dcommonTrans(dset1,itemTrans)
    var2=dcommonTrans(dset2,itemTrans)
    var=var1.copy()
    var.extend(var2)
    var=list(set(var))
    return len(var)/(len(var1)+len(var2))
#generate d-rules
def rules(k,minconf,context,item_trans,trans,maxSupp,sizelimit,xorth):
    drules1=[]
    newTrans=commonTrans(context,item_trans) #gives context transactions
    print("length of newTrans is {}".format(len(newTrans)))
#     print("total context transactions:{}".format(len(newTrans)))
    minVal=int(minconf*len(newTrans)/(100)) #<<<<<--------------if satisfy this make direct rule
#     print("intersection value for min confidence is {}".format(minVal))
    psiVal=int(minVal/k)
    if psiVal<=1 or minVal<=1:
        return None
#     print("psi-intersection value is {}".format(psiVal))
    newItemset=[]
    for i in newTrans:
        newItemset.extend(trans[i])
    #     print(trans[i])
    newItemset=list(set(newItemset))
    #remove context elements
    for i in context:
        newItemset.remove(i)
    print("length of newitemset is {}".format(len(newItemset)))
#     print("no. of items in context trans are {}".format(len(newItemset)))
    newItemTrans ={}                # dict key(item id)--->val(transactions)
    for i in newItemset:
        newItemTrans[i]=[]
        for j in newTrans:
            if i in trans[j]:
                newItemTrans[i].append(j)
    maxTransAllowed=int(len(newTrans)*maxSupp/100)
    tempIndx=[]
    for i,j in enumerate(newItemset):
        if len(newItemTrans[j])>maxTransAllowed:
            tempIndx.append(j)
    for i in tempIndx:
        newItemset.remove(i)
    largeItemsets=freq_itemset(minVal,newItemset,newItemTrans,sizelimit)

#     print(largeItemsets)
    for ke,va in largeItemsets.items():
        for r in va:
            drules1.append([r]) #[{a,b,c}]
#     print("drule1: ",drules1)
    delItems=[]
    for key,value in largeItemsets.items():
        if len(value)!=0:
            for i in value:
                delItems.extend(i)
    delItems=list(set(delItems))
    for i in delItems:
        newItemset.remove(i)
    if len(newItemset)==0:
        print("error here")
        return
    lc=freq_itemset(psiVal,newItemset,newItemTrans,sizelimit)
#     print(lc)
    il=[]
    for _,value in lc.items():
        for i in value:
            il.append([i])
#     print("length of large itemset is {}".format(len(il)))
    drule2=[]
    xth=xorth     #########################################threshold for xor
    while len(il)!=0:
        roVal=-1
        obj=None
        temprule=[]
        if len(il)==1:
            if len(dcommonTrans(il[0],newItemTrans))>=minVal and len(il[0])<=k:
                drule2.append(il[0])
            else:
                del il[0]
                continue
        for i in itertools.combinations(il,2):
            val=bestpairmetric(i[0],i[1],newItemTrans)
            if val>=roVal:
                roVal=val
                temprule=deepcopy(i)
#         print(roVal)
        if len(temprule[0])==1 and len(temprule[1])==1:
            if roVal>=xth:
                obj=Point(temprule[0][0],temprule[1][0],"xor")
            else:
                obj=Point(temprule[0][0],temprule[1][0],"or")
        elif len(temprule[0])!=1 and len(temprule[1])==1:
            if roVal>=xth:
                obj=Point(temprule[0][-1],temprule[1][0],"xor")
            else:
                obj=Point(temprule[0][-1],temprule[1][0],"or")
        elif len(temprule[0])==1 and len(temprule[1])!=1:
            if roVal>=xth:
                obj=Point(temprule[0][0],temprule[1][-1],"xor")
            else:
                obj=Point(temprule[0][0],temprule[1][-1],"or")
        else:
            if roVal>=xth:
                obj=Point(temprule[0][-1],temprule[1][-1],"xor")
            else:
                obj=Point(temprule[0][-1],temprule[1][-1],"or")



        tempdrule=deepcopy(temprule[0])
        if type(tempdrule[-1])!=list:
            del tempdrule[-1]
        tempdrule.extend(temprule[1])
        if type(tempdrule[-1])!=list:
            del tempdrule[-1]
        tempdrule.append(obj)



        il.remove(temprule[0])
        il.remove(temprule[1])
        if len(dcommonTrans(tempdrule,newItemTrans))>=minVal and len(tempdrule)-1<=k:
            drule2.append(tempdrule[-1])
        else:
            if len(tempdrule)-1<k:
                il.append(tempdrule)
#         print("current no. of items in dsets {}".format(len(il)))
#         print("no. of rules generated {}".format(len(drule2)))
    return [context,drules1,drule2]
def printContext(context,items):
    print("CONTEXT->> ( {} ".format(items[context[0]]),end="")
    for i1,e1 in enumerate(context):
        if i1!=0:
            print("and {} ".format(items[e1]),end="")
    print(")",end="")
    print("")
def printRule(drule,items):
    for i1,e1 in enumerate(drule):
        print("{}. ".format(i1+1), end="")
        for i2,e2 in enumerate(e1):
            if i2!=0:
                print(" OR ",end="")
            print("( {} ".format(items[e2[0]]),end="")
            for i3,e3 in enumerate(e2) :
                if i3==0:
                    continue
                print("and {} ".format(items[e3]),end="")
            print(") ",end="")
        print("")
def modPrint(var,items):
    print("[ ",end="")
    if type(var.x)==list:
        e2=var.x
        print("( {}".format(items[e2[0]]),end="")
        for i3,e3 in enumerate(e2) :
            if i3==0:
                continue
            print("",col("AND","green"), "{}".format(items[e3]),end="")
        print(" ) ",end="")
    else:
        modPrint(var.x,items)
    sign=var.z.upper()
    print("",col(sign,color[sign]),"",end="")
    if type(var.y)==list:
        e2=var.y
        print("( {} ".format(items[e2[0]]),end="")
        for i3,e3 in enumerate(e2) :
            if i3==0:
                continue
            print("",col("AND","green"), "{}".format(items[e3]),end="")
        print(" ) ",end="")
    else:
        modPrint(var.y,items)
    print("] ",end="")
def mprint(drule,items):
    for idx,i in enumerate(drule):
        print("{}. ".format(idx+1), end="")
        modPrint(i,items)
        print("")
def head(k,minconf,maxSupp,item_trans,context,items,trans,sizelimit=6,xorth=0.95):
    contextRule=[]
    var=rules(k,minconf,context,item_trans,trans,maxSupp,sizelimit,xorth)
    if var:
        if len(var[1])==0 and len(var[2])==0:
            print("NO rule found for context: {}".format(str(context)))
        else:
            contextRule.append(var)
    for i in contextRule:
        printContext(i[0],items)
        print("drule1")
        printRule(i[1],items)
        print("drule2")
        mprint(i[2],items)
        print("\n")
def func(m=70,path="/home/abhay/Desktop/project101/NSFabsDataset/*/*/*/*.txt"):  #<---------------------ADDRESS

    paths=glob.glob(path)
    print(paths)
#     paths=paths[0:50000]                     #<--------------------------change
#     if path.find('.csv')!=-1:
#         paths=path
    print("no. of files is {}".format(len(paths)))
    stems,stemDict=readRaw(paths)
    stemDict=list(set(stemDict))
    stemDict.sort()
    newDict=tfidf(m,stemDict,stems)

#     with open("./procDatat/tfidf.txt", "rb") as fp:
#         alld=pickle.load(fp)
#     take=int(len(alld)*m/100)
#     newDict=alld[:take]
#     with open("./procDatat/data.txt", "rb") as fp:
#         stems=pickle.load(fp)

    ############
#     a , _ = stem_and_lemmatize([tweet])
    if word1 not in newDict:
        newDict.append(word1)           #<---------------------------changes here
    if word2 not in newDict:
        newDict.append(word2)
    newDict.sort()
    ############
    revDict={}
    for idx,i in enumerate(newDict):
        revDict[i]=idx
    print("dictionary length is {}".format(len(newDict)))

    transactions=transGen(stems,newDict,revDict)
    item_trans=revTrans(transactions)
#     print(getsizeof(transactions),"\n")
    with open("./procDatat/trans.txt", "wb") as fp:  #<---------------------ADDRESS
        pickle.dump(transactions, fp)

    with open("./procDatat/item_trans.txt", "wb") as fp:  #<---------------------ADDRESS
        pickle.dump(item_trans, fp)

    with open("./procDatat/itemDict.txt", "wb") as fp:    #<---------------------ADDRESS
        pickle.dump(revDict, fp)

    with open("./procDatat/items.txt", "wb") as fp:         #<---------------------ADDRESS
        pickle.dump(newDict, fp)



# In[4]:
start=time.time()
func(m,path)
end=time.time()
print("time taken is {} minutes".format((end -start)/60))


# In[12]:


#load_data
with open("./procHappi/trans.txt", "rb") as fp:
    trans=pickle.load(fp)
with open("./procHappi/items.txt", "rb") as fp:
    items=pickle.load(fp)
with open("./procHappi/item_trans.txt", "rb") as fp:
    item_trans=pickle.load(fp)
with open("./procHappi/itemDict.txt", "rb") as fp:
    item_dict=pickle.load(fp)


# In[13]:


# if not context:
#     sizeLimit=1
#     freqsets=freq_itemset(minno,range(len(items)),item_trans,sizeLimit) #<-----------sizelimit---------------potential contexts
#     fsets=list(freqsets.values())[0]
#     for context in fsets:
#         head(k,minconf,maxSupp,item_trans,context,items,trans)
# else:



dataname="tweets_data"
tweet="happiness"
readData=500000
path="/home/abhay/Desktop/project101/output_got.csv"
word1="happi"
m=70 ##tfidf threshold
th=1 ##threshold for xor
supp=8
k=3
minconf=10
maxSupp=100
context="happi"
print("dataname={}".format(dataname))
print("tweet={}".format(tweet))
print("readData={}".format(readData))
print("m={}".format(m))
print("path={}".format(path))
print("context={}".format(context))
print("supp={}".format(supp))
print("k={}".format(k))
print("minconf={}".format(minconf))
print("maxSupp={}".format(maxSupp))
conf=[3,5,8,10,20,30,40]
conf.reverse()
context=[item_dict[word1]]
print("\n#################################################################################\n")
# a , _ = stem_and_lemmatize([tweet])
# conf=[20]
# context=[item_dict[word1]]
for minconf in conf:
    print("minconf value is: {}".format(minconf))
    head(k,minconf,maxSupp,item_trans,context,items,trans,sizelimit=6,xorth=th)
    print("\n#################################################################################\n")
