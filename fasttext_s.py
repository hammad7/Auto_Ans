fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/v2_text_dataEmbedrough2.txt')
vectors = []

with fileHandle as f:
    for line in f:
        fields = line.split()
        vector = np.fromiter((float(x) for x in fields),
                             dtype=np.float)
        vectors.append(vector)

fileHandle.close()

vectors = pd.DataFrame(vectors)
labels=[]

fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_label_data.txt')

with fileHandle as f:
    for line in f:
      
        fields = line.split()
        # word = fields[0].decode('utf-8')
        temp = -1
        if fields[0]=='__label__1':
          temp=1
        elif fields[0]=='__label__2':
          temp=0
        if temp==-1:
          print('error')
        # words.append(word)
        labels.append(temp)


fileHandle.close()
X = np.array(vectors)

Y = np.array(labels)
Y = Y.reshape(np.shape(X)[0],1)

train_x = X
train_y = Y

fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/v2_test_text_dataEmbedrough2.txt')
vectors = []

with fileHandle as f:
    for line in f:
        fields = line.split()
        vector = np.fromiter((float(x) for x in fields),
                             dtype=np.float)
        vectors.append(vector)

fileHandle.close()

vectors = pd.DataFrame(vectors)
labels=[]

fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_test_label_data.txt')
with fileHandle as f:
  for line in f:
    fields = line.split()
    temp = -1
    if fields[0]=='__label__1':
      temp=1
    elif fields[0]=='__label__2':
      temp=0
    if temp==-1:
      print('error')
    labels.append(temp)

fileHandle.close()

X = np.array(vectors)

Y = np.array(labels)
Y = Y.reshape(np.shape(X)[0],1)

test_x = X
test_y = Y



###textblob
import numpy as np
import pandas as pd
import csv
import json
import re
import datetime
from tqdm import tqdm
from textblob import TextBlob
import collections
from sklearn.preprocessing import normalize
from tensorflow import set_random_seed
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import gc
from scipy.sparse import csr_matrix
from keras.layers import Embedding,Input,GlobalAveragePooling1D,Dense,MaxPooling1D,Conv1D,Flatten,BatchNormalization,Dropout,Concatenate,Merge
from keras.models import Model,Sequential,model_from_json
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib




model = Sequential()
model.add(Dense(64, input_dim=np.shape(train_x)[1], kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.2))  ##imp because 0/1 train_text_data_x_est
model.add(BatchNormalization())
# model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
np.random.seed(7)
set_random_seed(123)
# earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=0, mode='auto')
history = model.fit(np.asarray(train_x), train_y, epochs=35, batch_size=10, verbose=1,validation_data=(test_x, test_y))#,class_weight = {0:4,1:1})
model.evaluate(test_x, test_y, verbose=1)
# preds=model.predict(train_text_data_x_est)
preds=model.predict(test_x)
confusion_matrix(test_y,preds >= 0.5)


/home/mohd/fastText/fasttext skipgram -input /home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_text_data.txt -output /home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2 -minCount 1 -thread 3 

/home/mohd/fastText/fasttext print-sentence-vectors /home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2.bin < /home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_text_data.txt > /home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/v2_text_dataEmbedrough2.txt
/home/mohd/fastText/fasttext print-sentence-vectors /home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2.bin < /home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_test_text_data.txt > /home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/v2_test_text_dataEmbedrough2.txt


# modelE = fasttext.skipgram('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_test_text_data.txt', '/home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2p',min_count=1,thread=3,silent=0,epoch=1)

modelE = fasttext.skipgram('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_text_data.txt', '/home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2p',min_count=1,thread=3,silent=0,epoch=30)

def sent2vec(doc,dim=100):## tocheck cbow/skipgram, to check oov words(fullstpop)
    words = str(doc)
    words = (words.split())
    M = []
    x=0
    for w in words:
        x=x+1
        try:
            M.append(en_model[w])
        except:
            print('OOV:'+w)
            M.append(modelE[w])
    M = np.array(M)
    N = np.linalg.norm(M, axis=1)
    M=M/N[:,np.newaxis]
    # for i in range(len(N)):
    #     for j in range(len(M[i])):
    #             M[i][j] = M[i][j] / N[i]
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(dim)
    return v / x










def sent2vec(doc,dim=100,algo='avg',params=None):
    words = str(doc)
    words = (words.split())
    if algo=='avg':
        words.append('</s>')
        docEmbedding = np.zeros((1,dim))
        x=0
        for w in words:
            w_emb = wordEmbeddings.get(w,None)
            if w_emb is not None:
                # w_emb=w_emb.reshape(1,dim) ##nn
                docEmbedding=np.add(docEmbedding,w_emb)
                x=x+1
        if x>0:
            docEmbedding = docEmbedding/x
        return docEmbedding
    elif algo=='unsupervised':
        docEmbedding = []
        for w in words:
            try:
                docEmbedding.append(params['embedding'][w])
            except:
                print('OOV:'+str(w))
                docEmbedding.append(params['model'][w])
        docEmbedding = np.array(docEmbedding)
        N = np.linalg.norm(docEmbedding, axis=1)
        docEmbedding=docEmbedding/N[:,np.newaxis]
        docEmbedding = docEmbedding.sum(axis=0)
        if len(words)>0:
            docEmbedding = docEmbedding/len(words)
        return docEmbedding



fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2p.vec')
en_model = {}
for line in fileHandle:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    en_model[word] = coefs

fileHandle.close()



model2=fasttext.load_model("/home/mohd/shiksha_repo/shikshaoss/review_classification/fastTextModel/review_model_v2rough2.bin")





fileHandle= open('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_text_data.txt')
train_text = []

with fileHandle as f:
    for line in f:
        train_text.append(line)

fileHandle.close()



fileHandle= open('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_test_text_data.txt')
test_text = []

with fileHandle as f:
    for line in f:
        test_text.append(line)

fileHandle.close()

X=list()
for row in tqdm(train_text):
    X.append(sent2vec(row,100,'unsupervised',{'model':modelE,'embedding':en_model}))

X = np.asarray(X)
train_x=X

X=list()
for row in tqdm(test_text):
    X.append(sent2vec(row,100,'unsupervised',{'model':modelE,'embedding':en_model}))

X = np.asarray(X)
test_x=X

labels=list()
fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_label_data.txt')
with fileHandle as f:
    for line in f:
      
        fields = line.split()
        # word = fields[0].decode('utf-8')
        temp = -1
        if fields[0]=='__label__1':
          temp=1
        elif fields[0]=='__label__2':
          temp=0
        if temp==-1:
          print('error')
        # words.append(word)
        labels.append(temp)


fileHandle.close()

Y = np.array(labels)
Y = Y.reshape(np.shape(train_x)[0],1)
train_y=Y

labels=list()
fileHandle= open ('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/v2_test_label_data.txt')
with fileHandle as f:
    for line in f:
      
        fields = line.split()
        # word = fields[0].decode('utf-8')
        temp = -1
        if fields[0]=='__label__1':
          temp=1
        elif fields[0]=='__label__2':
          temp=0
        if temp==-1:
          print('error')
        # words.append(word)
        labels.append(temp)


fileHandle.close()

Y = np.array(labels)
Y = Y.reshape(np.shape(test_x)[0],1)
test_y=Y


fileHandle = open('/home/mohd/shiksha_repo/shikshaoss/review_classification/testCase/rough.txt', 'w')
for item in vectors:
  fileHandle.write( item)

fileHandle.close()
