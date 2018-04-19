import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import gc,fasttext
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer############
from keras.models import Sequential
from keras.layers import Dense,Embedding,GlobalAveragePooling1D
from keras.models import model_from_json
import os
from tensorflow import set_random_seed
import collections


BASE_DIR='/home/mohd/shiksha_repo/shikshaoss/online_models/auto_ans'
# TYPE='LISTING';
# LISTING='EXAM';
# LISTING='COMPARISON';

def readFile(TYPE):
	inputFile='';
	if TYPE=='LISTING':
	    inputFile = BASE_DIR+'/listing_q.csv'
	elif TYPE=='EXAM':
	    inputFile = BASE_DIR+'/exam_q.csv'
	elif TYPE=='COMPARISON':
	    inputFile = BASE_DIR+'/comparison_q.csv'
	raw = pd.read_csv(inputFile)
	raw2=raw.dropna(how='any')#,subset=[questions,class,direct_indirect,factual_opinion,is_original]) 
	print(set(raw2['class'].tolist()))
	print(set(raw2['direct_indirect'].tolist()))
	print(set(raw2['factual_opinion'].tolist()))
	print(set(raw2['is_original'].tolist()))
	print(len(set(raw2['question_id'].tolist())))
	raw2['direct_indirect'] = raw2['direct_indirect'].str.strip()
	raw2['factual_opinion'] = raw2['factual_opinion'].str.strip()
	print np.shape(raw2)
	raw2=raw2.drop_duplicates(subset=['questions','class','factual_opinion','direct_indirect'],keep=False)##################,'factual_opinion','direct_indirect'
	print np.shape(raw2)
	raw2["label"] = TYPE + '-' + raw2["class"] + '-' + raw2["direct_indirect"] + '-' + raw2["factual_opinion"]
	# raw2["label"] = TYPE + '-' + raw2["class"] + '-' + raw2["factual_opinion"]
	orig=raw2[raw2.is_original==1]
	print(pd.crosstab(index=orig['class'],columns=['counts']))
	# col_0          counts
	# class                
	# GDPI               17
	# admit card          5
	# answer key          6
	# coaching           30
	# counseling         41
	# cut off            18
	# dates              28
	# eligibility        39
	# notification        3
	# pattern            25
	# preparation        28
	# sample papers      10
	# scores             36
	# syllabus            6
	print(pd.crosstab(index=orig['class'],columns=orig['factual_opinion']))
	# factual_opinion  factual  opinion
	# class                            
	# GDPI                   0       17
	# admit card             5        0
	# answer key             2        4
	# coaching               0       30
	# counseling             8       33
	# cut off                0       18
	# dates                 20        8
	# eligibility            8       31
	# notification           1        2
	# pattern                8       17
	# preparation            0       28
	# sample papers          5        5
	# scores                 2       34
	# syllabus               2        4
	print(pd.crosstab(index=orig['class'],columns=orig['direct_indirect']))
	# direct_indirect  direct  indirect
	# class                            
	# GDPI                 17         0
	# admit card            5         0
	# answer key            6         0
	# coaching             27         3
	# counseling           40         1
	# cut off              18         0
	# dates                22         6
	# eligibility          32         7
	# notification          3         0
	# pattern              24         1
	# preparation          25         3
	# sample papers         9         1
	# scores               34         2
	# syllabus              6         0
	print np.shape(raw2)
	return raw2


listing_q = readFile('LISTING')
exam_q = readFile('EXAM')
cmp_q = readFile('COMPARISON')

question_data = pd.concat((listing_q, exam_q,cmp_q), axis=0)
# question_data = np.concatenate((listing_q, exam_q,cmp_q), axis=0)
print question_data
print np.shape(question_data)
print len(set(question_data['questions'].tolist()))
print(len(set(question_data['label'].tolist())))

##############Preprocessing
question_data['questions'] = question_data['questions'].str.lower()
question_data['questions'] = question_data['questions'].str.replace('[\n\r"]',' ')

questions_with_multiple_labels= question_data.groupby("questions").filter(lambda g: (g.name != 0) and (g.questions.size >= 2))
questions_with_multiple_labels=questions_with_multiple_labels.sort_values('questions')
questions_with_multiple_labels.to_csv(BASE_DIR+'/questions_with_multiple_labels.csv') #, encoding='utf-8'



############
# train test split
question_unique = dict()
for index, row in tqdm(question_data.iterrows()):
	if row['questions'] not in question_unique:
		question_unique[row['questions']]=list()
	question_unique[row['questions']].append(row['label'])

question_unique_pd = pd.DataFrame.from_dict({'questions':question_unique.keys(),'label':question_unique.values()})


train_x, test_x, train_y_labels, test_y_labels = train_test_split(question_unique_pd.loc[:,['questions']], question_unique_pd.loc[:,['label']], train_size=0.90,random_state = 123)

##
encoder = MultiLabelBinarizer()

train_y = encoder.fit_transform(train_y_labels['label'].tolist())
test_y = encoder.transform(test_y_labels['label'].tolist())


###############################
tfv = TfidfVectorizer(min_df=1,  max_features=None, 
    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
    stop_words = 'english')

tfv.fit(train_x['questions'].tolist())
xtrain_tfv =  tfv.transform(train_x['questions'].tolist())
xtest_tfv =  tfv.transform(test_x['questions'].tolist())



svd = decomposition.TruncatedSVD(n_components=120)########  180
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xtest_svd = svd.transform(xtest_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xtest_svd_scl = scl.transform(xtest_svd)


#####################################fasttext
def write_list_to_file(file='',object=None):
	fileHandle= open (file,'w')
	for i,item in enumerate(object):
		fileHandle.write('%s\n'%item)
	fileHandle.close()

##Supervised, or starSpace*****

train_x, test_x, train_y_labels, test_y_labels = train_test_split(question_data.loc[:,['questions']], question_data.loc[:,['label']], train_size=0.90,random_state = 123)
fasttext_supervised_train_x = '__label__'+train_y_labels['label'].str.replace('[- ]+','_')+' '+train_x['questions']
fasttext_supervised_test_x = '__label__'+test_y_labels['label'].str.replace('[- ]+','_')+' '+test_x['questions']


write_list_to_file(BASE_DIR+'/fasttext_supervised_train_x.txt',fasttext_supervised_train_x.tolist())
write_list_to_file(BASE_DIR+'/fasttext_supervised_test_x.txt',fasttext_supervised_test_x.tolist())


s_model=fasttext.supervised(BASE_DIR+'/fasttext_supervised_train_x.txt',BASE_DIR+'fasttext_supervised_model',silent=0)

result=s_model.test(BASE_DIR+'/fasttext_supervised_test_x.txt')
print 'P@1:', result.precision
print 'R@1:', result.recall

preds = s_model.predict(test_x['questions'].tolist(),k=2)
preds_prob = s_model.predict_proba(test_x['questions'].tolist(),k=2)

preds_prob_d = list()
printer=1
if printer==1 :
	import re
	for row in preds_prob:
		row_str=str(row)
		preds_prob_d.append(re.sub("indirect|direct|College_comparison|course_comp","",row_str))
else:
	preds_prob_d=preds_prob
csvs=pd.DataFrame.from_dict({"questions":test_x['questions'].tolist(),'preds':preds_prob_d})
csvs.to_csv(BASE_DIR+'/fasttext_reponse.csv') #, encoding='utf-8'

###################
#Word embeddings skipgram

write_list_to_file(BASE_DIR+'/fasttext_unsupervised_train_x.txt',train_x['questions'].tolist())
write_list_to_file(BASE_DIR+'/fasttext_unsupervised_test_x.txt',test_x['questions'].tolist())

embedding_model = fasttext.skipgram(BASE_DIR+'/fasttext_unsupervised_train_x.txt',BASE_DIR+'fasttext_skipgram_model',min_count=1,thread=3,silent=0,epoch=200)

def sent2vec(doc,dim=100,algo='avg',params=None):
    words = str(doc)
    words = (words.split())
    if algo=='avg':
        words.append('</s>')
        docEmbedding = np.zeros((1,dim))
        x=0
        for w in words:
            w_emb = self.wordEmbeddings.get(w,None)
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

## for optimise
def read_embedding_file(file=''):
	fileHandle= open (file)
	en_model = {}
	for line in fileHandle:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    en_model[word] = coefs
	fileHandle.close()
	return en_model

en_model = read_embedding_file(BASE_DIR+'fasttext_skipgram_model.vec')


X=list()
for row in tqdm(train_x['questions'].tolist()):
    X.append(sent2vec(row,100,'unsupervised',params={'model':embedding_model,'embedding':en_model}))

train_x_emb = np.asarray(X)

X=list()
for row in tqdm(test_x['questions'].tolist()):
    X.append(sent2vec(row,100,'unsupervised',params={'model':embedding_model,'embedding':en_model}))

test_x_emb = np.asarray(X)


###### stack features:
train_x_nnet = np.concatenate([xtrain_svd_scl, train_x_emb],axis=1)
test_x_nnet = np.concatenate([xtest_svd_scl, test_x_emb],axis=1)




####neural net
model = Sequential()
model.add(Dense(96, input_dim=np.shape(train_x_nnet)[1], kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(32, kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(np.shape(test_y)[1], kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
np.random.seed(7)
set_random_seed(123)
# earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=0, mode='auto')
history = model.fit(train_x_nnet, train_y, epochs=2, batch_size=10, verbose=1,validation_data=(test_x_nnet,test_y))#,class_weight = {0:4,1:1}, callbacks=[earlystop])
# evaluate the model
model.evaluate(test_x_nnet, test_y, verbose=1)
preds = model.predict(test_x_nnet)

(preds>0.5).sum(axis=0)###############to observe zeros, class imbalance
## question multi labelled
unique, counts = np.unique((preds>0.5).sum(axis=1), return_counts=True)
dict(zip(unique, counts))

preds[preds>0.5]=1
preds[preds<=0.5]=0
pred_labels = encoder.inverse_transform(preds) #############to check
# collections.Counter(pred_labels.sum(axis=1))

printLables = list()
if printer:
	for row in pred_labels:
		printLables.append(tuple([row_label.replace('indirect','').replace('direct','').replace('College comparison','').replace('course comp','') for row_label in row]))
else:
	printLables = pred_labels


response = pd.DataFrame({'question':test_x['questions'].tolist(),'result':printLables})


response.to_csv(BASE_DIR+'/tfidf_embed_reponse.csv') #, encoding='utf-8'

# Handke multiple labels
# solr query enhancement
# Data analyses- skewness, changing listing names to entities ,data broadness, phrases instead of question, etc
# word embeddings, intent features, etc

# phrases or more questions abn,
# category pages and comparisons
# $$ for listing, 
# identification of common listings later
# no lnks
# 0-5  -> 5 counts
# cutoff factual???
# notification factual?????????

# $$$$$$$$$$$$$$$$$ below
# sylaabus
# pattern
# dates
# admitcard
# anser key
# question paper
# elleigi

# factual-0:
# cutoff-- no info hence all opinion
# ###-----> no info 
# coachin
# scores
# gdpi
# preparation
# notif
# ###
# notif and date related



# data go through and then communicate abnv
# solr- rplace numbers by placeholder, lisitngs,exams also,   dynamic limit
# code review,   \:  how

# looked for any anomaly, skewness, or shiksha.com info

# Observations:
# exam dates have 'counselling' word
# placements and cutoff to tags to revisit


# 'location',
# 'base_course',
# 'sub_spec',
# 'specialization',
# 'credential',
# 'course_level',
# 'et_dm',
# 'approvals',
# 'course_status',
# 'accreditation',
# 'college_ownership',
# 'fees',
# 'exam',
# 'facilities'
