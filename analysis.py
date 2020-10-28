
import argparse
import nltk
import numpy as np
import os
import pandas as pd
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model, preprocessing
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


df = pd.concat([pd.Series([x,y]) for (x,y) in texts.items()]).transpose()
df.columns = ['Link','FullContent']

data = alldata.merge(df,how='left',on='Link')
data.FullContent[data.FullContent.insull()] = data.Content[data.FullContent.isnull()]

data.to_csv('{}/{}'.format(DIR,'fullContent.csv'))

data = pd.read_csv('{}/{}'.format(DIR,'fullContent.csv'),index_col=0,encoding='latin1')
oped = open('{}/oped.txt'.format(DIR),'r').read().lower()


data = pd.concat([data,pd.Series({'Date':None, 
                                  'Content':'The Op-Ed', 
                                  'Official':None, 
                                  'Link':None, 
                                  'FullContent':oped}).to_frame().transpose()],
                 sort=False)
data.reset_index(drop=True,inplace=True)

#
# Feature extraction and modeling
#

data['words'] = data.FullContent.apply(lambda x: word_tokenizer.tokenize(x.lower()))
data['tokens'] = data.FullContent.apply(lambda x: nltk.word_tokenize(x.lower()))
data['sentences'] = data.FullContent.apply(lambda x: sentence_tokenizer.tokenize(x.lower()))

data['Wordiness'] = data.apply(lambda x: np.mean([len(word_tokenizer.tokenize(s)) for s in x.sentences]), axis=1)
data['LexicalDiversity'] = data.apply(lambda x: len(set(x.words))/len(x.words), axis=1)
data['commaUse'] = data.apply(lambda x: x.tokens.count(',')/len(x.sentences), axis=1)
data['colonUse'] = data.apply(lambda x: x.tokens.count(':')/len(x.sentences), axis=1)
data['semiColonUse'] = data.apply(lambda x: x.tokens.count(';')/len(x.sentences), axis=1)


# get most common words
NUM_TOP_WORDS = 20

all_text = ' '.join(list(data.FullContent))
all_text = all_text.lower()
all_tokens = nltk.word_tokenize(all_text)
fdist = nltk.FreqDist(all_tokens)
vocab = [word for word, _ in fdist.most_common(NUM_TOP_WORDS)]

# use sklearn to create the bag of words feature vector for each chapter
vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
fvs_bow = vectorizer.fit_transform(data.FullContent.apply(lambda x: x.lower())).toarray().astype(np.float64)
 
# normalise by dividing each row by its Euclidean norm
fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]
vocab[1] = 'period'
vocab[2] = 'comma'
vocab[15] = 'aposS'
fvs_bow = pd.DataFrame(fvs_bow,columns=vocab)


def syntaxType(tokens):
    text = [p[1] for p in nltk.pos_tag(tokens)]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = [text.count(pos)/len(text) for pos in pos_list]

    return fvs_syntax

fvs_syntax = data.tokens.apply(lambda x: syntaxType(x))
data['NN'] = fvs_syntax.apply(lambda x: x[0])
data['NNP'] = fvs_syntax.apply(lambda x: x[1])
data['DT'] = fvs_syntax.apply(lambda x: x[2])
data['IN'] = fvs_syntax.apply(lambda x: x[3])
data['JJ'] = fvs_syntax.apply(lambda x: x[4])
data['NNS'] = fvs_syntax.apply(lambda x: x[5])

data.to_csv('{}/statements.csv'.format(DIR))

xtrain = data[['Wordiness','LexicalDiversity','NN','NNP','DT','IN','JJ','NNS','commaUse','colonUse','semiColonUse']]
xtrain = pd.concat([xtrain,fvs_bow],axis=1)

tfidf = TfidfVectorizer(analyzer='word',
                        token_pattern=r'\w{1,}',
                        ngram_range=(1,3))

tfidf.fit(data.FullContent.apply(lambda x: x.lower()))

features = tfidf.transform(data.FullContent[xtrain.notnull().all(axis=1)&~data.Content.apply(lambda x: x.startswith('Tweet'))].apply(lambda x: x.lower()))
ofeat = tfidf.transform([oped.lower()])

xtrain['Y'] = data.Official
xtrain = xtrain[~data.Content.apply(lambda x: x.startswith('Tweet'))].dropna()
Y = xtrain.Y
xtrain = xtrain.drop('Y', axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(xtrain) 

xfeat = sparse.hstack((features,xtrain.values)).tocsr()

opedrow = pd.concat([data.iloc[13320][['Wordiness','LexicalDiversity','NN','NNP','DT','IN','JJ','NNS','commaUse','colonUse','semiColonUse']],
                     fvs_bow.iloc[13320]],sort=False)
opedrow = opedrow.astype(float)
opedrow = sparse.hstack((ofeat,opedrow.values)).tocsr()


def trainModel(features, labels):
    # Takes in features (X) and labels (Y) and trains a model to 
    # predict labels on new data. 
    # 
    # The model is trained using 3-fold cross validation and a grid 
    # search of relevant parameters:
    # 
    # loss func: logarithmic, hinge (SVM), or squared hinge (quad SVM)
    # penalty: no regularization, l1, l2, or elasticnet
    # alpha: 0.01, 0.001, or 0.0001
    # 
    # the best performing model, based on classification accuracy, is
    # returned.
    # 

    model = linear_model.SGDClassifier(n_jobs=-1,random_state=0)

    parameters = {'loss':('hinge','squared_hinge','log'), 
                  'penalty':('none','l2','l1','elasticnet'),
                  'alpha':(0.0001,0.001,0.01)}

    clf = GridSearchCV(model, parameters, scoring='accuracy')
    clf.fit(features,labels)

    return clf.best_estimator_


mod = trainModel(xfeat,Y)
ypred = mod.decision_function(xfeat)
predLabels = [mod.classes_[x] for x in np.argmax(ypred,axis=1)]



# Train model to predict any particular statement is from particular official
# Predict for oped





