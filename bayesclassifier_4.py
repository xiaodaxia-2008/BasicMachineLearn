# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:34:50 2018

@author: xiaozhe
"""

import logging
import datetime
from functionsforclassifier import read_pkl, write_pkl
from sklearn.naive_bayes import MultinomialNB

today = datetime.date.today().strftime('%Y-%m-%d')


#logging.basicConfig(filename='D:/LearningPython/classifierresult.txt',
#                    level=logging.DEBUG,
#                    format='%(asctime)s-%(message)s')

test_path = './word_bunch/test_tfidf_space.pkl'
train_path = './word_bunch/train_tfidf_space.pkl'

trainset = read_pkl(train_path)
testset = read_pkl(test_path)

clf = MultinomialNB(alpha=0.0001).fit(trainset.tdm, trainset.label)
write_pkl('./models/txt_classifier_%s.pkl'%(today), clf)
#clf = read_pkl('./models/clf.pkl')

predicted = clf.predict(testset.tdm)

total = len(predicted)
rate = 0
for flabel, filename, expect_cate in zip(testset.label, testset.filenames,
                                         predicted):
    if flabel != expect_cate:
        rate += 1
    logging.debug(filename+': 实际类别:'+flabel+'-->预测类别:'+expect_cate)
errorrate = float(rate)*100/float(total)
logging.debug('error rate:'+str(errorrate)+'%')



