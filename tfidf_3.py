# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:41:39 2018

@author: xiaozhe
"""


from sklearn.datasets.base import Bunch
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from functionsforclassifier import readfile, write_pkl, read_pkl


wordbag_path = './word_bunch/train_jieba_wordbunch_set.pkl'
stopwdpath = './停用词集/百度停用词列表.txt'
stopwdlist = readfile(stopwdpath).splitlines()

bunch = read_pkl(wordbag_path)
train_tfidfspace = Bunch(target_name = bunch.target_name, label = bunch.label,
                   filenames = bunch.filenames, tdm=[], vocabulary = {})
vectorizer = TfidfVectorizer(stop_words=stopwdlist,
                             sublinear_tf=True, max_df=0.5)
transformer = TfidfTransformer()
train_tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
train_tfidfspace.vocabulary = vectorizer.vocabulary_
space_path = './word_bunch/train_tfidf_space.pkl'
write_pkl(space_path, train_tfidfspace)


bunch_test_path = './word_bunch/test_jieba_wordbunch_set.pkl'
bunch_test = read_pkl(bunch_test_path)
testspace = Bunch(target_name = bunch_test.target_name,
                  filenames = bunch_test.filenames,
                  label = bunch_test.label,
                  contents = bunch_test.contents, tdm=[], vocabulary={})
vectorizer_test = TfidfVectorizer(stop_words=stopwdlist,
                                  sublinear_tf=True, max_df=0.5,
                                  vocabulary=train_tfidfspace.vocabulary)
testspace.tdm = vectorizer_test.fit_transform(bunch_test.contents)
testspace.vocabulary = train_tfidfspace.vocabulary
testspace_path = './word_bunch/test_tfidf_space.pkl'
write_pkl(testspace_path, testspace)

logging.debug('Finished creating the IF-TDF space for train and test set')
