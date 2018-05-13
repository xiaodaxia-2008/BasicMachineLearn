# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:13:54 2018

@author: xiaozhe
"""
import os
from sklearn.datasets.base import Bunch
import logging
from functionsforclassifier import readfile, write_pkl
import numpy as np



logging.basicConfig(level = logging.DEBUG, format = '%(message)s')


seg_path = './分词后的的语料库/'

# create all data bunch
bunch_total_seg_set = Bunch(target_name=[], label=[],
                            filenames=[], contents=[])
catelist = os.listdir(seg_path)
bunch_total_seg_set.target_name.extend(catelist)
for mydir in catelist:
    logging.debug(mydir)
    class_path = seg_path + mydir + '/'
    file_list = os.listdir(class_path)
    for file_path in file_list:
        full_name = class_path + file_path
        bunch_total_seg_set.label.append(mydir)
        bunch_total_seg_set.filenames.append(full_name)
        bunch_total_seg_set.contents.append(readfile(full_name).strip())
   
# save the Bunch data
wordbag_path = './word_bunch/train_jieba_wordbunch_set.pkl'
write_pkl(wordbag_path, bunch_total_seg_set)


# create test set data
bunch_test = Bunch(target_name=[], label=[], filenames=[], contents=[])
bunch_test.target_name = bunch_total_seg_set.target_name
np.random.seed = 1777
index_test = np.random.randint(0, len(bunch_total_seg_set.label),
                               size=10*len(bunch_total_seg_set.target_name))
for i in index_test:
    bunch_test.label.append(bunch_total_seg_set.label[i])
    bunch_test.filenames.append(bunch_total_seg_set.filenames[i])
    bunch_test.contents.append(bunch_total_seg_set.contents[i])

bunch_test.label.append('交通214')  # mannuual add a test data
bunch_test.filenames.append(
        './分词后的的语料库/交通214/xz.TXT')
bunch_test.contents.append('交通 拥堵 环境 污染')
bunch_test_path = './word_bunch/test_jieba_wordbunch_set.pkl'
write_pkl(bunch_test_path, bunch_test)

logging.debug('finished')
