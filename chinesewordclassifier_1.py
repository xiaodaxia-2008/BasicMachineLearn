# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:58:55 2018

@author: xiaozhe

Description: split the chinese text to words
"""
import logging
import os
import jieba
from functionsforclassifier import readfile, savefile

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

corpus_path = './文本分类语料库/'
seg_path = './分词后的的语料库/'

catelist = os.listdir(corpus_path)

for mydir in catelist:
    logging.debug(mydir)
    class_path = corpus_path + mydir + '/'
    seg_dir = seg_path + mydir + '/'
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)
    for file_path in file_list:
        full_name = class_path + file_path
        content = readfile(full_name)
        if content:
            content.strip()
            content = content.replace("\r\n", "").strip()
            content_seg = jieba.cut(content)
            savefile(seg_dir+file_path, ' '.join(content_seg))
    
logging.debug('中文分词结束')
        