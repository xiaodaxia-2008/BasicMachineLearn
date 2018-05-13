# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:46:35 2018

@author: xiaozhe
"""
import pickle
import os


def readfile(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            return content
    except UnicodeDecodeError:
        return


def savefile(filepath, content):
    filedir = os.path.split(filepath)[0]
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    with open(filepath, 'w') as file:
        file.write(content)


def read_pkl(path):
    with open(path, 'rb') as file:
        content = pickle.load(file)
    return content


def write_pkl(path, bunchobj):
    filedir = os.path.split(path)[0]
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    with open(path, 'wb') as file:
        pickle.dump(bunchobj, file)
