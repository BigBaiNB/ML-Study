#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:04:14 2018

@author: apple
"""
import tensorflow as tf
import os

os.environ['TF_CPP_LOG_LEVEL'] = '2'
vector = tf.constant([1,2,3])
matraix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])

result = vector * matraix
#张量定义后无法执行，必须填充到图中执行

with tf.Session() as sess:
    print(sess.run(result))