__author__ = 'nittella'
# coding: utf-8

from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

pred = clf.get_prediction_message('плохой экран')
print(pred)