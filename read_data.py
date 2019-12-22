

import sys
from tensorflow.core.example import example_pb2
import numpy as np
import nltk
import struct
import glob
import random
import re


LOC = 1




class Text:
  def __init__(self):
    self.article = None
    self.abstract = None
    self.sentences = None
    self.article_tokens = None
    self.abstract_tokens = None
    
    # Previous attributes
    self.sent_obj_list = None
    self.sent_tokens_list = None
    self.high_tfidf_words = []
    self.X = None
    self.Y = None

  def add_article(self,article):
    self.article = article
    self.sent_tokens_list = nltk.sent_tokenize(article)
    
  def add_abstract(self,abstract):
    self.abstract = abstract


regex1 = r"\"(.*?)\""
regex2 = r"<s>(.*?)</s>"
patterns1  = r"</s>|(\. </s>)"
patterns2 = r"\\\'\\\' |``"
patterns3 = r" \\\'"
patterns4 = r"-rrb-|-lrb-|-lsb-|-rsb-"


def processArticle(text):
  text = re.sub(patterns2, '\" ', text)
  text = re.sub(patterns3, '\'', text)
  text = re.sub(patterns4, '', text)
  return text


def processAbstract(text):
  text = re.sub('<s>', '', text)
  text = re.sub(patterns1, '. ', text)
  text = re.sub(patterns2, '\" ', text)
  text = re.sub(patterns3, '\'', text)  
  return text


# The following function modifies Abigail See's work
# https://github.com/abisee/pointer-generator/blob/master/data.py
def read_articles_and_summaries():
  if LOC == 1:
    data_path = "/Users/xiahu/Documents/Software/cnn-dailymail-master/finished_files/chunked/train_*"
  else:
    data_path = "/home/home3/xhu/Documents/Project/Summarization/data/train_*"
    
  single_pass = 1
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles

    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty

    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
      
    count, ind = 0, 0
    data_list = []

    for f in filelist:
      example_list = []
      reader = open(f, 'rb')
      article_list, abstract_list = [], []

      while True:
        len_bytes = reader.read(8)
        if not len_bytes:
          break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        tf_example = example_pb2.Example.FromString(example_str)

        for key in tf_example.features.feature:
          text = str(tf_example.features.feature[key])

          if key == 'article':
            matches = re.finditer(regex1, text)

            for _, match in enumerate(matches): 
              s, e = match.start()+1, match.end()-1

            text = processArticle(text[s:e])
            article_list.append(text)
          elif key == 'abstract':
            matches = re.finditer(regex2, text)
            s,e = np.inf,0

            for _, match in enumerate(matches): 
              s, e = min(s,match.start()+4), max(e,match.end()-5)

            text = processAbstract(text[s:e])
            abstract_list.append(text)
            count += 1 
            
          if count == 1000: 
            diff =  len(abstract_list) - len(article_list)
            if diff > 0: 
              abstract_list = abstract_list[:-diff]

            for i in range(len(article_list)):

              if len(article_list[i]) > 10:
                T = Text()
                T.add_article(article_list[i])
                T.sentences = nltk.sent_tokenize(article_list[i])
                T.add_abstract(abstract_list[i])
                example_list.append(T)

            data_list += example_list
            print('len(data_list)',len(data_list))
            example_list = []
            count = 0
            
        
          if len(data_list) > 3000: 
            return data_list

    return data_list
  



