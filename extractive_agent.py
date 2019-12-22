import sys, os, re, string, copy, itertools, queue, random

from nltk import word_tokenize, pos_tag, ne_chunk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from nltk.stem.porter import PorterStemmer
from tensorflow.contrib import layers
from pyrouge import Rouge155

import matplotlib, scipy, multiprocessing
import matplotlib.pyplot as pt

import tensorflow as tf
import numpy as np
import sklearn
import nltk
import gensim

from class_utils import *
from readData import *


if LOC == 1:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('/Users/xiahu/Documents/Software/word2vec/trunk/GoogleNews-vectors-negative300.bin', binary=True)
else:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('/home/home3/xhu/Documents/Project/word2vec/trunk/GoogleNews-vectors-negative300.bin', binary=True)


if LOC == 0: 
    matplotlib.use('Agg')


path = './tf-idf'
token_dict = {}
disjunction_words = ['but', 'however', '']

"""f
Some of the code here (gets the inspiration | is taken) from 
https://github.com/dennybritz/reinforcement-learning
http://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php
"""

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems



def tf_idf(data_list):
    for i, text in enumerate(data_list):
        art = text.article
        token_dict[i] = art.translate(string.punctuation)

    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(token_dict.values())
    return tfidf, tfs


# get the sum of the tfidf value of all words in a sentence and divide its
def get_sentence_tfidf(tfidf,tfs_vec,sent):
    s, vals, count = sent.sentence, [], 0
    response = tfidf.transform([s])
    feature_names = tfidf.get_feature_names()
    word_list = []
    for word_idx in response.nonzero()[1]:
        if feature_names[word_idx].isalnum():
            vals.append(tfs_vec[0, word_idx])
            word_list.append((feature_names[word_idx], tfs_vec[0, word_idx]))
            count += 1
            
    # we only pick the top 25% tfidf values of the sentence
    val = np.mean(sorted(vals, reverse=True)[:max(round(len(vals)/4), 1)])
    if np.isnan(val): 
        val = 0
    return val, word_list

        
    
def get_token_depth(token):
    text = token.split('(')
    return len(text[0]) / 2


inner_parent = re.compile("\((?:(?!\()(?!\)).)*\)")


def get_POS_and_words(text):
    res = inner_parent.findall(text)
    res = list(x.rstrip(')').lstrip('(').split() for x in res)
    return res
    
    
def get_parent_node(dep_dict,d):
    return dep_dict[d-1][-1]


def get_word_val(word,word_list,tfidf_list):
    w = PorterStemmer().stem(word)
    if w in word_list: return tfidf_list[word_list.index(w)]
    return 0


def find_and_add_POS_Word(cur_node,token,d,depth_dict,words,tfidf_vals):
    pos_word_pairs = get_POS_and_words(token)
    if len(pos_word_pairs) > 1: # if cur_node has multiple leave nodes
        node_list = list(Node(d+1).addPosWordVal(x[0], x[1],
                        get_word_val(x[1], words, tfidf_vals)) for x in pos_word_pairs)
        cur_node.add_children(node_list)
        cur_node.isPhrase = True   
        return len(node_list)

    else: # we only find one POS tag and word pair under cur_node
        if token.count('(') > 1:
            #print('we only find one POS tag and word pair under cur_node')
            node = Node(d+1)
            cur_node.add_child(node)
            node.addPosWordVal(pos_word_pairs[0][0], pos_word_pairs[0][1],
                               get_word_val(pos_word_pairs[0][1], words, tfidf_vals))

            if d+1 in depth_dict.keys(): 
                depth_dict[d+1].append(node)
            else: 
                depth_dict[d+1] = [node]

        else: # if this is a leaf node 
            cur_node.addWordVal(pos_word_pairs[0][1],
                                get_word_val(pos_word_pairs[0][1], words, tfidf_vals))
        return 1
    
    
def get_sentence_tree_representation(sent,tokens,tfidf,tfs_vec): # for each sentence, we construct its tree rep
    tree = Tree()
    depth_dict = {}
    top_tfidf_mean, tfidf_list = get_sentence_tfidf(tfidf, tfs_vec, sent)

    if len(tfidf_list):
        words, tfidf_vals = zip(*tfidf_list) # the words are stemmed
    else: 
        words, tfidf_vals = [], []
    
    sent.word_list, sent.tfidf_vals = words, tfidf_vals
    sent.top_tfidf_mean = top_tfidf_mean
    
    for token in tokens:
        d = get_token_depth(token)
        if d == 1: 
            tree.root = Root()
            tree.root.isSentence = True
            tree.node_count += 1
            cur_node = tree.root.node

        elif d == 2:
            cur_node = tree.root.add_child(2)    

            if '. .' in token: 
                cur_node.addPosWordVal('.','.',0) # period appears the 2nd level
                cur_node.isWord = True
            else: 
                pos = token.rstrip().lstrip()[1:].split()[0]
                cur_node.addPos(pos)

                tree.node_count += 1
                if ')' in token: 
                    tree.node_count += find_and_add_POS_Word(cur_node, token, d, depth_dict, words, tfidf_vals)
        else:
            cur_node = Node(d)
            get_parent_node(depth_dict,d).add_child(cur_node)
            pos = token.rstrip().lstrip()[1:].split()[0]
            cur_node.addPos(pos)              

            if ')' in token:
                tree.node_count += find_and_add_POS_Word(cur_node, token, d, depth_dict, words, tfidf_vals) 
            
        if cur_node.POS == 'PP': 
            sent.pp_list.append(cur_node)
        if cur_node.POS == 'S': 
            sent.subsentence_list.append(cur_node)
        
        if d in depth_dict.keys(): 
            depth_dict[d].append(cur_node)
        else: 
            depth_dict[d] = [cur_node]

    sent.tree = tree
    sent.depth = max(list(depth_dict.keys())) # get the depth of the parse tree
    

def get_sent_list(arg):
    if LOC == 1: 
        command = " /Users/xiahu/Documents/Software/stanford-parser-full-2017-06-09/lexparser.sh "
    else: 
        command = " /home/home3/xhu/Documents/Project/stanford-parser-full-2017-06-09/lexparser.sh "
    article, tfidf, tfs_vec, i = arg
    name1, name2 = "tmp" + str(i), "output" + str(i)
    command2 = name1 + " > " + name2
    command += command2 

    f = open(name1,'w')
    f.write(article)
    f.close()
    os.system(command)
    sent_list, count = [], 0
    sentences = nltk.sent_tokenize(article)

    f = open(name2,'r')

    for line in f:

        if '(ROOT\n' in line: # we have a new sentence

            if count >= len(sentences): 
                return None
            sent = Sent(sentences[count])
            line = f.readline()
            token_list = []

            while line != '\n' and line:
                token_list.append(line)
                line = f.readline()
            get_sentence_tree_representation(sent, token_list, tfidf, tfs_vec)
            sent_list.append(sent)
            count += 1

    f.close()
    if len(sentences) != count: 
        return None    
    
    return sent_list


# get the depth of the sentence's parse tree
def get_sentence_info2(data_list, tfidf, tfs_vecs, pool):
    num = len(data_list)
    args = [[data_list[i].article, tfidf, tfs_vecs[i], i] for i in range(num)]
    results = pool.map(get_sent_list, args)
    total_sent_list = []
    total_T_list = []

    for i,res in enumerate(results):
        if res != None: 
            total_sent_list.append(res)
            total_T_list.append(data_list[i])
            
    return total_sent_list, total_T_list


# get the depth of the sentence's parse tree
def get_sentence_info(article, tfidf, tfs_vec):
    if LOC == 1: 
        command = " /Users/xiahu/Documents/Software/stanford-parser-full-2017-06-09/lexparser.sh tmp > output"
    else: 
        command = " /home/home3/xhu/Documents/Project/stanford-parser-full-2017-06-09/lexparser.sh tmp > output"

    f = open('tmp', 'w')
    f.write(article)
    f.close()
    os.system(command)

    f = open('output', 'r')
    sent_list = []
    count = 0
    sentences = nltk.sent_tokenize(article)

    for line in f:
        if '(ROOT\n' in line: # we have a new sentence
            if count >= len(sentences): return None
            sent = Sent(sentences[count])
            line = f.readline()
            token_list = []
            while line != '\n' and line:
                token_list.append(line)
                line = f.readline()
            get_sentence_tree_representation(sent, token_list, tfidf, tfs_vec)
            sent_list.append(sent)
            count += 1
    f.close()
    if len(sentences) != count: 
        return None
    return sent_list


def get_sent_length(sent):
    return len(nltk.word_tokenize(sent.sentence))


def get_length_ratio(summ, k):
    # the length limit of the summarization is k
    length = sum(list(len(sent.split()) for sent in summ))
    return length / k


def get_preposition_size(node, pp):
    count = 0
    if not pp:
        for n in node.children:
            if n.POS == 'PP':  
                count += get_preposition_size(n,1)
    else:
        for n in node.children:
            count += get_preposition_size(n,1)
        if not len(node.children): 
            return 1
    return count


def contains_name(sent):
    ner = ne_chunk(pos_tag(word_tokenize(sent)))
    return int('PERSON' in str(ner))


def contains_number(sent):
    ner = ne_chunk(pos_tag(word_tokenize(sent)))
    return int('CD' in str(ner))  

     
def get_sentence_representation(sent,S):
    """
    Here, we care about a sentence's length with respect to k, a sentence's tfidf
    value, the words followed by PPs', etc
    """
    
    sent_vec = np.zeros((1,5),dtype=np.float64)
    sent_vec[0,0] = sent.top_tfidf_mean
    sent_vec[0,1] = sent.length/S.k 
    sent_vec[0,2] = get_preposition_size(sent.tree.root.node,0)/sent.tree.node_count
    sent_vec[0,3] = contains_name(sent.sentence)
    sent_vec[0,4] = contains_number(sent.sentence)

    return sent_vec



def get_tree_size(node):
    if not len(node.children): return node.depth, 1
    max_depth = node.depth
    total_count = 0
    for ch in node.children:
        md,count = get_tree_size(ch)
        max_depth = max(md, max_depth)
        total_count += count
    return max_depth, total_count


def update_sentence(sent,S):
    
    sent.depth, sent.tree.node_count = get_tree_size(sent.tree.root.node)
    sent.top_tfidf_mean, tfidf_list = get_sentence_tfidf(S.tfidf, S.tfs, sent)

    sent.word_count = len(sent.tree.root.node.get_words())
    if len(tfidf_list): 
        sent.word_list, sent.tfidf_vals = zip(*tfidf_list)
    else: 
        sent.word_list, sent.tfidf_vals = [], []
    


def get_compressed_sentence(sent, S):
    if not len(sent.pp_list) and not len(sent.subsentence_list): return sent
    pp_dict = {}
    new_sent = copy.deepcopy(sent)
    sent_node = new_sent.tree.root.node
    sig_words = new_sent.get_significant_words(0.5) # get top 0.5 words
    
    # delete subtrees with preposition as root if no significant words exist
    for pp in new_sent.pp_list:
        important = False
        sub = ' '.join(x for x in pp.get_words()) 
        for word in sig_words:
            if word in sub: important = True
        if not important: 
            pp.parent.children.remove(pp)
            new_sent.pp_list.remove(pp)

    # delete subsentences if no important words exist
    for s in new_sent.subsentence_list:
        important = False
        sub = ' '.join(x for x in s.get_words())
        count = 0
        for word in sig_words:
            if word in sub: 
                important = True
                count += 1
        if len(sig_words) and count / len(sig_words) > 0.7: # if the subsentence contains 
                                         # most of the important words
            new_sent = Sent(sub)
            tree = Tree()
            new_sent.tree = tree
            new_sent.tree.root.node = s
            
            update_sentence(new_sent,S)
            return new_sent
        
        if not important:
            s.parent.children.remove(s)
            new_sent.subsentence_list.remove(s)
            
    ns = sent_node.get_words()
    ns = list(filter(lambda x: x != None, ns))
    print('orig_sent',sent.sentence)
    ns = ' '.join(x for x in ns)
    new_sent.sentence = ns
    print('new_sent',ns)
    update_sentence(new_sent,S)
    
    return new_sent



def get_preposition_size(node, pp):
    count = 0
    if not pp:
        for n in node.children:
            if n.POS == 'PP':  
                count += get_preposition_size(n,1)
    else:
        for n in node.children:
            count += get_preposition_size(n,1)
        if not len(node.children): return 1
    return count
            
            
# get the sum of the tfidf value of all words in a sentence and divide its
def get_sentence_tfidf(tfidf, tfs_vec, sent):
    s,vals = sent.sentence,[]
    response = tfidf.transform([s])
    feature_names = tfidf.get_feature_names()
    word_list = []
    for word_idx in response.nonzero()[1]:
        if feature_names[word_idx].isalnum():
            vals.append(tfs_vec[0,word_idx])
            word_list.append((feature_names[word_idx],tfs_vec[0,word_idx]))
            
    # we only pick the top 25% tfidf values of the sentence
    #val = np.mean(sorted(vals,reverse=True)[:max(round(len(vals)/4),1)])
    val = np.mean(sorted(vals,reverse=True)[:5])
    if np.isnan(val): val = 0
    return val, word_list


def get_top_tfidf_words(tfidf,tfs_vec,sents,op):
    text = ' '.join(x for x in sents)
    response = tfidf.transform([text])
    feature_names = tfidf.get_feature_names()
    word_list = []
    vals_list = []
    for word_idx in response.nonzero()[1]:
        if feature_names[word_idx].isalnum():
            word_list.append(feature_names[word_idx])
            vals_list.append(tfs_vec[0,word_idx])
            
    sorted_vals = sorted(vals_list,reverse=True)
    inds_list = list(vals_list.index(x) for x in sorted_vals)
    sorted_words = list(np.array(word_list)[inds_list])
    if op == 'full':
        return sorted_words, sorted_vals
    elif type(op) == int:
        return sorted_words[:op],sorted_vals[:op]
    else:
        error('Invalid argument')
        
        
def get_positions(summ,article):
    front_pos = np.zeros((1,5))
    back_pos = np.zeros((1,5))
    n = len(article)
    pos_list = []
    for sent in summ:
        ind = list(i for i in range(len(article)) if sent in article[i])
        if len(ind): pos_list.append(ind[0])
    for p in pos_list:
        if p < 5: front_pos[0,p] = 1
        if n-p < 5: back_pos[0,n-p] = 1
    return np.hstack([front_pos,back_pos])


def check_top_words_inclusion(words,top_words):
    included = np.zeros((1,20)) # it denotes the 10 words in the article with
                                 # the top tfidf scores
    for word in words:
        if word in top_words: 
            included[0,top_words.index(word)] = 1
            
    return included

         
def calculate_final_reward(results,references, rouge_num,file):
    f_pre1,f_post1 = 'summaries/summary', '.txt'
    f_pre2,f_post2 = 'references/reference', '.txt'
    r_list = []
    for i in range(len(results)):
        summary = results[i]
        summary = ' '.join(x for x in summary)
        reference = references[i]
        s_name = f_pre1 + str(1) + f_post1
        r_name = f_pre2 + str(1) + f_post2
        f1 = open(s_name,'w')
        f1.write(summary)
        f2 = open(r_name,'w')
        f2.write(reference)
        f1.close()
        f2.close()
        r = Rouge155()
        if LOC == 1:
            r.system_dir = '/Users/xiahu/Documents/Projects/Summarization/summaries/'
            r.model_dir = '/Users/xiahu/Documents/Projects/Summarization/references/'            
        else:
            r.system_dir = '/home/home3/xhu/Documents/Project/Summarization/summaries/'
            r.model_dir = '/home/home3/xhu/Documents/Project/Summarization/references/'
        r.system_filename_pattern = 'summary(\d+).txt'
        r.model_filename_pattern = 'reference(\d+).txt'
        
        output = r.convert_and_evaluate()
        op = re.split('\n',output)
        rouge = "ROUGE-" + str(rouge_num)
        r = 0
        for line in op:
            if (rouge + " Average_R") in line:
                r = float(re.split('\(',re.split(':',line)[1])[0])
        print('r',r)
        r_list.append(r)
        
    file.write('\nTesting results:\n')
    s = ' '.join(str(x) for x in r_list)
    print(s)
    print('mean',np.mean(r_list))
    s += '\n'
    file.write(s)
    s1 = 'mean scores is: ' + str(np.mean(r_list)) + '\n'
    file.write(s1)
    return r_list


def calculate_final_rewards(results,references, rouge_num,file):
    f_pre1, f_post1 = 'summaries/summary', '.txt'
    f_pre2, f_post2 = 'references/reference', '.txt'
    
    for i in range(len(results)):
        summary = results[i]
        summary = ' '.join(x for x in summary)
        reference = references[i]
        s_name = f_pre1 + str(i+1) + f_post1
        r_name = f_pre2 + str(i+1) + f_post2
        f1 = open(s_name, 'w')
        f1.write(summary)
        f2 = open(r_name, 'w')
        f2.write(reference)
        f1.close()
        f2.close()

    r = Rouge155()
    if LOC == 1:
        r.system_dir = '/Users/xiahu/Documents/Projects/Summarization/summaries/'
        r.model_dir = '/Users/xiahu/Documents/Projects/Summarization/references/'
    else:
        r.system_dir = '/home/home3/xhu/Documents/Project/Summarization/summaries/'
        r.model_dir = '/home/home3/xhu/Documents/Project/Summarization/references/'
        
    r.system_filename_pattern = 'summary(\d+).txt'
    r.model_filename_pattern = 'reference(\d+).txt'
    output = r.convert_and_evaluate()
    op = re.split('\n',output)
    rouge = "ROUGE-" + str(rouge_num)
    r, p, f = 0, 0, 0
    for line in op:
        if (rouge + " Average_R") in line:
            r = float(re.split('\(', re.split(':',line)[1])[0])
        elif (rouge + " Average_P") in line:
            p = float(re.split('\(', re.split(':',line)[1])[0])
        elif (rouge + " Average_F") in line:
            f = float(re.split('\(', re.split(':',line)[1])[0])
    print('r,p,f',r,p,f)
    results = str(r) + ' ' + str(p) + ' ' + str(f)
    file.write(results)
    return r 


def get_reward(S, rouge_num):
    summary,reference = S.summ, S.reference
    length = sum(list(len(sent.split()) for sent in summary))
    if length > S.k:
        return 0, 0, 0 # if the length exceeds k, we return reward 0
    summ = ' '.join(x for x in summary)
    f1 = open('summaries/summary1.txt','w')
    f1.write(summ)
    f1.close()
    f2 = open('references/reference1.txt','w')
    f2.write(reference)
    f2.close()

    r = Rouge155()
    if LOC == 1:
        r.system_dir = '/Users/xiahu/Documents/Projects/Summarization/summaries/'
        r.model_dir = '/Users/xiahu/Documents/Projects/Summarization/references/'
    else:
        r.system_dir = '/home/home3/xhu/Documents/Project/Summarization/summaries/'
        r.model_dir = '/home/home3/xhu/Documents/Project/Summarization/references/'
        
    r.system_filename_pattern = 'summary(\d+).txt'
    r.model_filename_pattern = 'reference(\d+).txt'

    output = r.convert_and_evaluate()
    op = re.split('\n',output)
    rouge = "ROUGE-" + str(rouge_num)
    r,p,f = 0,0,0
    for line in op:
        if (rouge + " Average_R") in line:
            r = float(re.split('\(',re.split(':',line)[1])[0])
        elif (rouge + " Average_P") in line:
            p = float(re.split('\(',re.split(':',line)[1])[0])
        elif (rouge + " Average_F") in line:
            f = float(re.split('\(',re.split(':',line)[1])[0])
    print('r,p,f',r,p,f,len(summary))
    return r,p,f



def get_compression_action(S,sent,epsilon,model_comp):
    prob = np.zeros(2)
    sent_vec = get_sentence_representation(sent,S)
    new_sent = get_compressed_sentence(sent,S)
    comp_vec = get_sentence_representation(new_sent,S)
    sent_val = model_comp.predict(sent_vec)
    comp_val = model_comp.predict(comp_vec)
    if comp_val >= sent_val and len(new_sent.sentence) < len(sent.sentence): 
        prob[0],prob[1] = epsilon,1-epsilon
    else: prob[0],prob[1] = 1-epsilon,epsilon
    action_comp = np.random.choice(np.arange(2),p=prob)
    return action_comp, new_sent, comp_vec

    
def run_test(T_list,model,tfidf,tfs,k,model_comp,epsilon):
    state_list = []
    summ_list = []
    prob = np.zeros(2)
    for i in range(len(T_list)):
        T = T_list[i]
        sent_objects = T.sent_obj_list

        if sent_objects != None:
            S = State(T,k,tfidf,tfs[i,:])
            sentences_tmp = copy.deepcopy(sent_objects)

            while len(sentences_tmp):
                sent = random.choice(sentences_tmp)
                sentences_tmp.remove(sent)
                new_vec,cur_vec = S.get_new_state_vec(sent),S.feature_vec
                new_val,cur_val = model.predict(new_vec),model.predict(cur_vec)

                if new_val > cur_val: 
                    prob[0], prob[1] = epsilon, 1-epsilon
                else: 
                    prob[0], prob[1] = 1-epsilon, epsilon

                action = np.random.choice(np.arange(2), p=prob)

                if action == 1:
                    action_comp, new_sent, _ = get_compression_action(S, sent, epsilon, model_comp)
                    if action_comp == 1: 
                        sent = new_sent
                    if S.word_count + len(sent.sentence.split()) <= S.k:
                        S.update_state(sent, new_vec, action) 
                    if S.word_count / S.k > 0.95: 
                        break

                # if we are far from reaching the limit length
                if not len(sentences_tmp) and S.word_count < S.k * 0.7: 

                    for sent in sent_objects:

                        if sent.sentence not in S.summ:
                            sentences_tmp.append(sent)
            
            state_list.append(S)
            summ_list.append(S.summ)
            
    return state_list,summ_list



def copy_state(S):
    s_new = State(S.T,S.k,S.tfidf,S.tfs)
    s_new.summ = copy.deepcopy(S.summ)
    s_new.word_count = S.word_count
    s_new.top_tfidf_words = copy.deepcopy(S.top_tfidf_words)
    s_new.top_tfidf_vals = copy.deepcopy(S.top_tfidf_vals)
    
    return s_new


def learn(S,T,model,model_comp,epsilon):
    max_val = 0 
    rouge_num = 1
    sent_objects = T.sent_obj_list
    sentences_tmp = copy.deepcopy(sent_objects)
    while len(sentences_tmp):
        sent = random.choice(sentences_tmp)
        sentences_tmp.remove(sent)
        
        new_vec = S.get_new_state_vec(sent)
        print('new_vec type',type(new_vec))
        cur_vec = S.feature_vec
        print('cur_vec',type(cur_vec))
        new_val,cur_val = model.predict(new_vec),model.predict(cur_vec)
        prob = np.zeros(2)
        if new_val > cur_val: prob[0],prob[1] = epsilon,1-epsilon
        else: prob[0],prob[1] = 1-epsilon,epsilon
        action = np.random.choice(np.arange(2),p=prob)
        if action == 1: 
            action_comp,new_sent,comp_vec = get_compression_action(S,sent,epsilon,model_comp)
            if action_comp == 1: sent = new_sent
        S_tmp = copy_state(S)
        S_tmp.update_state(sent,new_vec,action) 
        if action == 1:
            val,p,f = get_reward(S_tmp,rouge_num) # we take the recall value
            print('learn: new_vec', new_vec)
            print('learn: val', val)
            model.partial_fit(new_vec,np.array([val]))
            max_val = max(max_val,val)
            if action_comp == 1: model_comp.partial_fit(comp_vec, np.array([p]))
            if val != 0: 
                S = S_tmp
                if S.word_count > S.k * 0.95: return max_val
            else: return max_val
            
        # if we are far from reaching the limit length
        if len(sentences_tmp) and S.word_count < S.k * 0.7: 
            for sent in sent_objects:
                if sent.sentence not in S.summ:
                    sentences_tmp.append(sent)
            
    return max_val #, model

# helper function taken from online for deleting rows in a sparse matrix
#https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices

def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])
    
    
def get_processed_data(data_list, tfidf, tfs):
    del_list = []
    T_list,r_values = [],[]
    for i in range(len(data_list)):
        T = data_list[i]
        T.sent_obj_list = get_sentence_info(T.article, tfidf, tfs[i]) 
        T.sent_tokens_list = nltk.sent_tokenize(T.article)
        if T.sent_obj_list != None:
            T_list.append(T)
        else: del_list.append(i)
    del_list = sorted(del_list,reverse=True)  
    print('del_list')
    for d in del_list: del data_list[d]
    tfidf,tfs = tf_idf(data_list)      
    
    return T_list, data_list, tfidf, tfs


def get_references(T_list):
    return list(T.abstract for T in T_list)
            
        
def value_function_approximation(): # main function
    multiprocess = 0
    num_episodes = 100 #100
    batch_size = 1 # each doc contains 1000 article-abstract pairs
    alpha = 0.005 # learning rate
    epsilon = 0.1
    k = 100
    itr = 30 # 50 iterations per article
    data_list = read_articles_and_summaries()[:10] # a list of Text objects
    l = int(len(data_list) / 2) 
    linear = 1

    if linear:
        model = SGDRegressor(learning_rate="constant",alpha=alpha)
    else:
        model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', alpha=0.00,\
                          learning_rate='constant', learning_rate_init=0.01, max_iter=1000)
    
    model.partial_fit(np.zeros((1,36)),[0])
    model_comp = SGDRegressor(learning_rate="constant", alpha=alpha)
    model_comp.partial_fit(np.zeros((1,5)),[0])
    
    test_list = data_list[l:]
    data_list = data_list[:l]
    tfidf, tfs = tf_idf(data_list)
    tfidf1, tfs1 = tf_idf(test_list)

    if multiprocess:
        total_sent_list, total_T_list = [], []
        pool = multiprocessing.Pool(processes=10)
        for i in range(int(len(data_list)/10)):
            sent_list,T_list = get_sentence_info2(data_list[i*10:(i+1)*10], tfidf, tfs[i*10:(i+1)*10,:], pool)
            total_sent_list += sent_list
            total_T_list += T_list
    # ================ Get processed data ======================
    T_list, data_list, tfidf, tfs = get_processed_data(data_list, tfidf, tfs)
    T_test_list, test_list, tfidf1, tfs1 = get_processed_data(test_list, tfidf1, tfs1)
    # ================= Training model =========================
    r_values = []
    for i,T in enumerate(T_list):
        rs = []
        for j in range(itr):
            S = State(T, k, tfidf, tfs[i,:])
            r = learn(S, T, model, model_comp, epsilon) #, model, model_comp = 
            if linear: 
                print('linear: model.coef_', model.coef_)
            else: 
                print('not linear: model.coef_', model.coefs_)
            rs.append(r)
        r_values.append(rs)
    # ================= Record parameters =======================
   
    num = len(r_values[0])
    format_str = '%2.2f ' * num
    file = open('results_linear1400', 'w')
    string = str(l) + ' for training and ' + str(l) + ' for testing\n'
    file.write(string)

    for rs in r_values: 
        s = format_str % tuple(rs)
        print(s)
        file.write(s)

    if linear: 
        file.write(str(model.coef_))
    else: 
        file.write(str(model.coefs_))
    file.write('\n')
    
    final_states, results = run_test(T_test_list, model, tfidf1, tfs1, k, model_comp, 0.01)
    reference_list = get_references(T_test_list)
    calculate_final_reward(results, reference_list, 1, file)
    calculate_final_rewards(results, reference_list, 1, file)
    
    file.close()
    return T_list, model, tfidf, tfs, k, model_comp



T_list, model, tfidf, tfs, k, model_comp = value_function_approximation()









