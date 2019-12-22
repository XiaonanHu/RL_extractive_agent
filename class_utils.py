class Node:
    def __init__(self, depth):
        self.POS = None
        self.word = None
        self.val = 0
        self.isWord = False
        self.isPhrase = False
        self.isRoot = False
        self.isSentence = False
        self.children = []
        self.parent = False
        self.depth = depth

    def addPos(self, POS):
        self.POS = POS

    def addWordVal(self, word, val):
        self.word = word
        self.val = val
        self.isWord = True

    def addPosWordVal(self, POS, word, val):
        self.POS = POS
        self.word = word
        self.val = val
        self.isWord = True
        return self

    def get_word(self):
        return self.word

    def get_words(self):
        if self.isWord:
            return [self.get_word()]
        else:
            words = list(x.get_words() for x in self.children)
            if len(words) > 1: 
                return list(itertools.chain(*words))
            elif len(words) == 1: 
                return words[0]
            else: 
                return []
        
    def add_child(self,node):
        self.children.append(node)
        node.parent = self

    def add_children(self,node_list):
        self.children += node_list
        for node in node_list: 
            node.parent = self
            
class Root:
    def __init__(self):
        self.children = []
        self.node = Node(1)

    def add_child(self,depth):
        node = Node(depth)
        self.node.children.append(node)
        node.parent = self.node
        return node
        

class Tree:
                
    def __init__(self):
        self.root = Root()
        self.node_count = 0


class Sent:
    def __init__(self,sent):
        self.sentence = sent
        self.length = len(sent.split())
        self.tree = None
        self.depth = 0
        self.word_list = []
        self.tfidf_vals = []
        # the mean tfidfs of words that have the highest 25% tfidf values
        self.top_tfidf_mean = 0 
        self.subsentence_list = []
        self.pp_list = []
        
    def get_significant_words(self,ratio):
        if len(self.tfidf_vals): 
            th = sorted(self.tfidf_vals,reverse=True)[int(len(self.word_list)*ratio)]
            inds = list(i for i in range(len(self.word_list)) if self.tfidf_vals[i] >= th )
            return np.array(self.word_list)[inds]
        else: 
            return np.array([])
    
    def __lt__(self,sent):
        return self.top_tfidf_mean < sent.top_tfidf_mean
        

class State:
    def __init__(self,T,k,tfidf,tfs_vec): # here tfs is a vector
        self.summ = []
        self.article = T.sent_tokens_list
        self.reference = T.abstract
        self.feature_vec = np.zeros((1,36))
        self.feature_vec[0,5] = 1
        self.word_count = 0
        self.tfidf = tfidf
        self.tfs = tfs_vec
        self.value = 0
        self.T = T
        self.k = k
        w,v = get_top_tfidf_words(tfidf,tfs_vec,self.article,'full')
        self.top_tfidf_words = w[:20] # we only care about the top 10 words
        self.top_tfidf_vals = v[:20]
    

    def get_new_state_vec(self,sent):
        
        new_feature = np.copy(self.feature_vec[0,:6]).reshape((1,6))
        # Notes: should I consider the presence of emotion words (words
        # with strong sentiments 
        # should I consider the current sentence's 
        top_words,top_vals = get_top_tfidf_words(self.tfidf,self.tfs,self.summ+[sent.sentence],'full')
        if len(top_vals):
            new_feature[0,0] = top_vals[0]
            new_feature[0,1] = np.mean(top_vals[:3])
            new_feature[0,2] = np.mean(top_vals[:5])
            new_feature[0,3] = np.mean(top_vals[:10])
        # sum of the dep/len ratio
        new_feature[0,4] += (sent.depth/sent.length-new_feature[0,4])/(len(self.summ)+1)
        new_feature[0,5] = 1-get_length_ratio(self.summ+[sent.sentence], self.k) # total word count to k
        # among the 5 sentences at the beginning and the end of the article, we
        # record whether they have been included in the representation
        endpos_occupancy = get_positions(self.summ+[sent.sentence],self.article) # length 10 vector
        topWords_included = check_top_words_inclusion(top_words,self.top_tfidf_words) # len
        return np.hstack([new_feature,endpos_occupancy,topWords_included])

    def update_state(self,sent,new_feature,action):
        if action:
            self.summ += [sent.sentence]
            self.feature_vec = new_feature
            self.word_count += len(sent.sentence.split())
        
        