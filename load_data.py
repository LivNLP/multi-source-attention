"""

preprocess data


"""
import sys
import glob
import pickle,os
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from operator import add


# review list without labels (bigram version)
def review_list(fname,sent=False):
    if sent==False:
        return list([(line.strip().split()) for line in open(fname)])#.replace("__"," ")
    else:
        return list([line for line in open(fname)])

# review list with labels (bigram version)
def review_list_with_labels(fname,label):
    return list([(line.strip().split(),label) for line in open(fname)])

# prepare test data with labels for one domain
def prepare_train(domain):
    pos_file = "../../pivot-selection/data/%s/train.positive" % domain
    neg_file = "../../pivot-selection/data/%s/train.negative" % domain
    return review_list_with_labels(pos_file,1)+review_list_with_labels(neg_file,0)

# prepare test data with labels for one domain
def prepare_test(domain,label=True,sent=False):
    pos_file = "../../pivot-selection/data/%s/test.positive" % domain
    neg_file = "../../pivot-selection/data/%s/test.negative" % domain
    if label == True:
        return review_list_with_labels(pos_file,1)+review_list_with_labels(neg_file,0)
    else:
        return review_list(pos_file,sent)+review_list(neg_file,sent)

# prepare source labelled data: S_L+, S_L-
def prepare_labelled(domain,sent=False):
    pos_file = "../../pivot-selection/data/%s/train.positive" % domain
    neg_file = "../../pivot-selection/data/%s/train.negative" % domain
    return review_list(pos_file,sent),review_list(neg_file,sent)

# prepare unlabelled from source and target
def prepare_unlabelled(source,target,sent=False):
    src_file = "../../pivot-selection/data/%s/train.unlabeled" % source
    tgt_file = "../../pivot-selection/data/%s/train.unlabeled" % target
    return review_list(src_file,sent),review_list(tgt_file,sent)

###############################################################

def load_filtered_glove(filtered_features,gloveFile):
    print "Loading GloVe Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        if word in filtered_features:     
            model[word] = embedding
        if word.replace('.','__') in filtered_features:
            model[word.replace('.','__')] = embedding
    print "After filtering, ",len(model)," words loaded!"
    return model

def collect_all():
    files = glob.glob('../../pivot-selection/data/*/*')
    print files
    all_data = list()
    for fname in files:
        all_data += review_list(fname)
    all_features = set(x for reivew in all_data for x in reivew)
    print len(all_features)
    return all_features

def save_new_glove_model():
    all_features = collect_all()
    path = "../../pivot-selection/data/glove.42B.300d.txt"
    embeddings = load_filtered_glove(all_features,path)
    new_model = save_obj(embeddings,'glove.filtered')
    print "Saved"
    pass

###############################################################
# SIF  embedding
def make_sentence_vector(sentence,embeddings):
    temp = np.array(np.zeros(300))
    # print embeddings
    count = 0
    for word in sentence:
        count += 1
        if word in embeddings:
            temp = map(add, temp, np.array(embeddings[word]))
        else:
            # print "%s is not in pretrained embeddings"%word
            temp = map(add,temp,np.array(np.zeros(300)))
            # temp =np.add(temp,embedding_for_word(word,pre_model,self_model))
        # print len(temp)
    if count == 0:
        temp = np.array(temp)
    else:
        temp = np.multiply(np.array(temp) ,(1.0/float(count)))
    return temp

def set_up_data(sentences,embeddings):
    u = list()
    for sent in sentences:
        u.append(make_sentence_vector(sent,embeddings))
    return u

def prepare_data(source,target):
    embeddings = load_obj('glove.filtered')
    train = prepare_train(source)
    test = prepare_test(target)
    
    train_data = [reivew for reivew,_ in train]
    test_data = [reivew for reivew,_ in test]

    X_train = set_up_data(train_data,embeddings)
    y_train = [label for _,label in train]
    
    X_test = set_up_data(test_data,embeddings)
    y_test = [label for _,label in test]
    return X_train,y_train,X_test,y_test

def prepare_source(source):
    train = prepare_train(source)
    embeddings = load_obj('glove.filtered')
    train_data = [reivew for reivew,_ in train]
    X_train = set_up_data(train_data,embeddings)
    y_train = [label for _,label in train]
    return X_train,y_train


def prepare_evaluate(target):
    test = prepare_test(target)
    test_data = [reivew for reivew,_ in test]
    embeddings = load_obj('glove.filtered')
    X_test = set_up_data(test_data,embeddings)
    y_test = [label for _,label in test]
    return X_test,y_test

def set_up_labelled(domain):
    pos,neg = prepare_labelled(domain)
    embeddings = load_obj('glove.filtered')
    src_pos = set_up_data(pos,embeddings)
    src_neg = set_up_data(neg,embeddings)
    return src_pos,src_neg

def set_up_unlabelled(source,target,src_ON=True):
    src,tgt = prepare_unlabelled(source,target)
    embeddings = load_obj('glove.filtered')
    src_un = None
    # if we need source unlabelled data
    if src_ON ==True:
        src_un = set_up_data(src,embeddings)
    tgt_un = set_up_data(tgt,embeddings)
    return src_un,tgt_un

###############################################################
# get the origin sentence and source domain by index
def index_to_source_sentence(sent_index,target,domains,X_index=None):
    sents = []
    if X_index == None:
        for source in domains:
            # print source
            if target == source:
                # print "this is target domain"
                continue
            else:
                temp_pos,temp_neg = prepare_labelled(source,sent=True)
                temp = temp_pos+temp_neg
                labels = concatenate(np.ones(len(temp_pos)),np.zeros(len(temp_neg)))
                # print len(temp)
                sents+=[(x,source,label) for x,label in zip(temp,labels)]
    else:
        for source in domains:
            # print source
            if target == source:
                # print "this is target domain"
                continue
            else:
                temp_pos,temp_neg = prepare_labelled(source,sent=True)
                temp = temp_pos+temp_neg
                labels = concatenate(np.ones(len(temp_pos)),np.zeros(len(temp_neg)))
                # print len(temp)
                sents.append([(x,source,label) for x,label in zip(temp,labels)])
        new_sents = []
        for sents_domain,index in zip(sents,X_index):
            new_sents += list(np.array(sents_domain)[index])
        sents = new_sents
        # sents = [sent for sent in sents]

    # print len(sents)
    sentence,source,label = sents[sent_index]
    # print sentence
    # print source,label
    return source,float(label),sentence



###############################################################

def concatenate(a,b):
    if len(a)>0 and len(b)>0:
        return np.concatenate((a,b),axis=0)
    elif len(a)==0 and len(b)!=0:
        print "a empty!!"
        return np.array(b)
    elif len(b)==0 and len(a)!=0: 
        print "b empty!! length a = %d"%len(a)
        return np.array(a)
    else:
        return list()
    pass


###############################################################
# baseline: NoAdapt
# default: LogisticRegression
def baseline(X_train,y_train,X_test,y_test,clf='lr'):
    clf_func = get_clf_func(clf)
    clf_func.fit(X_train,y_train)
    pred = clf_func.predict(X_test)
    acc = accuracy_score(y_test, pred)
    # print acc
    return acc

# stroe all the classifiers and get the right one to be used
def get_clf_func(clf='lr',k=15):
    if clf == 'knn':
        clf_func = KNeighborsClassifier(n_neighbors=k) # knn
    elif clf == 'lr':
        clf_func = LogisticRegression()#C=0.0001
    elif clf == 'tree':
        clf_func = DecisionTreeClassifier()
    elif clf == 'naive':
        clf_func = GaussianNB()
    else: # nn
        clf_func = LogisticRegression()#MLPClassifier()
    return clf_func

###############################################################
# save and load after preprocessing
def save_obj(obj, name):
    filename = '../data/'+name + '.pkl'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print filename, 'saved.'

def load_obj(name):
    with open('../data/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)





###############################################################

if __name__ == '__main__':
    save_new_glove_model()
    # source = "bs1"
    # target = "d1"
    # target = "books"
    # domains = ["books","dvd","electronics","kitchen"]
    # sent_index = 1922
    # index_to_source_sentence(sent_index,target,domains)
    
    # X_train,y_train,X_test,y_test = prepare_data(source,target)
    

