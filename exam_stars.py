"""
Examine the pesudo-labelled datasets
1. only sort by prob
2. only sort by sim
3. sort by prob, then sort by sim
4. sort by sim, then sort by prob
5. prob + sim
6. prob x sim


"""

import sys
import itertools
from mlxtend.classifier import EnsembleVoteClassifier
from load_data import prepare_evaluate,set_up_labelled,set_up_unlabelled,get_clf_func,save_obj,load_obj,index_to_source_sentence
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score


def exam_tops(target,k=100,theta=0.5,sort_prob="dsc",sim="target"):
    tgt_un =load_obj("%s/X_un"%target)
    print "loaded data %s." %target
    tmp_name = target.upper()[0] if "large" not in target else "large/"+target.upper()[6]
    eclf = load_obj('%s_eclf'%tmp_name)
    print "loaded trained classifier"
    tgt_sim = ""
    if sim == "target":
        tgt_sim = load_obj("%s/tgt_sim"%target)  
        print "loaded target similarity"
    else:
        tgt_sim =load_obj("%s/src_sim"%target)
        print "loaded source similarity"
    labels_proba = eclf.predict_proba(tgt_un)
    
    resFile = ""
    if sim == "target":
        if "large" not in target:
            resFile = open("../work/params/%s/exam_stars_%s_n_%s.csv"%(sort_prob,target,k),'w')
        else:
            resFile = open("../work/params/%s/exam_stars_%s_n_%s.csv"%(sort_prob,target.replace("large/",""),k),'w')
    else:
        if "large" not in target:
            resFile = open("../work/params/src/%s/exam_stars_%s_n_%s.csv"%(sort_prob,target,k),'w')
        else:
            resFile = open("../work/params/src/%s/exam_stars_%s_n_%s.csv"%(sort_prob,target.replace("large/",""),k),'w')
    resFile.write("method,num_pos,num_neg,acc\n")


    pos_star,neg_star,pos_len,neg_len = just_prob(tgt_un,labels_proba,tgt_sim,k,theta,sort_prob)
    acc = test(target,pos_star,neg_star)
    resFile.write("just_prob,%d,%d,%f\n"%(pos_len,neg_len,acc))
    resFile.flush()

    pos_star,neg_star,pos_len,neg_len = just_sim(tgt_un,labels_proba,tgt_sim,k,theta)
    acc = test(target,pos_star,neg_star)
    resFile.write("just_sim,%d,%d,%f\n"%(pos_len,neg_len,acc))
    resFile.flush()

    pos_star,neg_star,pos_len,neg_len = prob_sim(tgt_un,labels_proba,tgt_sim,k,theta,sort_prob)
    acc = test(target,pos_star,neg_star)
    resFile.write("prob_sim,%d,%d,%f\n"%(pos_len,neg_len,acc))
    resFile.flush()

    pos_star,neg_star,pos_len,neg_len = sim_prob(tgt_un,labels_proba,tgt_sim,k,theta,sort_prob)
    acc = test(target,pos_star,neg_star)
    resFile.write("sim_prob,%d,%d,%f\n"%(pos_len,neg_len,acc))
    resFile.flush()

    pos_star,neg_star,pos_len,neg_len = add_prob_sim(tgt_un,labels_proba,tgt_sim,k,theta,sort_prob)
    acc = test(target,pos_star,neg_star)
    resFile.write("prob+sim,%d,%d,%f\n"%(pos_len,neg_len,acc))
    resFile.flush()

    pos_star,neg_star,pos_len,neg_len = multi_prob_sim(tgt_un,labels_proba,tgt_sim,k,theta,sort_prob)
    acc = test(target,pos_star,neg_star)
    resFile.write("prob*sim,%d,%d,%f\n"%(pos_len,neg_len,acc))
    resFile.flush()

    resFile.close()
    pass

# just probs
def just_prob(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    # tgt_sim is not used!
    pos_list = [(x,y[1],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star = starize(pos_list,neg_list,k/2)
    return pos_star,neg_star,len(pos_star),len(neg_star)

# just labels
def just_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    # labels_proba is not used
    pos_list = [(x,y[1],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=2)
    neg_list = sort_by_index(neg_list,index=2)
    pos_star,neg_star = starize(pos_list,neg_list,k/2)
    return pos_star,neg_star,len(pos_star),len(neg_star)

# pesudo labels first
def prob_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    pos_list = [(x,y[1],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    num = min(min(k,len(pos_list)),len(neg_list))
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)[:num]
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)[:num]
    pos_list = sort_by_index(pos_list,index=2)
    neg_list = sort_by_index(neg_list,index=2)
    pos_star,neg_star = starize(pos_list,neg_list,k/2)
    return pos_star,neg_star,len(pos_star),len(neg_star)

# tgt_sim first
def sim_prob(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    sum_dict = [(x,y,z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim)]
    num = min(k*2,len(sum_dict))
    sum_dict = sort_by_index(sum_dict,index=2)[:num]
    pos_list = [(x,y[1],z) for x,y,z in sum_dict if y[1] >theta]
    neg_list = [(x,y[0],z) for x,y,z in sum_dict if y[0] >theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star = starize(pos_list,neg_list,k/2)
    return pos_star,neg_star,len(pos_star),len(neg_star)

# prob+sim
def add_prob_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    pos_list = [(x,y[1]+z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0]+z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star = starize(pos_list,neg_list,k/2)
    return pos_star,neg_star,len(pos_star),len(neg_star)

# prob*sim
def multi_prob_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    pos_list = [(x,(y[1])*z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,(y[0])*z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star = starize(pos_list,neg_list,k/2)
    return pos_star,neg_star,len(pos_star),len(neg_star)

# 1 = probs
# 2 = sim
def sort_by_index(list_to_sort,index=1,sorting='dsc'):
    if sorting == "dsc": 
        list_to_sort.sort(lambda x,y: -1 if x[index]>y[index] else 1)
    else: #asc
        list_to_sort.sort(lambda x,y: -1 if x[index]<y[index] else 1)
    return list_to_sort

def starize(pos_list,neg_list,k):
    temp = min(len(pos_list),len(neg_list))
    if k < temp:
        pos_star = [x for x,y,z in pos_list[:k]]
        neg_star = [x for x,y,z in neg_list[:k]]
    else:
        pos_star = [x for x,y,z in pos_list[:temp]]
        neg_star = [x for x,y,z in neg_list[:temp]]
    return pos_star,neg_star


################################################################

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
        return np.array([])
    pass

def test(target,pos_star,neg_star,clf='lr'):
    X_train = concatenate(pos_star,neg_star)
    y_train = concatenate(np.ones(len(pos_star)),np.zeros(len(neg_star)))
    X_test = load_obj("%s/X_test"%target)
    y_test = load_obj("%s/y_test"%target)
    # print X_train.shape,y_train.shape
    if len(X_train) == 0:
        return 0.0
    clf_func = get_clf_func(clf)
    clf_func.fit(X_train,y_train)
    pred = clf_func.predict(X_test)
    acc = accuracy_score(y_test,pred) if "large" not in target else f1_score(y_test,pred,average='macro')
    # print acc*100
    return acc*100

if __name__ == '__main__':
    if len(sys.argv)>5:
        target = sys.argv[1]
        k = int(sys.argv[2])
        theta = float(sys.argv[3])
        sort_prob = sys.argv[4]
        sim = sys.argv[5]
        print "target:",target,
        print "k:", k
        print "theta:",theta
        print "sort_prob:",sort_prob
        print "sim:",sim
        exam_tops(target,k,theta,sort_prob,sim)
    elif len(sys.argv)>4:
        target = sys.argv[1]
        k = int(sys.argv[2])
        theta = float(sys.argv[3])
        sort_prob = sys.argv[4]
        print "target:",target,
        print "k:", k
        print "theta:",theta
        print "sort_prob:",sort_prob
        exam_tops(target,k,theta,sort_prob)
    elif len(sys.argv) >3:
        target = sys.argv[1]
        k = int(sys.argv[2])
        theta = float(sys.argv[3])
        print "target:",target,
        print "k:", k
        print "theta:",theta
        exam_tops(target,k,theta)
    elif len(sys.argv) >2:
        target = sys.argv[1]
        k = int(sys.argv[2])
        print "target:",target,
        print "k:", k
        exam_tops(target,k)
    elif len(sys.argv) >1 :
        target = sys.argv[1]
        print "target:",target,
        exam_tops(target)
    else:
        print "usage:<target,k,theta> or <target,k> or <target>"
