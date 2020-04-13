"""
Domain Similarity Matrix
1. pesudo labelling T_U (select top k = 2000, pos+neg)
2. a matrix for these word vectors with source domains
3. source instances selection

"""
import sys
import itertools
from mlxtend.classifier import EnsembleVoteClassifier
from load_data import prepare_evaluate,set_up_labelled,set_up_unlabelled,get_clf_func,save_obj,load_obj,index_to_source_sentence
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sorting_methods import find_best_method
from scipy.special import softmax
from sklearn.preprocessing import normalize


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
        return list()
    pass

################################################################

def test(target):
    X_test = []
    y_test = []
    if "large" not in target:
        X_test, y_test = prepare_evaluate(target)
        eclf = load_obj('%s_eclf'%(target.upper()[0]))
    else:
        X_test = load_obj("%s/X_test"%target)
        y_test = load_obj("%s/y_test"%target)
        tmp_name = "large/"+target.upper()[6]
        eclf = load_obj('%s_eclf'%tmp_name)
    pred = eclf.predict(X_test)
    acc = accuracy_score(y_test,pred) if "large" not in target else f1_score(y_test,pred,average='macro')
    print acc
    pass

def predict_tops(target,k=2000,theta=0.5):
    # source = "d1"
    tgt_un =np.array(load_obj("%s/X_un"%target))
    print "loaded data %s." %target
    eclf = ""
    # if "large" not in target:
    #     eclf = load_obj('%s_eclf'%(target.upper()[0]))
    # else:
    #     tmp_name = "large/"+target.upper()[6]
    #     eclf = load_obj('%s_eclf'%tmp_name)
    eclf = load_obj("%s/joint_clf"%target)
    # eclf = load_obj("%s/self_clf"%target)
    # eclf = load_obj("%s/tri_clf"%target)
    print "loaded trained classifier"
    tgt_sim = load_obj("%s/tgt_sim"%target)
    print "loaded target similarity"
    labels_proba = eclf.predict_proba(tgt_un)
    best_acc = 0.0
    best_sorting = ""
    best_pos_star = ""
    best_neg_star = ""
    best_pos_start = 0.0
    best_pos_end = 0.0
    best_neg_start = 0.0
    best_neg_end = 0.0
    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,acc,method,pos_proba,neg_proba\
        = find_best_method(target,tgt_un,labels_proba,tgt_sim,k,theta,'asc')
    if best_acc< acc:
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_acc,best_method,best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,\
            acc,method,pos_proba,neg_proba
        best_sorting = "asc"
    
    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,acc,method,pos_proba,neg_proba\
        = find_best_method(target,tgt_un,labels_proba,tgt_sim,k,theta,'dsc')
    if best_acc< acc:
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_acc,best_method,best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,\
            acc,method,pos_proba,neg_proba
        best_sorting = "dsc"

    print "got pseudo labels"
    print "pos:",len(best_pos_star),best_pos_start,best_pos_end
    print "neg:",len(best_neg_star),best_neg_start,best_neg_end
    # save_obj(pos_star,"%s/pos_star"%target)
    # save_obj(neg_star,"%s/neg_star"%target)
    return best_pos_star,best_neg_star,best_acc,best_method,best_sorting,best_pos_proba,best_neg_proba
    # return pos_start,pos_end,len(pos_star),neg_start,neg_end,len(neg_star)


##################################################################
# computer psi and filter by top k (source instances selection)
def compute_psi(target,k=None):
    pos_star = load_obj('%s/pos_star'%target)
    neg_star = load_obj('%s/neg_star'%target)
    star_matrix = concatenate(pos_star,neg_star)
    # print star_matrix
    X_joint = load_obj('%s/X_joint'%target)
    y_joint = load_obj('%s/y_joint'%target)
    # print np.array(X_joint).shape
    src_cost = load_obj("%s/src_cost"%target)
    # X_train = get_sents(X_joint)
    # print np.matmul(star_matrix,X_train.T)
    # psi_matrix = np.dot(star_matrix,X_train.T).T
    #softmax(np.dot(star_matrix,X_train.T).T)
    # print k
    if k == None:
        psi_matrix = []
    
        for X_split in X_joint:
            # print np.dot(star_matrix,np.array(X_split).T).T
            # print np.array(X_split).shape
            # temp = softmax(np.dot(star_matrix,np.array(X_split).T).T,axis=0)
            temp = softmax(normalize(np.dot(star_matrix,np.array(X_split).T).T),axis=0)
            psi_matrix.append(temp)
            # print temp
        save_obj(np.array(psi_matrix),"%s/psi_matrix"%(target))
        # np.save("../data/%s/psi_matrix"%(target),np.array(psi_matrix))
    else:
        psi_matrix = []
        X_psi = []
        y_psi = []
        cost_psi = []
        X_index = []
        for (X_split,y_split,cost_split) in zip(X_joint,y_joint,src_cost):
            temp = normalize(np.dot(star_matrix,np.array(X_split).T).T)
            # temp = np.dot(star_matrix,np.array(X_split).T).T
            filtered,index = top_k(temp,k)
            # print softmax(filtered,axis=0),index
            psi_matrix.append(softmax(filtered,axis=0))
            # print filtered,filtered.shape
            X_temp = np.array(X_split)[index]
            X_psi.append(X_temp)
            y_temp = np.array(y_split)[index]
            y_psi.append(y_temp)
            cost_temp = np.array(cost_split)[index]
            cost_psi.append(cost_temp)
            X_index.append(index)
            # print y_temp.shape
        # print top_k(psi_matrix,k)
        # print psi_matrix[0].sum(axis=0).shape,psi_matrix[0].sum(axis=0)
        psi_matrix = np.array(psi_matrix)
        X_psi = np.array(X_psi)
        y_psi = np.array(y_psi)
        cost_psi = np.array(cost_psi)
        save_obj(psi_matrix,"%s/%s/psi_matrix"%(target,k))
        save_obj(X_psi,"%s/%s/X_psi"%(target,k))
        save_obj(y_psi,"%s/%s/y_psi"%(target,k))
        save_obj(cost_psi,"%s/%s/src_cost_psi"%(target,k))
        save_obj(X_index,"%s/%s/X_index"%(target,k))
        # print sum([y for domain in y_psi for y in domain  if y==1])
    return np.array(psi_matrix)

# get top k values and indexes
def top_k(a,k):
    temp,index = torch.topk(torch.from_numpy(a.sum(axis=1)),k,dim=0)
    return a[index.numpy()],index.numpy()

# return an all-in-one matrix
def get_sents(X_joint):
    return np.array([x for domain in X_joint for x in domain])



# psi for test data vs train data
def compute_psi_for_test(X_joint, X_test):
    psi_matrix = []
    for X_split in X_joint:
        # temp = softmax(np.dot(X_test,np.array(X_split).T).T,axis=0)
        temp = softmax(normalize(np.dot(X_test,np.array(X_split).T).T),axis=0)
        psi_matrix.append(temp)
    # print psi_matrix,np.array(psi_matrix).shape
    # return psi_matrix
    return get_sents(np.array(psi_matrix))

# evaluation on baselines
def test_confidence(target,pos_star,neg_star,option=0,clf='lr',theta=0.5,k=2000):
    X_train = []
    y_train = []
    if option == 0: # T_L*
        # pos_star = load_obj('%s/pos_star'%(target))[:k]
        # neg_star = load_obj('%s/neg_star'%(target))[:k]
        # pos_star,neg_star = predict_tops(target,k=k,theta=0.5,sorting="dsc")
        X_train = concatenate(pos_star,neg_star)
        y_train = concatenate(np.ones(len(pos_star)),np.zeros(len(neg_star)))
    elif option == 1: # S_L
        X_joint = load_obj("%s/X_joint"%target)
        y_joint = load_obj("%s/y_joint"%target)
        X_train = get_sents(X_joint)
        y_train = get_sents(y_joint)
    else:
        X_joint = load_obj("%s/X_joint"%target)
        y_joint = load_obj("%s/y_joint"%target)
        X_train1 = get_sents(X_joint)
        y_train1 = get_sents(y_joint)
        # pos_star = load_obj('%s/pos_star'%(target))[:k]
        # neg_star = load_obj('%s/neg_star'%(target))[:k]
        X_train2 = concatenate(pos_star,neg_star)
        y_train2 = concatenate(np.ones(len(pos_star)),np.zeros(len(neg_star)))
        X_train = concatenate(X_train1,X_train2)
        y_train = concatenate(y_train1,y_train2)
    X_test = load_obj("%s/X_test"%target)
    y_test = load_obj("%s/y_test"%target)
    clf_func = get_clf_func(clf)
    clf_func.fit(X_train,y_train)
    pred = clf_func.predict(X_test)
    acc = accuracy_score(y_test,pred) if "large" not in target else f1_score(y_test,pred,average='macro')
    print acc*100
    return acc*100


##############################################################################################
def test_embedding(target,k):
    domains = ["books","dvd","electronics","kitchen"]
    index_set = load_obj("%s/%s/X_index"%(target,k))
    # print index_set
    temp = index_to_source_sentence(index_set,target,domains)
    f = open("../work/example_%s.txt"%target, "w")
    for sentence,source,label in temp:
        f.write('%s,%d\n'%(source,label))
        f.write("%s\n\n"%sentence)
    f.close()
    pass


def generate_un():
    source = "d1"
    domains = ["books","dvd","electronics","kitchen"]
    for target in domains:
        src_un,tgt_un = set_up_unlabelled(source,target,False)
        save_obj(tgt_un,"%s/X_un"%target)
    pass

################################################################################


# find best theta and show the probabilies for best results
def find_theta(target):
    print target
    resFile = open("../work/params/%s_theta.csv"%target,"w")
    resFile.write("theta, acc, method\n")
    thetas = [0.5,0.6,0.7,0.8,0.9]
    best_theta = 0.0
    best_acc = 0.0
    best_pos = ""
    best_neg = ""
    best_method = ""
    for theta in thetas:
        pos_star,neg_star,acc,method,sorting,pos_proba,neg_proba = predict_tops(target,theta=theta)
        # print "S_L:",
        # acc1 = test_confidence(target,option=1,theta=theta)
        print "PL(T_L*):",acc,theta
        # acc = test_confidence(target, pos_star,neg_star,theta=theta)
        # print "S_L+T_L*:",
        # acc3 = test_confidence(target,option=2,theta=theta)
        resFile.write("%f, %f, %s, %s\n"%(theta,acc,method,sorting))
        if best_acc<acc:
            best_acc = acc
            best_theta = theta
            best_pos = pos_star
            best_neg = neg_star
            best_method = method
            best_sorting = sorting
            best_pos_proba = pos_proba
            best_neg_proba = neg_proba
        resFile.flush()
    resFile.close()
    print "####################################"
    print "best_theta:",best_theta,"best_acc:",best_acc, "best_method:",best_method,best_sorting
    save_obj(best_pos,"%s/pos_star"%target)
    save_obj(best_neg,"%s/neg_star"%target)
    save_obj(best_pos_proba,"%s/pos_proba"%target)
    save_obj(best_neg_proba,"%s/neg_proba"%target)
    pass

# find best k and show the probabilties for the best results
def find_k(target,theta=0.5,sorting="asc"):
    print target
    resFile = open("../work/params/%s_k.csv"%target,"w")
    resFile.write("k, LR(S_L), LR(T_L*), LR(S_L+T_L*)\n")
    ks = [100,200,400,600,800,1000]
    # ks = [100,500,1000,2000,3000,4000]
    for k in ks:
        print k
        print "S_L:",
        acc1 = test_confidence(target,option=1,theta=theta)
        print "T_L*:",
        acc2 = test_confidence(target,theta=theta,k=k)
        print "S_L+T_L*:",
        acc3 = test_confidence(target,option=2,theta=theta,k=k)
        resFile.write("%d, %f, %f, %f\n"%(k,acc1,acc2,acc3))
        resFile.flush()
    resFile.close()
    pass

##################################################################################

# ascending and descending ordered pseudo labelled instances 
# hypothesis: which one is better for target / source?
def hypothesis(target,compare="S_L"):
    print target,compare
    A = predict_tops(target,k=100,sorting="asc")
    D = predict_tops(target,k=100,sorting="dsc")
    print "A",compare,
    sim1 = set_target_sim(A,target,compare=compare)
    print sim1
    print "D",compare,
    sim2 = set_target_sim(D,target,compare=compare)
    print sim2
    print "sim(A,%s)>sim(D,%s)"%(compare,compare),sim1>sim2
    pass

# calculate similarity between centroid of T_U and save to tgt_sim
def unlabel_sim(target):
    tgt_un = load_obj("%s/X_un"%target)
    # print target,tgt_un.shape
    c_t = compute_centriod(tgt_un)
    computed_tgt_sim = [cos_sim(x,c_t) for x in tgt_un]
    save_obj(computed_tgt_sim,"%s/tgt_sim"%target)
    pass

# calculate similarity between src_train and centroid of T_U and save to src_cost
def src_cost(target):
    X_joint = load_obj("%s/X_joint"%target)
    src_train = get_sents(X_joint)
    tgt_un = load_obj("%s/X_un"%target)
    c_t = compute_centriod(tgt_un)
    # print src_train
    sim = [cos_sim(x,c_t) for x in src_train]
    # s = sum(sim)
    # sim = [x/s for x in sim]
    # print normalized_sim
    sim = list(split_list(sim,3))
    save_obj(sim,"%s/src_cost"%target)
    pass

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

# take a list of embeddings and return a centroid
def compute_centriod(instances):
    a = np.array(instances)
    # a = instances
    # print np.mean(a,axis=0).shape
    return np.mean(a,axis=0)

def cos_sim(a,b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) if (np.linalg.norm(a)*np.linalg.norm(b))!=0 else 0
    return cos_sim

###############################################################################################
if __name__ == '__main__':
    if len(sys.argv) >1:
        target = sys.argv[1]
        print target
        # main(target) # multiple source tri-training
        # main(target,loaded=True) # pre-loaded before
        # test(target)
        # hypothesis(target,compare="T_U")
        
        # find_k(target,sorting="dsc") # sort by similarity to target
        
        # label_sim(target)
        
        # predict_unbalanced_tops(target,percent=0.15,sorting="dsc")
        # compute_psi(target)
        # generate_un()

        unlabel_sim(target)
        src_cost(target)


        find_theta(target)
        ks = [None,5,10,30,50,70,100,200]
        for k in ks:
            compute_psi(target,k)
            
        # multi_source_loader(target,tgt_un_ON=False) # store X_joint and y_joint
    else:
        print "usage: <target>"

