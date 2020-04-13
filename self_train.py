"""
Self-training for multi-source domain

"""
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils import resample
from load_data import save_obj,load_obj,get_clf_func,set_up_labelled
from mlxtend.classifier import EnsembleVoteClassifier
import sys
import itertools
import numpy as np
from generate_new import sim_features

# Approach 1: separate models on each source domain labelled data
# using source unlabelled data to increase data size
def self_train(domain,X_train,y_train,X_test,y_test,X_un,theta=0.5):
    total = len(X_train)+len(X_un)
    # print np.array(X_un).shape
    t = 0
    count = 0
    unlabelled_features = np.array(X_un)
    acc,clf_func = get_acc_clf(domain,X_train,y_train,X_test,y_test)
    best_clf = clf_func
    best_acc = acc
    print 'initial acc:',best_acc
    while count < total:    
        t += 1
        X_new,y_new = predict_features(X_train,y_train,unlabelled_features,domain,theta)
        # print np.array(X_new).shape
        if len(X_new)==0 and t>1:
            print "no new features added"
            break
        X_train = concatenate(X_train,X_new)
        y_train = concatenate(y_train,y_new)
        count = len(X_train)
        acc,clf_func = get_acc_clf(domain,X_train,y_train,X_test,y_test)
        if best_acc<acc:
            best_acc = acc
            best_clf = clf_func
            print "*NEW BEST! best acc:", best_acc
        else:
            print "no improvement..skip.."
            break
        if count == total:
            print "reach end.."
            break
        # print np.array(X_new).shape
        X_new = [list(x) for x in X_new]
        unlabelled_features =[x for x in unlabelled_features if list(x) not in list(X_new)]
    
    print "best for %s:"%theta,best_acc
    return best_acc,best_clf

def find_best_self(domain):
    X_train = load_obj("%s/X_train"%domain)
    y_train = load_obj("%s/y_train"%domain)
    X_test = load_obj("%s/X_test"%domain)
    y_test = load_obj("%s/y_test"%domain)
    X_un = load_obj("%s/X_un"%domain)

    thetas = [0.5,0.6,0.7,0.8,0.9]
    best_acc = 0.0
    best_clf =""
    best_theta = 0.0
    resFile = open("../work/params/%s_in_theta.csv"%domain,"w")
    resFile.write("theta, acc\n")
    for theta in thetas:
        print "##############################"
        print "start with theta=%s"%theta
        print "##############################"
        acc,clf_func = self_train(domain,X_train,y_train,X_test,y_test,X_un,theta=theta)
        
        if best_acc<acc:
            best_acc = acc
            best_clf = clf_func
            best_theta = theta

        resFile.write("%f, %f\n"%(theta,acc))
        resFile.flush()
    resFile.close()
    print "##############################"
    print "best_theta:",best_theta,"best_acc:",best_acc
    save_obj(best_clf,"%s/self_clf"%domain)
    pass

##############################################################################################
def majority_vote(target):
    X_test = load_obj("%s/X_test"%target)
    y_test = load_obj("%s/y_test"%target)

    domains = []
    if "mlp" in target:
        domains = ["mlp/books","mlp/dvd","mlp/electronics","mlp/kitchen"]
    else:
        if "large" not in target:
            domains = ["books","dvd","electronics","kitchen"]
            if target not in domains:
                return
        else:
            domains =["large/baby","large/cell_phone","large/imdb","large/yelp2014"]

    models = []
    for source in domains:
        if target == source:
            continue
        else:
            print source
            clf_func = load_obj("%s/self_clf"%source)
            models.append(clf_func)


    eclf = EnsembleVoteClassifier(clfs=models,refit=False)#weights=[1,1,1],
    eclf.fit(X_test,y_test) # this line is not doing work
    tmp_name = target.upper()[0] if "large" not in target else "large/"+target.upper()[6]
    tmp_name = target.upper()[0] if "mlp" not in target else "mlp/"+target.upper()[4]
    save_obj(eclf, '%s_eclf'%(tmp_name))
    pred = eclf.predict(X_test)
    acc = accuracy_score(y_test,pred) if "large" not in target else f1_score(y_test,pred,average='macro')
    print 'self-train',acc
    pass

def majority_vote_mlp(target):
    X_test = load_obj("%s/X_test"%target)
    y_test = load_obj("%s/y_test"%target)

    # domains = ["mlp/books","mlp/dvd","mlp/electronics","mlp/kitchen"]
    data_name = ["books", "dvd", "electronics", "kitchen"]
    X_joint = load_obj("%s/X_joint"%target)
    y_joint = load_obj("%s/y_joint"%target)
    temp_un = load_obj("%s/X_un"%target)
    meta_sources = []
    for i in range(len(data_name)):
        if 'mlp/'+data_name[i] != target:
            meta_sources.append(data_name[i])
    # print meta_sources
    models = []
    for j in range(len(meta_sources)):
        temp_X = X_joint[j]
        temp_y = y_joint[j]
        thetas = [0.5,0.6,0.7,0.8,0.9]
        best_acc = 0.0
        best_clf =""
        best_theta = 0.0
        resFile = open("../work/params/%s_theta_self-%s.csv"%(target,meta_sources[j].upper()[0]),"w")
        resFile.write("theta, acc\n")
        for theta in thetas:
            print "##############################"
            print "start with theta=%s"%theta
            print "##############################"
            acc,clf_func = self_train(target,temp_X,temp_y,X_test,y_test,temp_un,theta=theta)
            
            if best_acc<acc:
                best_acc = acc
                best_clf = clf_func
                best_theta = theta

            resFile.write("%f, %f\n"%(theta,acc))
            resFile.flush()
        resFile.close()
        print "##############################"
        print "best_theta:",best_theta,"best_acc:",best_acc
        models.append(best_clf)

    eclf = EnsembleVoteClassifier(clfs=models,refit=False)#weights=[1,1,1],
    eclf.fit(X_test,y_test) # this line is not doing work
    # tmp_name = target.upper()[0] if "large" not in target else "large/"+target.upper()[6]
    # tmp_name = 'mlp/'+target.upper()[4]
    save_obj(eclf, "%s/self_clf"%target)
    pred = eclf.predict(X_test)
    # print pred
    acc = accuracy_score(y_test,pred)
    print 'self-train',acc
    pass

##############################################################################################
# Joint Tri-Training
def tri_train(domain,X_train,y_train,X_test,y_test,X_un,theta=0.5,dis=False):
    models = list()
    accs = list()
    for i in range(3):   
        X_split,y_split = bootstrap_sample(X_train,y_train)
        acc,clf_func = get_acc_clf(domain,X_split,y_split,X_test,y_test)
        models.append(clf_func)
        accs.append(acc)

    for (j,k) in itertools.combinations(models,2):
        # i_features = list()
        unlabelled_features = np.array(X_un)
        total = len(X_train)+len(X_un)
        t = 0
        count = 0
        X_i = X_train
        y_i = y_train
        # find current classifier
        clf_i = [x for x in models if x!=j and x!=k][0]
        index_i = models.index(clf_i)
        print "***classifier %d***"%index_i
        while count < total and len(unlabelled_features)!=0:
            t += 1            
            X_tgt,y_tgt = get_features(unlabelled_features,j,k,clf_i,models,theta=theta,dis=dis)
            if len(X_tgt)==0 and t>1:
                print "no new features added"
                break
            
            X_i = concatenate(X_i,X_tgt)
            y_i = concatenate(y_i,y_tgt)
            count = len(X_i)
            print "%d %d %d"%(t,count,total)
            # clf_i.fit(X_i,y_i)
            # update classifier
            acc,clf_i = get_acc_clf(domain,X_i,y_i,X_test,y_test)
            if accs[index_i]<acc:
                accs[index_i] = acc
                # best_clf = clf_i
                print "*NEW BEST! best acc:", acc
                models[index_i] = clf_i
            else:
                print "no improvement..skip.."
                break
            if count == total:
                print "reach end.."
                break
            # update the unlabelled features for speed-up
            print np.array(X_tgt).shape
            X_tgt = [list(x) for x in X_tgt]
            unlabelled_features =[x for x in unlabelled_features if list(x) not in X_tgt]
            print np.array(unlabelled_features).shape
    # majority vote classifiers
    eclf = EnsembleVoteClassifier(clfs=models,weights=[1,1,1],refit=False)
    eclf.fit(X_test,y_test) # this line is not doing work
    # tmp_name = domain.upper()[0] if "large" not in domain else "large/"+domain.upper()[6]
    pred = eclf.predict(X_test)
    acc = accuracy_score(y_test,pred) if "large" not in domain else f1_score(y_test,pred,average='macro')

    print "acc:%s theta:%s"%(acc,theta),"seprate accs:",accs
    return acc,eclf

def bootstrap_sample(X_train,y_train,n_splits=3):
    # generate boostrap samples and then restore the format (split tuples into two lists)
    return zip(*resample(zip(X_train,y_train),n_samples = X_train.shape[0]/n_splits))

def get_features(tgt_un,j,k,i,models,theta,dis=False):
    # print np.array(tgt_un).shape 
    tgt_un = np.array(tgt_un)
    j_labels = predict_labels(j,tgt_un,theta)
    k_labels = predict_labels(k,tgt_un,theta)
    true_labels = np.equal(j_labels,k_labels)
    pred = true_labels
    if dis == True:
        false_labels = np.not_equal(predict_labels(i,tgt_un,theta),k_labels)
        pred = np.all([true_labels,false_labels],axis=0)
    # drop features not reach the confidence level
    temp = [(x,y) for x,y,z in zip(tgt_un,j_labels,pred) if z==True and y!=-1]
    # avoid emtpy list for unpacking
    if len(temp)==0:
        print "empty"
        return list(),list()
    else:
        X_tgt,y_tgt = zip(*temp)
        X_tgt = [list(x) for x in X_tgt]
        y_tgt = list(y_tgt)
        return X_tgt,y_tgt
    pass

def predict_labels(clf,input_data,theta):
    labels_proba = clf.predict_proba(input_data)
    classes = clf.classes_
    labels = [classes[list(y).index(max(y))] if max(y)>theta else -1 for y in labels_proba]
    # print labels
    return labels

def find_best_tri(target,dis=False):
    X_test = load_obj("%s/X_test"%target)
    y_test = load_obj("%s/y_test"%target)

    domains = []
    temp_X = []
    temp_y = []
    temp_un = []
    if "mlp" in target:
        domains = ["mlp/books","mlp/dvd","mlp/electronics","mlp/kitchen"]
        temp_X = get_sents(load_obj("%s/X_joint"%target))
        temp_y = get_sents(load_obj("%s/y_joint"%target))
        for source in domains:
            if target == source:
                continue
            else:
                X_un = load_obj("%s/X_un"%source)
                temp_un = concatenate(temp_un,X_un)
    else:
        if "large" not in target:
            domains = ["books","dvd","electronics","kitchen"]
            if target not in domains:
                return
        else:
            domains =["large/baby","large/cell_phone","large/imdb","large/yelp2014"]
    
        for source in domains:
            if target == source:
                continue
            else:
                print source
                X_train = load_obj("%s/X_train"%source)
                y_train = load_obj("%s/y_train"%source)
                X_un = load_obj("%s/X_un"%source)
                temp_X = concatenate(temp_X,X_train)
                temp_y = concatenate(temp_y,y_train)
                temp_un = concatenate(temp_un,X_un)

    thetas = [0.5,0.6,0.7,0.8,0.9]
    best_acc = 0.0
    best_clf =""
    best_theta = 0.0
    method = "tri" if dis==False else "tri-d"
    resFile = open("../work/params/%s_theta_%s.csv"%(domain,method),"w")
    resFile.write("theta, acc\n")
    for theta in thetas:
        print "##############################"
        print "start with theta=%s"%theta
        print "##############################"
        acc,clf_func = tri_train(target,temp_X,temp_y,X_test,y_test,temp_un,theta=theta,dis=dis)
        
        if best_acc<acc:
            best_acc = acc
            best_clf = clf_func
            best_theta = theta

        resFile.write("%f, %f\n"%(theta,acc))
        resFile.flush()
    resFile.close()
    print "##############################"
    print "best_theta:",best_theta,"best_acc:",best_acc
    save_obj(best_clf,"%s/%s_clf"%(target,method))
    return acc,clf_func

##############################################################################################
def joint_train(target):
    X_test = load_obj("%s/X_test"%target)
    y_test = load_obj("%s/y_test"%target)

    domains = []
    temp_X = []
    temp_y = []
    temp_un = []
    if "mlp" in target:
        domains = ["mlp/books","mlp/dvd","mlp/electronics","mlp/kitchen"]
        temp_X = get_sents(load_obj("%s/X_joint"%target))
        temp_y = get_sents(load_obj("%s/y_joint"%target))
        for source in domains:
            if target == source:
                continue
            else:
                X_un = load_obj("%s/X_un"%source)
                temp_un = concatenate(temp_un,X_un)
    else:
        if "large" not in target:
            domains = ["books","dvd","electronics","kitchen"]
            if target not in domains:
                return
        else:
            domains =["large/baby","large/cell_phone","large/imdb","large/yelp2014"]
    
        for source in domains:
            if target == source:
                continue
            else:
                print source
                X_train = load_obj("%s/X_train"%source)
                y_train = load_obj("%s/y_train"%source)
                X_un = load_obj("%s/X_un"%source)
                temp_X = concatenate(temp_X,X_train)
                temp_y = concatenate(temp_y,y_train)
                temp_un = concatenate(temp_un,X_un)

    thetas = [0.5,0.6,0.7,0.8,0.9]
    best_acc = 0.0
    best_clf =""
    best_theta = 0.0
    resFile = open("../work/params/%s_theta_joint.csv"%domain,"w")
    resFile.write("theta, acc\n")
    for theta in thetas:
        print "##############################"
        print "start with theta=%s"%theta
        print "##############################"
        acc,clf_func = self_train(target,temp_X,temp_y,X_test,y_test,temp_un,theta=theta)
        
        if best_acc<acc:
            best_acc = acc
            best_clf = clf_func
            best_theta = theta

        resFile.write("%f, %f\n"%(theta,acc))
        resFile.flush()
    resFile.close()
    print "##############################"
    print "best_theta:",best_theta,"best_acc:",best_acc
    save_obj(best_clf,"%s/joint_clf"%target)
    return acc,clf_func

##############################################################################################
def predict_features(X_train,y_train,input_data,target,theta=0.5,clf='lr'):
    clf_func = get_clf_func(clf)
    clf_func.fit(X_train,y_train)
    input_data = np.array(input_data)
    # print np.array(input_data).shape
    labels_proba = clf_func.predict_proba(input_data)
    # print labels_proba
    classes = clf_func.classes_

    pos,neg = sim_features(input_data,labels_proba,classes,target,theta)
    print "%f original: %d, after throttling: %d"%(theta, len(input_data),len(pos)+len(neg))
    print "new_pos: %d, new_neg: %d" %(len(pos),len(neg))
    # print np.array(pos).shape,np.array(neg).shape
    X_new = concatenate(pos,neg)
    y_new = concatenate(np.ones(len(pos)),np.zeros(len(neg)))
    return X_new,y_new

# return an all-in-one matrix
def get_sents(X_joint):
    return np.array([x for domain in X_joint for x in domain])

def concatenate(a,b):
    if len(a)>0 and len(b)>0:
        return np.concatenate((a,b),axis=0)
    elif len(a)==0 and len(b)!=0:
        # print "a empty!!"
        return np.array(b)
    elif len(b)==0 and len(a)!=0: 
        # print "b empty!! length a = %d"%len(a)
        return np.array(a)
    else:
        return list()
    pass

def get_acc_clf(domain,X_train,y_train,X_test,y_test,clf='lr'):
    clf_func = get_clf_func(clf)
    clf_func.fit(X_train,y_train)
    pred = clf_func.predict(X_test)
    acc = accuracy_score(y_test,pred) if "large" not in domain else f1_score(y_test,pred,average='macro')
    return acc,clf_func

def switch_method(domain,method="self"):
    if method == "self":
        if "mlp" in domain:
            majority_vote_mlp(domain)
        else:
            majority_vote(domain)
    elif method == "joint":
        joint_train(domain)
    elif method == "tri":
        find_best_tri(domain)
    elif method == "tri-d":
        find_best_tri(domain,dis=True)
    else:
        print "no valid method"
    pass


################################################################################

if __name__ == '__main__':
    if len(sys.argv) >2:
        domain = sys.argv[1]
        method = sys.argv[2]
        print domain,method
        # find_best_self(domain)
        # majority_vote(domain)
        # joint_train(domain)
        # majority_vote_mlp(domain)
        # find_best_tri(domain)#,dis=True
        switch_method(domain,method)
    else:
        print "usage: <domain,method>"
