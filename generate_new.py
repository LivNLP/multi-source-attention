"""
Brand-new self-training confidence threshold way
combining target similarity in the source domain (between S_U and T_U)
and probabilities in source domain

 Xia Cui
 04/2019
"""
import numpy as np
from load_data import load_obj,save_obj

def compute_centriod(instances):
    a = np.array(instances)
    return np.mean(a,axis=0)

def compute_sim(train_data,target):
    # src_un = load_obj("%s/X_un"%source)
    tgt_un = load_obj("%s/X_un"%target)
    c_t = compute_centriod(tgt_un)
    sim = [cos_sim(x,c_t) for x in train_data]
    return sim

def cos_sim(a,b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) if (np.linalg.norm(a)*np.linalg.norm(b))!=0 else 0
    return cos_sim

def sim_features(input_data,labels_proba,classes,target,theta,sim_theta=0):
    sim = compute_sim(input_data,target)
    print max(sim)
    pos = [x for x,y,z in zip(input_data,labels_proba,sim) if max(y)>theta and classes[list(y).index(max(y))]==1 and z >sim_theta*max(sim)]
    neg = [x for x,y,z in zip(input_data,labels_proba,sim) if max(y)>theta and classes[list(y).index(max(y))]==0 and z >sim_theta*max(sim)]
    # pos = [x for x,y in zip(pos)]
    return pos,neg
