"""
Examine the pesudo-labelled datasets
1. only sort by prob
2. only sort by sim
3. sort by prob, then sort by sim
4. sort by sim, then sort by prob
5. prob + sim
6. prob x sim

dsc and asc and return the best one.


"""
from exam_stars import test
import numpy as np

def find_best_method(target,tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sorting="dsc"):
    best_acc = 0.0
    best_method = ""

    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba = just_prob(tgt_un,labels_proba,tgt_sim,k,theta,sorting)
    acc = test(target,pos_star,neg_star)
    if acc>best_acc:
        best_acc = acc
        best_method = "just_prob"
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba = just_sim(tgt_un,labels_proba,tgt_sim,k,theta,sorting)
    acc = test(target,pos_star,neg_star)
    if acc>best_acc:
        best_acc = acc
        best_method = "just_sim"
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end

    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba = prob_sim(tgt_un,labels_proba,tgt_sim,k,theta,sorting)
    acc = test(target,pos_star,neg_star)
    if acc>best_acc:
        best_acc = acc
        best_method = "prob_sim"
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba = sim_prob(tgt_un,labels_proba,tgt_sim,k,theta,sorting)
    acc = test(target,pos_star,neg_star)
    if acc>best_acc:
        best_acc = acc
        best_method = "sim_prob"
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba = add_prob_sim(tgt_un,labels_proba,tgt_sim,k,theta,sorting)
    acc = test(target,pos_star,neg_star)
    if acc>best_acc:
        best_acc = acc
        best_method = "prob+sim"
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

    pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba = multi_prob_sim(tgt_un,labels_proba,tgt_sim,k,theta,sorting)
    acc = test(target,pos_star,neg_star)
    if acc>best_acc:
        best_acc = acc
        best_method = "prob*sim"
        best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,\
            best_pos_proba,best_neg_proba = pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba
    return best_pos_star,best_neg_star,best_pos_start,best_pos_end,best_neg_start,best_neg_end,best_acc,best_method,best_pos_proba,best_neg_proba

# just probs
def just_prob(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    # tgt_sim is not used!
    pos_list = [(x,y[1],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star,pos_proba,neg_proba = starize(pos_list,neg_list,k/2,index=1,sort_prob=sort_prob)
    if len(neg_star) > 0 and len(neg_star) >0:
        pos_start,pos_end = pos_list[0][1],pos_list[len(pos_star)-1][1]
        neg_start,neg_end = neg_list[0][1],neg_list[len(neg_star)-1][1]
    else:
        pos_start,pos_end = -1,-1
        neg_start,neg_end = -1,-1
    return pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba


# just labels
def just_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    # labels_proba is not used
    pos_list = [(x,y[1],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=2)
    neg_list = sort_by_index(neg_list,index=2)
    pos_star,neg_star,pos_proba,neg_proba = starize(pos_list,neg_list,k/2,index=2)
    if len(neg_star) > 0 and len(pos_star) >0:
        pos_start,pos_end = pos_list[0][1],pos_list[len(pos_star)-1][1]
        neg_start,neg_end = neg_list[0][1],neg_list[len(neg_star)-1][1]
    else:
        pos_start,pos_end = -1,-1
        neg_start,neg_end = -1,-1
    return pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

def prob_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    pos_list = [(x,y[1],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0],z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    num = min(min(k,len(pos_list)),len(neg_list))
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)[:num]
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)[:num]
    pos_list = sort_by_index(pos_list,index=2)
    neg_list = sort_by_index(neg_list,index=2)
    pos_star,neg_star,pos_proba,neg_proba = starize(pos_list,neg_list,k/2,index=2,sort_prob=sort_prob)
    if len(neg_star) > 0 and len(pos_star) >0:
        pos_start,pos_end = pos_list[0][1],pos_list[len(pos_star)-1][1]
        neg_start,neg_end = neg_list[0][1],neg_list[len(neg_star)-1][1]
    else:
        pos_start,pos_end = -1,-1
        neg_start,neg_end = -1,-1
    return pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

# tgt_sim first
def sim_prob(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    sum_dict = [(x,y,z) for x,y,z in zip(tgt_un,labels_proba,tgt_sim)]
    num = min(k*2,len(sum_dict))
    sum_dict = sort_by_index(sum_dict,index=2)[:num]
    pos_list = [(x,y[1],z) for x,y,z in sum_dict if y[1] >theta]
    neg_list = [(x,y[0],z) for x,y,z in sum_dict if y[0] >theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star,pos_proba,neg_proba = starize(pos_list,neg_list,k/2,index=1,sort_prob=sort_prob)
    if len(neg_star) > 0 and len(pos_star) >0:
        pos_start,pos_end = pos_list[0][1],pos_list[len(pos_star)-1][1]
        neg_start,neg_end = neg_list[0][1],neg_list[len(neg_star)-1][1]
    else:
        pos_start,pos_end = -1,-1
        neg_start,neg_end = -1,-1
    return pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba

# prob+sim
def add_prob_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    pos_list = [(x,y[1]+z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,y[0]+z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star,pos_proba,neg_proba = starize(pos_list,neg_list,k/2,index=1,sort_prob=sort_prob)
    if len(neg_star) > 0 and len(pos_star) >0:
        pos_start,pos_end = pos_list[0][1],pos_list[len(pos_star)-1][1]
        neg_start,neg_end = neg_list[0][1],neg_list[len(neg_star)-1][1]
    else:
        pos_start,pos_end = -1,-1
        neg_start,neg_end = -1,-1
    return pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba


# prob*sim
def multi_prob_sim(tgt_un,labels_proba,tgt_sim,k=100,theta=0.5,sort_prob="dsc"):
    pos_list = [(x,(y[1])*z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[1]>theta]
    neg_list = [(x,(y[0])*z,0) for x,y,z in zip(tgt_un,labels_proba,tgt_sim) if y[0]>theta]
    pos_list = sort_by_index(pos_list,index=1,sorting=sort_prob)
    neg_list = sort_by_index(neg_list,index=1,sorting=sort_prob)
    pos_star,neg_star,pos_proba,neg_proba = starize(pos_list,neg_list,k/2,index=1,sort_prob=sort_prob)
    if len(neg_star) > 0 and len(pos_star) >0:
        pos_start,pos_end = pos_list[0][1],pos_list[len(pos_star)-1][1]
        neg_start,neg_end = neg_list[0][1],neg_list[len(neg_star)-1][1]
    else:
        pos_start,pos_end = -1,-1
        neg_start,neg_end = -1,-1
    return pos_star,neg_star,pos_start,pos_end,neg_start,neg_end,pos_proba,neg_proba


# 1 = probs
# 2 = sim
def sort_by_index(list_to_sort,index=1,sorting='dsc'):
    if sorting == "dsc": 
        list_to_sort.sort(lambda x,y: -1 if x[index]>y[index] else 1)
    else: #asc
        list_to_sort.sort(lambda x,y: -1 if x[index]<y[index] else 1)
    return list_to_sort

# return pseudo labelled positive and negative datasets
# also give the corresponding prediction confidence
def starize(pos_list,neg_list,k,index,sort_prob="dsc"):
    temp = min(len(pos_list),len(neg_list))
    if k > temp and temp != 0:
        k = temp
    if temp == 0:
        print "empty!"
        return [],[],[],[]
    pos_star = [x for x,y,z in pos_list[:k]]
    neg_star = [x for x,y,z in neg_list[:k]]
    pos_proba,neg_proba = compute_normalized_weights(pos_list[:k],neg_list[:k],index,sort_prob)
    # print pos_proba
    return pos_star,neg_star,pos_proba,neg_proba

def compute_normalized_weights(pos_list,neg_list,index,sort_prob="dsc"):
    pos_proba = [l[index] for l in pos_list]
    neg_proba = [l[index] for l in neg_list]
    if sort_prob == "dsc":
        return normalize(pos_proba),normalize(neg_proba)
    else:
        return normalize([1.0-x for x in pos_proba]),normalize([1.0-x for x in neg_proba])

def normalize(a):
    return (1.0/sum(a))*np.array(a)
