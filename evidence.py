"""
evidence getter


"""
import torch
import torch.nn as nn
from torch import optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from load_data import load_obj,prepare_test,index_to_source_sentence,save_obj
import os,sys,glob
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_single_evidence(target,test_index,tops=20):
    domains = ["books","dvd","electronics","kitchen"]
    # sentence (text), domain, label, psi,theta, psi_theta
    evidence = [] # [0] : test instance, [1:] : contributed source instances
    x = load_obj("%s/X_test"%target)[test_index]
    test_sent = prepare_test(target,label=True,sent=True)[test_index]

    filename = "../work/%s/*/*_*.model"%(target)
    model_paths = glob.glob(filename)
    model_path = "" # best_path
    best_acc = 0.0
    best_k = 0.0
    for path in model_paths:
        k = int(os.path.basename(os.path.dirname(path)))
        acc = float(os.path.basename(path).replace('.model','').split('_')[1])/100.0 if path is not None else 0.0
        if best_acc < acc:
            best_acc = acc
            best_k = k
            model_path = path
    print model_path
    model = torch.load(model_path,map_location=device)

    k = best_k if "mlp" not in target else 0
    print k
    example = open("../work/examples/test_%s_%s_k_%s.txt"%(target,test_index,k),'w')
    
    print "###test instance###"
    print target
    org_label = test_sent[1]
    org_sent = ' '.join(test_sent[0])
    print org_sent,org_label

    evidence.append([org_sent,target,org_label])

    if k == None or k==1600:
        X_joint = load_obj("%s/X_joint"%target)
        y_joint = load_obj("%s/y_joint"%target)
    else:
        X_joint = load_obj("%s/%s/X_psi"%(target,k))
        y_joint = load_obj("%s/%s/y_psi"%(target,k))
        X_index = load_obj("%s/%s/X_index"%(target,k))

    # compute psi_matrix for test_data
    psi_matrix = []
    for X_split in X_joint:
        temp = softmax(np.dot(x,X_split.T).T)
        psi_matrix.append(temp)
    psi_matrix = label_to_tensor(get_all(psi_matrix)).to(device)
    psi_splits = torch.chunk(psi_matrix,len(domains)-1,dim=0)
    # psi_matrix = psi_matrix.view(-1,len(psi_matrix)).to(device)
    # psi_splits = torch.chunk(psi_matrix,len(domains),dim=1)

    x = sent_to_tensor(x)
    y = label_to_tensor(get_all(y_joint))
    y_splits = torch.chunk(y,len(domains)-1,dim=0)

    theta_splits = []
    sum_src = 0.0
    for i in range(len(domains)-1):
        phi_src = model['phi_srcs.%s'%i].to(device)
        temp = torch.exp(torch.mm(x,phi_src))
        theta_splits.append(temp)
        sum_src+=temp
    # print theta_splits
    
    sum_matrix = 0
    sum_matrix2 = 0
    psi_values = 0
    theta_values = 0
    count = 0
    for psi_split,theta_split,y_split in zip(psi_splits,theta_splits,y_splits):
        # phi_src = model['phi_srcs.%s'%i].to(device)
        theta_split = theta_split/sum_src
        temp = psi_split*theta_split
        # print psi_split.size(),theta_split.size()
        temp2 = y_split*psi_split*theta_split
        # print temp.size(),temp2.size()
        # print torch.sum(y_split*psi_split),torch.sum(temp2)
        theta_temp = theta_split.expand_as(psi_split)
        if count == 0:
            sum_matrix = temp
            psi_values = psi_split
            theta_values = theta_temp
            # print theta_values.size()
            count+=1
        else:
            sum_matrix = torch.cat((sum_matrix,temp),dim=0)
            psi_values = torch.cat((psi_values,psi_split),dim=0)
            theta_values = torch.cat((theta_values,theta_temp),dim=0)
        sum_matrix2+=torch.sum(temp2)

    # print sum_matrix2
    sum_matrix2 = sum_matrix2  + model['bias'].to(device)
    sigmoid = torch.nn.Sigmoid()
    y_hat = sigmoid(sum_matrix2)
    # print y_hat
    predicted_label = np.round(y_hat.data.item())
    print predicted_label,org_label==predicted_label


    example.write("###test instance###\n%s %s %s %s\n%s\n\n"%(target,org_label,
        predicted_label,org_label==predicted_label,org_sent))
    example.write("########################\n")

    # print sum_matrix.size(),sum_matrix.sum()
    tops = tops if k*3>tops else k*3
    temp,index = torch.topk(sum_matrix,tops,dim=0)
    # print temp.size()
    psi_list = psi_values.data[index]
    theta_list = theta_values.data[index]
    # print psi_list.size(),theta_list.size()

    for value,sent_index,psi,theta in zip(temp.data,index.data,psi_list,theta_list):
        source,label,sent = 0,0,0
        if k == 0 or k==1600:
            source,label,sent = index_to_source_sentence(sent_index,target,domains)
        else:
            source,label,sent = index_to_source_sentence(sent_index,target,domains,X_index)
        if label == org_label:
            example.write('+')
        else:
            example.write('-')
        evidence.append([sent,source,label,psi.data.item(),theta.data.item(),value.data.item()])
        example.write("%s %s %s\n"%(source,label,value.data.item()))
        example.write("%s\n\n"%sent)
        example.flush()
    example.close()
    return evidence

# draw chart for evidence
def contruct_collection(evidence,option='theta'):
    orginal = evidence[0]
    evidence_sents = evidence[1:]
    # sentence (text), domain, label, psi,theta, psi_theta
    domains = []
    theta_values = []
    psi_values = []
    temp = []
    if option == "theta":
        for evd in evidence_sents:
            if evd[1] not in domains:
                domains.append(evd[1])
                theta_values.append(evd[4])
        collected = domains,theta_values
        print collected
    else:
        sents,domains,labels,psis,thetas,temps = zip(*evidence_sents)
        collected = domains,labels,psis,temps
    return collected

############################################################

# return an all-in-one matrix
def get_all(X_joint):
    return np.array([x for domain in X_joint for x in domain])

# this transposes the originial vector (a,b) to tensor (b,a) 
def sent_to_tensor(v):
    return Variable(torch.FloatTensor(v).view(-1,len(v)).to(device))

# will keep the original shape of vector
def to_tensor(v):
    return Variable(torch.FloatTensor(v).to(device))

def label_to_tensor(v):
    # print len(v)
    return Variable(torch.FloatTensor(v).view(len(v),-1).to(device))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


if __name__ == '__main__':
    if len(sys.argv) >2:
        target = sys.argv[1]
        test_index = int(sys.argv[2])
        # option = sys.argv[3]
        evd = compute_single_evidence(target,test_index,tops=2000)
        collected = contruct_collection(evd,option='all')
        save_obj(collected,'collected-all')
        collected = contruct_collection(evd)
        save_obj(collected,'collected-theta')
    # elif len(sys.argv) >2:
    #     target = sys.argv[1]
    #     # k = int(sys.argv[2])
    #     # tops = int(sys.argv[3])
    #     test_index = int(sys.argv[2])
    #     evd = compute_single_evidence(target,test_index,tops=2000)
        
    else:
        print "<target,test_index>"
