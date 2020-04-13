"""
Attention-based Model 



"""
import torch
import torch.nn as nn
from torch import optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from load_data import prepare_source,prepare_evaluate,prepare_data,load_obj,save_obj
import os,sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from domain_similarity import compute_psi_for_test
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DomainAttention(nn.Module):
    # embedding_dim: embedding dimensionality
    # hidden_dim: hidden layer dimensionality
    # source_size: number of source domains
    # hidden_dim, batch_size, label_size, ,num_instances=1600
    def __init__(self, embedding_dim, source_size, y,label_size=2):
        super(DomainAttention, self).__init__()
        # bias
        self.bias = nn.Parameter(torch.Tensor([0]).to(device))
        # source phi, multi sources, just like python list
        self.phi_srcs = nn.ParameterList([init_w(embedding_dim) for i in range(source_size)])
        # labels for source domain (y(d_j))
        self.y = y
        self.label_size = label_size
        self.sigmoid = nn.Sigmoid()
        self.source_size = source_size


    def forward(self, x, psi_matrix):
        x = x.view(-1,len(x)).to(device)
        y = self.y.view(-1,len(self.y)).to(device)
        psi_matrix = psi_matrix.view(-1,len(psi_matrix)).to(device)
        psi_splits = torch.chunk(psi_matrix,self.source_size,dim=1)
        y_splits = torch.chunk(y,self.source_size,dim=1)

        # get the sum of x * phi_src
        theta_splits = []
        sum_src = 0.0
        for phi_src in self.phi_srcs:
            temp = torch.exp(torch.mm(x,phi_src))
            
            # temp = torch.tensor([[0.]]).to(device) if torch.isnan(temp)==True else temp
            # prod = torch.mm(x,phi_src)
            # temp = torch.exp(prod)
            # temp = torch.mm(x,phi_src)
            theta_splits.append(temp)
            sum_src+=temp
            # print temp,torch.sum(x),torch.sum(phi_src)

        sum_matrix = 0.0
        count = 0
        for theta,psi_split,y_split in zip(theta_splits,psi_splits,y_splits):
            count += 1
            theta_matrix = theta/sum_src
            temp = y_split*psi_split*theta_matrix
            sum_matrix += torch.sum(temp)

        sum_matrix = sum_matrix  + self.bias

        y_hat = self.sigmoid(sum_matrix)

        return y_hat

def init_w(embedding_dim):
    w = torch.Tensor(embedding_dim,1)
    # w= torch.Tensor(embedding_dim,embedding_dim)
    return nn.Parameter(nn.init.xavier_uniform_(w).to(device)) # sigmoid gain=1

# this transposes the originial vector (a,b) to tensor (b,a) 
def sent_to_tensor(v):
    return Variable(torch.FloatTensor(v).transpose(0,1).to(device))

def label_to_tensor(v):
    # print len(v)
    return Variable(torch.FloatTensor(v).view(len(v),-1).to(device))

# will keep the original shape of vector
def to_tensor(v):
    return Variable(torch.FloatTensor(v).to(device))

# concatenate
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

######################################################################
def train(target,EPOCH = 5000, k=0, src_train=0,rescale=0):
    # EMBEDDING_DIM = 300
    # EMBEDDING_DIM = 1024
    # EMBEDDING_DIM = 500
    SOURCE_SIZE = 3
    best_test_acc = 0.0
    # print k
    psi_matrix = []
    X_joint = []
    y_joint = []
    if k == 0:
        psi_matrix =load_obj("%s/psi_matrix"%target)
        X_joint = load_obj("%s/X_joint"%target)
        y_joint = load_obj("%s/y_joint"%target)
    else:
        psi_matrix = load_obj("%s/%s/psi_matrix"%(target,k))
        X_joint = load_obj("%s/%s/X_psi"%(target,k))
        y_joint = load_obj("%s/%s/y_psi"%(target,k))
    
    psi_matrix = get_all(psi_matrix).T
    y_train_np = get_all(y_joint)
    y_train = label_to_tensor(y_train_np)
    # print np.array(psi_matrix).shape,np.array(X_joint).shape

    X_test_np = load_obj("%s/X_test"%target)
    y_test_np = load_obj("%s/y_test"%target)
    psi_test = compute_psi_for_test(X_joint,X_test_np)
    X_test = to_tensor(X_test_np)
    y_test = to_tensor(y_test_np).view(len(y_test_np),-1)
    psi_test = sent_to_tensor(psi_test)
    EMBEDDING_DIM =  X_test.size(1)

    pos_star = load_obj('%s/pos_star'%target)
    neg_star = load_obj('%s/neg_star'%target)
    X_star = concatenate(pos_star,neg_star)
    y_star = concatenate(np.ones(len(pos_star)),np.zeros(len(neg_star)))

    pos_proba = load_obj('%s/pos_proba'%target)
    neg_proba = load_obj('%s/neg_proba'%target)
    if len(pos_proba) == 1:
        # remove duplicate brackets
        pos_proba = np.array(pos_proba[0])[0]
        neg_proba = np.array(neg_proba[0])[0]
    proba = concatenate(pos_proba,neg_proba)
    # print proba.shape

    
    print "#train:",len(y_star)
    if src_train > 0:
        print "source training enabled"
        src_data = get_all(X_joint)
        src_data = [src_train*np.array(x) for x in src_data] if src_train != 1 else src_data
        src_labels = get_all(y_joint)
        src_cost = load_obj("%s/src_cost"%target) if k == 0 else load_obj("%s/%s/src_cost_psi"%(target,k))
        src_cost = get_all(src_cost)
        s = sum(src_cost)
        src_cost = [x/s for x in src_cost]
        # print np.array(src_cost).shape
        psi_src = compute_psi_for_test(X_joint,src_data)

        tgt_train = 1.0 - src_train
        X_star = [tgt_train*np.array(x) for x in X_star]
        X_star = concatenate(X_star,src_data)
        y_star = concatenate(y_star,src_labels)
        proba = concatenate(proba,src_cost)
        # print psi_matrix.shape,psi_src.shape
        psi_matrix = concatenate(psi_matrix,psi_src)
        print "UPDATED #train:", len(y_star)


    model = DomainAttention(embedding_dim = EMBEDDING_DIM, 
                           source_size = SOURCE_SIZE,
                           y = y_train)
    # LR = 2e-4
    LR = 1e-3
    # LR =1.0
    # optimizer = optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    # optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.SGD(model.parameters(),lr=0.01)
    # optimizer = optim.Adam([{'params': model.phi_srcs.parameters(),'lr':LR},
    #     {'params': model.bias,'lr':LR}])
    loss_function = nn.BCELoss(reduction='none') # Binary Cross Entropy Loss
    # loss_function = nn.BCELoss()
    # loss_function = nn.CrossEntropyLoss()
    no_up = 0

    for i in range(EPOCH):
        print 'epoch: %d start!' % i
        X,y,psi,cost = shuffle(X_star,y_star,psi_matrix,proba,random_state=0)#
        X = to_tensor(X)
        y = to_tensor(y).view(len(y),-1)
        psi = to_tensor(psi)
        cost = to_tensor(cost)
        # print X.shape,y.shape,psi.shape
        train_epoch(model,X,y,psi,cost,loss_function,optimizer,i,rescale=rescale)
        test_acc = evaluate_epoch(model, X_test, y_test, psi_test, loss_function)
        # evaluate_epoch(model,X_joint,y_joint,X_test_np,y_test_np)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if k == 0:
                os.system('rm ../work/%s/*.model'%target)
                print 'New Best Test!!!'
                filename = '../work/%s/best_model'%target
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                torch.save(model.state_dict(), '../work/%s/%s_'%(target,i) + str(int(test_acc*10000)) + '.model')
            else:
                os.system('rm ../work/%s/%s/*.model'%(target,k))
                print 'New Best Test!!!'
                filename = '../work/%s/%s/best_model'%(target,k)
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                torch.save(model.state_dict(), '../work/%s/%s/%s_'%(target,k,i) + str(int(test_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
        if no_up >= 20 and i > 100:
            exit()
        print "######################################################"
    pass

def train_epoch(model,X_train,y_train,psi_matrix,cost_list,loss_function,optimizer,i,rescale=0):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    # print X_train.size(),y_train.size(),psi_matrix.size(),cost_list.size()
    for sent,label,psi,cost in zip(X_train,y_train,psi_matrix,cost_list):
        # print sent.view(-1,len(sent)).shape
        # print torch.sum(psi)
        # label = label.data.item().max(1)[0]
        org_label = label.cpu().numpy()
        truth_res += list(org_label)
        
        pred = model(sent,psi)
        # pred_label = np.round(pred.data.item()) # gives 1 if > 0.5
        # pred_label = pred.data.max(1)[1].cpu().numpy()
        # print pred.view(1,-1).shape,label.shape
        pred_label = (pred.data>0.5).float()
        # pred_label = pred.cpu().numpy()
        pred_res.append(pred_label)
        # print pred_label,org_label
        
        # print pred,label
        # print label
        #.long()
        # assert (pred >= 0. & pred <= 1.).all()
        optimizer.zero_grad()
        if rescale==1:
            loss_function = nn.BCELoss(weight=cost,reduction='none')
            loss = loss_function(pred, label)
        else:
            loss = loss_function(pred, label)
        avg_loss += loss.data.item()
        # print model.bias
        # count += 1
        # if count % 100 == 0:
        #     print 'epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0])
        
        loss.backward()#retain_graph=True
        optimizer.step()
    # print truth_res[1032],pred_res[1032]
    avg_loss /= len(X_train)
    print 'epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res))
    pass


def evaluate_epoch(model, X_test, y_test, psi_test, loss_function):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    # print X_test.size(),y_test.size(),psi_test.size()
    for sent,label,psi in zip(X_test,y_test,psi_test):
        # print sent.size()
        # label = label.data.item().max(1)[0]
        org_label = label.cpu().numpy()
        truth_res += list(org_label)
        pred = model(sent,psi)
        # print pred,label
        # pred_label = np.round(pred.data.item())
        pred_label = (pred.data>0.5).float()
        # pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res.append(pred_label)
        #.long()
        loss = loss_function(pred, label)
        avg_loss += loss.data.item()
    # print pred_res
    avg_loss /= len(X_test)
    acc = get_accuracy(truth_res, pred_res)
    print('test avg_loss:%g acc:%g' % (avg_loss, acc))
    return acc
#####################################################################
# return an all-in-one matrix
def get_all(X_joint):
    return np.array([x for domain in X_joint for x in domain])

# accuracy
def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    correct = 0.0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            correct += 1.0
    return correct/len(truth)
    # return f1_score(truth,pred,average="macro")

if __name__ == '__main__':
    if len(sys.argv) >4:
        target = sys.argv[1]
        k = int(sys.argv[2])
        src_train = float(sys.argv[3])
        rescale = int(sys.argv[4])
        # epoch = int(sys.argv[3])
        print "target:",target,
        print "k:", k
        print "src_train:",src_train
        print "rescale:",rescale
        train(target,k=k,src_train=src_train,rescale=rescale)
    elif len(sys.argv) >2:
        target = sys.argv[1]
        k = int(sys.argv[2]) 
        print "target:",target,
        print "k:", k
        train(target,k=k)
    elif len(sys.argv) > 1:
        target = sys.argv[1]
        train(target)
    else:
        # run()
        print "usage: <target, k, src_train, rescale> or <target, k> or <target>"
    # toy_test()
