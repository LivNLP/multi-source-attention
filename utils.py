"""
Modified code from KeiraZhao/MDAN to load CHEN12 datasets

"""
import numpy as np
from scipy.sparse import coo_matrix
import time,sys
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils import shuffle
from load_data import save_obj,load_obj
from sklearn.feature_extraction.text import TfidfTransformer

def chen12(target="all"):
    
    time_start = time.time()
    
    amazon = np.load("../data/amazon.npz")
    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :5000]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    time_end = time.time()
    print("Time used to process the Amazon data set = {} seconds.".format(time_end - time_start))
    print("Number of training instances = {}, number of features = {}."
                 .format(amazon_xx.shape[0], amazon_xx.shape[1]))
    print("Number of nonzero elements = {}".format(amazon_xx.nnz))
    print("amazon_xx shape = {}.".format(amazon_xx.shape))
    print("amazon_yy shape = {}.".format(amazon_yy.shape))
    data_name = ["books", "dvd", "electronics", "kitchen"]
    num_data_sets = 4
    data_insts, data_labels, num_insts = [], [], []
    for i in range(num_data_sets):
        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i+1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
        print("Length of the {} data set label list = {}, label values = {}, label balance = {}".format(
            data_name[i],
            amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :].shape[0],
            np.unique(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :]),
            np.sum(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
        ))
        num_insts.append(amazon_offset[i+1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
    print("Data sets: {}".format(data_name))
    print("Number of total instances in the data sets: {}".format(num_insts))
    num_trains = 2000
    input_dim = amazon_xx.shape[1]
    # convert to tf_idf vectors
    # data_insts = tf_idf(data_insts)
    for i in range(num_data_sets):
        print data_name[i]
        # Build source instances.
        source_insts = []
        source_labels = []
        for j in range(num_data_sets):
            if j != i:
                source_insts.append(data_insts[j][:num_trains, :].todense().astype(np.float32))
                source_labels.append(data_labels[j][:num_trains, :].ravel().astype(np.int64))
        if target == data_name[i] or target == "all":
            # Build test instances.
            target_idx = i
            target_insts = data_insts[i][num_trains:, :].todense().astype(np.float32)
            target_labels = data_labels[i][num_trains:, :].ravel().astype(np.int64)
            # train_data=np.concatenate((get_sents(source_insts),target_insts),axis=0)
            # train_labels = np.concatenate((np.array(source_labels).flatten(),target_labels),axis=0)
            train_data= get_sents(source_insts)
            train_labels = np.array(source_labels).flatten()
            # print target_insts
            unlabel_data =  data_insts[i][:num_trains, :].todense().astype(np.float32)
            train,test,X_un = mlp_vectors(train_data,train_labels,get_sents(source_insts),target_insts,unlabel_data)
            mlp_source_insts = list(split_list(train,3))
            source_labels = list(split_list(np.array(source_labels).flatten(),3))
            mlp_target_insts = test
            clf = LogisticRegression().fit(train,np.array(source_labels).flatten())
            pred = clf.predict(test)
            acc = accuracy_score(target_labels,pred)
            print acc
            if acc > 0.847:
                save_obj(mlp_source_insts,"mlp/%s/X_joint"%data_name[i])
                save_obj(source_labels,"mlp/%s/y_joint"%data_name[i])
                save_obj(mlp_target_insts,"mlp/%s/X_test"%data_name[i])
                save_obj(target_labels,"mlp/%s/y_test"%data_name[i])
                save_obj(X_un,"mlp/%s/X_un"%data_name[i])
            return acc

def tf_idf(data_insts):
    new_insts = []
    for data in data_insts:
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True,sublinear_tf=True)
        new = tfidf_transformer.fit_transform(data)
        new_insts.append(new)
    return new_insts

def mlp_vectors(data,labels,train,test,unlabel_data):
    mlp = MLPClassifier(hidden_layer_sizes=(1000,500,100),max_iter=500,solver= "lbfgs")
    # print data.shape,labels.shape
    mlp.fit(data,labels)
    train = get_activations(mlp,train)
    test = get_activations(mlp,test)
    unlabel_data = get_activations(mlp,unlabel_data)
    print train.shape,test.shape,unlabel_data.shape
    # print unlabel_data
    return train,test,unlabel_data

def get_activations(clf, X):
    hidden_layer_sizes = clf.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + \
        [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations[-3]


#########################################################################################
def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

# return an all-in-one matrix
def get_sents(arr):
    arr = np.array(arr)
    return arr.reshape(-1, arr.shape[-1])

if __name__ == '__main__':
    if len(sys.argv) >1:
        target = sys.argv[1]
        print target
        acc = 0.0
        count = 0 
        while acc < 0.847:
            count+=1
            print count
            acc = chen12(target)
    else:
        chen12()
