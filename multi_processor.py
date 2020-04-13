"""
multiprocess script for each target domain

 
"""
import os,sys                                                                    
from multiprocessing import Pool 

def multi_run_wrapper(args):
    return run_process(*args)

def run_process(target,k,src_train,rescale):
        # os.system('time python att_model.py %s'%(target))
    
    print 'time python att_model.py %s %d %f %d'%(target,k,src_train,rescale)
    os.system('time python att_model.py %s %d %f %d'%(target,k,src_train,rescale))
    # os.system('time python exam_stars.py %s %d 0.5 dsc'%(target,k))
    pass

def main(target,src_train=0,rescale=0):
    processes = []
    ks = [0,5,10,30,50,70,100,200]
    # ks = [5,10,30,50,70,100,200]
    # ks = [100,500,1000,2000,3000,4000]
    for k in ks:
        processes.append((target,k,src_train,rescale))

    pool = Pool(processes=4)                                                        
    pool.map(multi_run_wrapper, processes) 
    pass

if __name__ == '__main__':
    
    if len(sys.argv) > 3:
        target = sys.argv[1]
        print target
        src_train = float(sys.argv[2])
        rescale = int(sys.argv[3])
        print "src_train",src_train
        print "rescale",rescale
        print "##############################"
        main(target,src_train,rescale)
    elif len(sys.argv) >1:
        target = sys.argv[1]
        print target
        main(target)
    else:
        print "usage: <target> or <target,src_train,rescale>"
