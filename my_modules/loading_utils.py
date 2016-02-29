import pandas as pd
import numpy as np
import gc

def read_my_data(fname,**kwargs):
    """Load my data from file into np.arrays.
    
    I had to use garbage collector, because pandas read_csv leaves garbage around.
    """
    
    #load data
    print "Loading data... "
    x=pd.read_csv(fname,sep='\t',header=None)
    
    # for some reason not everything is cleaned up
    #when using the pandas read_csv
    gc.collect()
    
    #probe_id=x[0]
    #y=x.iloc[:,-1].values.astype(np.int8)
    #x=x.iloc[:,1:-1].values.astype(np.int8)
    #return probe_id,x,y
    
    return x[0],x.iloc[:,1:-1].values.astype(np.int8),x.iloc[:,-1].values.astype(np.int8)

def create_sets(x,y,N_train=8000,N_valid=1000,N_test=1000,length=1000):
    """Create train,valid,test sets from data."""
    #select a random subset of data
    rng = np.random.RandomState(23455) # not so random now
    perm=rng.permutation(len(y))
    train_idx=perm[:N_train]
    valid_idx=perm[N_train:N_train+N_valid]
    test_idx=perm[N_train+N_valid:N_train+N_valid+N_test]
    
    #determine the slice of data needed
    if length%2==0:
        start=500-length/2
    else:
        start=500-length/2 - 1
        
    train_x=np.int8(x[train_idx,start:start+length])
    valid_x=np.int8(x[valid_idx,start:start+length])
    test_x=np.int8(x[test_idx,start:start+length])

    train_y=np.int8(y[train_idx])
    valid_y=np.int8(y[valid_idx])
    test_y=np.int8(y[test_idx])
    
    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)

def create_sets_f32(x,y,N_train=8000,N_valid=1000,N_test=1000,length=1000):
    """Create train,valid,test sets from data."""
    #select a random subset of data
    rng = np.random.RandomState(23455) # not so random now
    perm=rng.permutation(100000)
    train_idx=perm[:N_train]
    valid_idx=perm[N_train:N_train+N_valid]
    test_idx=perm[N_train+N_valid:N_train+N_valid+N_test]
    
    #determine the slice of data needed
    if length%2==0:
        start=500-length/2
    else:
        start=500-length/2 - 1
        
    train_x=np.float32(x[train_idx,start:start+length])
    valid_x=np.float32(x[valid_idx,start:start+length])
    test_x=np.float32(x[test_idx,start:start+length])

    train_y=np.int32(y[train_idx])
    valid_y=np.int32(y[valid_idx])
    test_y=np.int32(y[test_idx])

    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)

def load_data(fname,N_train=8000,N_valid=1000,N_test=1000,length=1000):
    """Load and segment my data into random train,valid,test sets."""
    (train_x,train_y),(valid_x,valid_y),(test_x,test_y) = create_sets(
        read_my_data(fname),N_train,N_valid,N_test,length)
    
    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)


def load_data_4_theano(fname,N_train=8000,N_valid=1000,N_test=1000,length=1000):
    """Load and segment my data into random train,valid,test sets for theano."""
    (train,valid,test)=load_data(fname,N_train,N_valid,N_test,length)
   
    train_x,train_y=map(theano.shared,train)
    valid_x,valid_y=map(theano.shared,valid)
    test_x,test_y=map(theano.shared,test)
    
    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)
