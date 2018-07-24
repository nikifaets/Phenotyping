import numpy as np
import bcolz 
 
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, 
mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]
