#import sys
#sys.path.append('../Project_Clustering')
import numpy as np
import math
import matplotlib.pyplot as plt
    
# extreme low value masking
def thresholdMaskLow(W, threshold): 
    
    
    lim_lower = np.percentile(W, threshold)
    B = W < lim_lower
    B.astype(np.int)
    Wm = np.multiply (W, B)
    return Wm
    
def weight_perc(Wold,W,threshold):
        
    Wn = thresholdMaskLow(W, threshold)        
    Wt = np.multiply(Wn,Wold)
    Wold = Wt.astype(np.bool) 
    Wold = Wold.astype(np.int)
    cnt_zero = len(np.ravel(Wold))-np.count_nonzero(Wold)
    perc = (( cnt_zero)*100)/len(np.ravel(Wold))
  #  print( "% of weight masked", perc)  
   # perc_weights.append(perc)
    return Wold, perc