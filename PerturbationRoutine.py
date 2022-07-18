#import sys
#sys.path.append('../Project_Clustering')



import numpy as np
import math
import matplotlib.pyplot as plt
    
# extreme low value masking
def maskExtLow(W): 
    
    lim_upper =(W.max())*0.05
    lim_lower = (W.min())*0.05
    B1 = W > lim_upper
    B2 = W < lim_lower
    B = np.logical_or(B1, B2)
    #B = np.logical_not(B)
    B.astype(np.int)
    Wm = np.multiply (W, B)
    return Wm
    
    
# Expo masking

def maskExpo(W):
    
    w_max = np.max(W)
    alpha = np.divide (2.3, w_max) # constrains all weight values between 0 and 10
    Wm = np.exp(alpha*W)
    
    return Wm

    
#extreme boundary value masking
def maskExtBound(W):
    
    lower_limit = np.mean(W) - 3*np.std(W)
    upper_limit = np.mean(W) + 3*np.std(W)

    B1 = W > upper_limit
    B2 = W < lower_limit
    B = np.logical_or(B1, B2)
    B = np.logical_not(B)
    B.astype(np.int)
    Wm = np.multiply (W, B)
    
    return Wm
        
       
    
# negative weight masking
def maskNeg(W):
        
    B = W > 0.00
    B.astype(np.int)
    Wm = np.multiply (W, B)
    return Wm
    
#histogram based masking
def histMask(W,m,d):
        
    ind = m[:-1]
    lim = d.mean()
    m = ind[d>=lim]
    lim_lower = m.min()
    lim_upper = m.max()
    B1 = W > lim_upper
    B2 = W < lim_lower
    B = np.logical_or(B1, B2)
    #B = np.logical_not(B)
    B.astype(np.int)
    Wm = np.multiply (W, B)
    return Wm
    
def weight_perc(Wold,W,mtype):
        
    if mtype =='expo':
        
        Wn = maskExpo(W)
    
    if mtype == 'Low':
        
            Wn = maskExtLow(W)
            
    if mtype == 'Bound':
         
            Wn = maskExtBound(W)
              
    if mtype == 'Neg':
         
            Wn = maskNeg(W)
            
    Wt = np.multiply(Wn,Wold)
    Wold = Wt.astype(np.bool) 
    Wold = Wold.astype(np.int)
    cnt_zero = len(np.ravel(Wold))-np.count_nonzero(Wold)
    perc = (( cnt_zero)*100)/len(np.ravel(Wold))
  #  print( "% of weight masked", perc)  
   # perc_weights.append(perc)
    return Wold, perc
        
def scale_0_1 (data):

    scaled_data = np.divide((data-data.min()), (data.max()-data.min()))
    return scaled_data

def scale_1_1 (data):

    norm_data = scale_0_1(data)
    scaled_data = (norm_data*2)-1
    return scaled_data

def plotPercMask (perc_weight, fileName):
      
    fig = plt.figure()
    plt.plot([*range(len(perc_weight))], perc_weight,'b.-',linewidth=2,markersize=12)

  #  plt.scatter([*range(len(perc_weight))],perc_weight)
    plt.ylim(0,30)
    plt.xlim(0,len(perc_weight))
    
    plt.tick_params(axis='x', labelsize=14, labelcolor='k')
    plt.tick_params(axis='y', labelsize=14, labelcolor='k')
    plt.xlabel ('Number of perturbations',color='k',fontsize=14)
    plt.ylabel ('% of Weight Masked',color='k', fontsize=14)

    plt.savefig('Figure/Perc_Weight/'+fileName+ 'wt_drop.pdf', bbox_inches='tight')
#     plt.clf()
    plt.close(fig)
        
def plotLossCurve (loss_1, loss_2, fileName):
    
    fig = plt.figure()
    plt.plot(-np.log(loss_1[:1000]),'r.-',label='with masking')
    plt.plot(-np.log(loss_2[:1000]),'g.-',label="without masking")
    plt.legend(fontsize=14)

        
    plt.xlim((0,1000))
    plt.tick_params(axis='x', labelsize=14, labelcolor='k')
    plt.tick_params(axis='y', labelsize=14, labelcolor='k')
    plt.xlabel ('Number of epochs',color='k',fontsize=14)
    plt.ylabel ('Negative log loss',color='k', fontsize=14)
    
    plt.savefig('Figure/LossCurves/'+fileName+ 'loss.pdf', bbox_inches='tight')
  #  plt.clf()
    plt.close(fig)

def activationMap (dBName):
    
    if (dBName == 'ad_data' or dBName == 'heart_failure'):
        
        en_act = 'relu'
        de_act = 'relu'
        
    elif (dBName == 'gene_seq'):
        
        en_act = 'sigmoid'
        de_act = 'relu'
        
    else: 
    
        en_act = 'sigmoid'
        de_act = 'sigmoid'
        
        
    return en_act, de_act
    
    
        
    
