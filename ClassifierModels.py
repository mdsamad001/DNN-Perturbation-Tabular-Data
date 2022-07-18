# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:23:28 2018

@author: mdsamad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

#import figurePlot  
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import ( pipeline, preprocessing)
from sklearn.ensemble import GradientBoostingClassifier


#'GBF' -- Gradient boosting forest


   
def crossValid (Xdata, y, clfmodel, nfolds=10):
   
    
    n_classes = len(np.unique(y))
    
    
    testPredict = []
    Stest = []
     
    # Classifier model selection 
    if clfmodel == 'GBT':
    
        clf = GradientBoostingClassifier(n_estimators= 5,
                                random_state=42
                               )
    
        param_grid = {
                      "clf__n_estimators":[50, 80 ,110],
           #   "min_samples_split": [2, 5, 10],
            "clf__learning_rate":[0.1, 0.5],
             "clf__max_depth": [2, 5, 10]
             }
    
    
    kfold = StratifiedKFold(n_splits= nfolds, random_state= 42, shuffle=True)
    
    pipe = pipeline.Pipeline([('scl', preprocessing.StandardScaler()), ('clf', clf)])
    
    fold_auc = []
    accuracy_scores = []
    
    # print('Start time', datetime.now().time())
    
    for k, (train, test) in enumerate (kfold.split(Xdata, y)):
         
        # Accumulating training fold data and labels
        trfold = Xdata [train]#[Xdata [x] for x in train]
        trSen  = y[train]#[y[x] for x in train]
        
      #  print (trfold)
        
       # print (tsfold)
        
        # Accumulating test fold data and labels
        tsfold = Xdata[test]#[Xdata[x] for x in test]
        tsSen = y[test]# [y[x] for x in test]
        
        #print (trSen)
        
        gsModel = GridSearchCV(estimator=pipe, 
                                  param_grid = param_grid, n_jobs=1,
                                  scoring = 'accuracy', cv = 2)
        
    
        # Training and validation
        trModel = gsModel.fit(trfold,trSen)
        
        
        bestModel = trModel.best_estimator_
        
        # test with the best model found in the validation step
        testPred =  bestModel.predict(tsfold)
        
      
        
        # print ('Accuracy is:',accuracy_score( tsSen, testPred))
        accuracy_scores.append(accuracy_score( tsSen, testPred))
        print(classification_report(tsSen, testPred))
    
    # print(datetime.now().time())
    # print ('Mean 10-fold Accuracy', np.mean(accuracy_scores))
    # print ('Mean 10-fold AUC', np.mean(fold_auc))
    # print ('Std 10-fold AUC', np.std(fold_auc))
    
    return np.mean(accuracy_scores),np.std(accuracy_scores)
          

    
