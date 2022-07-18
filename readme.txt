
PerturbationRoutine.py ----> Equation number 12 with the proposed perturbation routine 

 
TABLE # 2 IN THE PAPER
_________________________

A. Basic DNN Studies - basic DNN folder (Perturbation, Dropout, Baseline)

        
B. Autoencoder Perturbation 
             
           - AE_baseline.ipynb -- no perturbation or dropout on DAE and SDAE implementations
           - AE_dropout.ipnynb --- Dropout during retraining of DAE and SDAE 
           - AE_perturbation.ipnyb --- Perturbation during retraining of DAE and SDAE


C. L1_L2 Regularization - L1 and L2 regularizations scripts on baseline models


TABLE 3 IN THE PAPER 
_____________________

GBT_ML.ipynb - script for Gradient Boosting Tree classifier model

ClassifierModels.py - GBT classifier 5x2 cross-validation implementation 


TABLE 4 IN THE PAPER
_____________________

Compression Experiments 

	- Han masking (Han et al. adapted to do percentile based pruning) - baseline pruning
        - DAE_SDAE_COMPRESSION.ipynb -- baseline pruning of DAE and SDAE
        - DNN_COMPRESSION.ipynb --- baseline pruning of basic DNN


Figure 4 - Ablation Study Folder

