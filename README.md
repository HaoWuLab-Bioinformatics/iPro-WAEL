# iPro-WAEL

A Comprehensive and Robust Framework for Identifying Promoters in Multiple Species

## Framework
![image](https://github.com/HaoWuLab-Bioinformatics/iPro-WAEL/blob/main/Figure/Figure1.png)

## Overview
 
The folder "data" contains the data of the promoter, containing the sequences of 12 datasets for seven species.  
The folder "EPdata" contains the data of the enhancer and promoter, containing the sequences of the independent test sets and training sets. The first half of each file is labeled 1, and the second half is labeled 0.  
The folder "model" contains the trained models on 12 datasets of seven species for use or validation.    
The file "index_promoters.txt" and "word2vec_promoters.txt" are benchmark files used to extract word2vec features of human.  
The file "RF.py" is the code of the random forest model.  
The file "CNN.py" is the code of the CNN model.  
The file "Weighted_average.py" is the code of the weighted average algorithm.  
The file "main.py" is the code of the entire model and will import RF, CNN and Weighted_average.  
The file "feature_code.py" is the code used to extract word2vec features.  
The file "feature_importance.rar" is a compressed file for exploring important motifs, including code and specific results.  
The file "R.capsulatus.rar" is an example feature file to verify model performance (the feature file for the human dataset is too large to upload to github).  

## Dependency
Mainly used libraries:  
Python 3.6   
keras  2.3.1  
tensorflow 2.0.0  
sklearn  
numpy  
h5py  
See requirements.txt for all detailed libraries.  
## Usage
First, you should extract features of promoters, you can run the script to extract word2vec-based features as follows:  
`python feature_code.py`  
The extraction of other features is done using iLearnPlus [1].  
Then run the script as follows to compile and run iPro-WAEL:  
`python main.py`  
Note that the variable 'cell_lines' needs to be manually modified to change the predicted cell line.  
## Reference
1. Chen Z, Zhao P, Li C, et al. ILearnPlus: A comprehensive and automated machine-learning platform for nucleic acid and protein sequence analysis, prediction and visualization. Nucleic Acids Res. 2021; 49:1–19
