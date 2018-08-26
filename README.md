# Solution #7 for competition trackML
## **Please note this is still work in progress**
Yuval Reina:  Yuval.reina@gmai.com
Trian Xylouris: t.xylouris@gmail.com

Below you can find a outline of how to reproduce our solution for the trackML competition.

For any questions, please contact us 

## ARCHIVE CONTENTS
- files: Directory containing the competition's event files and the user prepared training files
  - df_test_v1.pkl				:user perpared validation files
   - df_train_v2_reduced.pkl			:user prepared training file
   - event*-*.csv 				:competition's event files
- functions: Directory with python code
  - cluster.py			:the clustering functions
  - expand.py			:expanding functions
  - ml_model.py			:fuctions related to the Machine Learning algorithm
  - other.py			:utility functions
- trackml-library-master: Direcoty competition utility files (https://www.kaggle.com/c/trackml-particle-identification/discussion/55708)
- conda_python-dependencies.yml		:conda environment file
- create clustering.ipynb				:jupyter notebook, used to create solutions for traning
- Create training.ipynb				:jupyter notebook, used to create training files
- trackML_solution.ipynb				:jupyter notebook, our main solution notebook		

## HARDWARE: 
We used verious hardware to train and run our solution

any modern computer which can run ipython and jupyter notebooks will be ok

The software was tested on windows 10 and ubuntu

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
Conda - 4.6.11

Python 3.6

IPython 6.2.1

On linux machine you can build your conda environment like this:

conda env create -f conda_python-dependencies.yml

## Preparing training files
The notebooks used to prepare the training files are:

1. create clustering.ipynb - used to create solutions, which are used to select false tracks for training.

Change *path* to point to the path where you put the training events from kaggle

Change *out_path* to point to the path where you want to store the clustring results

If you want to try the ML algorithm with another solution algorithm, you can still use clustering to build false tracks, or use your algorithm to do it.

2. Create training.ipynb - used to create the training files. 

Change *train_path* to point to the path where you put the training events from kaggle

Change *clustered_path* to point to the path where you storeed the clustring results

The training results would be stored if the directory 'files'

## Running
Follow trackML_solution.ipynb and run the full solution.



You can also see an run most mart of this solution on [kaggle's kernel](https://www.kaggle.com/yuval6967/7th-place-clustering-extending-ml-merging-0-75)
