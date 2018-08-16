import numpy as np
import pandas as pd

from tqdm import tqdm_notebook
from sklearn.cluster.dbscan_ import dbscan

def merge_with_probabilities(sub1,sub2,preds1,preds2,truth=None,length_factor=0,at_least_more=0):
    """
    Input:  sub1 and sub2 are two dataframes, which assign track_ids to hits
            preds1 and preds2 are dataframes, which assign a quality (probability to be correct) to each track_id
            truth: if given, then calculate score of the merge
            length_factor: merge not only by quality, but also by length
            at_least_more: ask new quality to be at_least_more than old, to overwrite
    
    Output: new dataframe with updated track_ids
    """
    un,inv,count = np.unique(sub1['track_id'],return_inverse=True, return_counts=True)
    sub1['group_size']=count[inv]
    un,inv,count = np.unique(sub2['track_id'],return_inverse=True, return_counts=True)
    sub2['group_size']=count[inv]
    sub1=sub1.merge(preds1,on='track_id',how='left')
    sub2=sub2.merge(preds2,on='track_id',how='left')
    
    sub1['quality']=sub1['quality']+length_factor*sub1['group_size']
    sub2['quality']=sub2['quality']+length_factor*sub2['group_size']
    
    sub=sub1.merge(sub2,on='hit_id',suffixes=('','_new'))
    mm=sub.track_id.max()+1
    sub['track_id_new']=sub['track_id_new']+mm
    
    sub['quality']=sub['quality']+at_least_more
    cond=(sub['quality']>=sub['quality_new'])
    for col in ['track_id','z0','kt']:
        sub[col]=sub[col].where(cond,sub[col+'_new'])
    
    sub=sub[['hit_id','track_id','event_id','kt','z0']]
    if not truth is None:
        print('Score',score_event(truth,sub))
    
    # calculate track_ids again to make them smaller
    un,inv,count = np.unique(sub['track_id'],return_inverse=True, return_counts=True)
    sub['track_id']=inv
    return sub

def get_features(track_hits,cluster_size=10):    
    """
    Input: dataframe with hits of 1 track
    Output: array with features of track
    """
    nhits = len(track_hits)
    svolume=track_hits['volume_id'].values.min()
    X=np.column_stack([track_hits.x.values, track_hits.y.values, track_hits.z.values])
    _, labels = dbscan(X, eps=cluster_size, min_samples=1, algorithm='ball_tree', metric='euclidean')
    uniques = np.unique(labels)
    nclusters = len(uniques)
    nhitspercluster = nhits/nclusters
    xmax=track_hits['x'].values.max()
    xmin=track_hits['x'].values.min()
    xvar=track_hits['x'].values.var()
    ymax=track_hits['y'].values.max()
    ymin=track_hits['y'].values.min()
    yvar=track_hits['y'].values.var()
    zmax=track_hits['z'].values.max()
    zmin=track_hits['z'].values.min()
    zvar=track_hits['z'].values.var()
    zmean=track_hits['z'].values.mean()
    features=np.array([svolume,nclusters,nhitspercluster,xmax,ymax,zmax,xmin,ymin,zmin,zmean,xvar,yvar,zvar])
    return features

def get_predictions(sub,hits,model,min_length=4):  
    """
    Input: dataframe sub with track id for each hit, 
           dataframe hits with hit information, 
           model=ML model to get prediction
    Output: dataframe with predicted probability for each track
    """
    preds=pd.DataFrame()
    sub=sub.merge(hits,on='hit_id',how='left')
    trackids_long=[]
    trackids_short=[]
    features=[]
    
    trackids=np.unique(sub['track_id']).astype("int64")
    for track_id in tqdm_notebook(trackids):        
        track_hits=sub[sub['track_id']==track_id]
        if len(track_hits) < min_length:
            trackids_short.append(track_id)
        else:
            features.append(get_features(track_hits))
            trackids_long.append(track_id)

    probabilities_long=model.predict(np.array(features))
    probabilities_short=np.array([0]*len(trackids_short))
    
    preds['quality']=np.concatenate((probabilities_long,probabilities_short))
    preds['track_id']=np.concatenate((trackids_long,trackids_short))
    preds['quality']=preds['quality'].fillna(1)  # assume it is a good track, if no probability can be calculated
    return preds


def precision_and_recall(y_true, y_pred,threshold=0.5):
    tp,fp,fn,tn=0,0,0,0

    for i in range(0,len(y_true)):
        if y_pred[i]>=threshold:
            if y_true[i]>0:
                tp+=1
            else:
                fp+=1
        elif y_true[i]==0:
            tn+=1
        else:
            fn+=1
    precision=tp/(tp+fp) if (tp+fp != 0) else 0
    recall=tp/(tp+fn) if (tp+fn != 0) else 0
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    print('Threshold',threshold,' --- Precision: {:5.4f}, Recall: {:5.4f}, Accuracy: {:5.4f}'.format(precision,recall,accuracy))
    return precision, recall, accuracy