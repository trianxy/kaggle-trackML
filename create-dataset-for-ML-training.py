#Create the training and validation data for the ML algorithm
#by TX, has not been run, so may contain certain bugs. Also 2 todos remain as regards to gathering the wrong tracks (those which will get target=0)

from tqdm import tqdm
import pickle
from sklearn.cluster.dbscan_ import dbscan
import sys
sys.path.insert(0, '/home/kaggle/2018_08_13_TrackMLCern/trackml-library-master')
from trackml.dataset import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df_all_subs=pd.DataFrame()
min_hits=4

def save_obj(obj, filename):
    """
    Example:
        filename = "folder/filename.pkl
        arr = [3,4,5]
        save_obj(arr ,filename)
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("saved to " + filename)


def load_obj(filename):
    """
    Example:
        filename = "folder/filename.pkl
        arr = load_obj(arr ,filename)
    """
    with open(filename, 'rb') as f:
        print("loaded from " + filename)
        return pickle.load(f)


def get_features(track_hits):    
    nhits = len(track_hits)
    svolume=track_hits['volume_id'].values.min()
    
    X=np.column_stack([track_hits.x.values, track_hits.y.values, track_hits.z.values])
        _, labels = dbscan(X, eps=10, min_samples=1, algorithm='ball_tree', metric='euclidean')
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


def get_true_tracks(hits,particles,truth):
# speed of function can be improved, eg by calculating things in parallel (or even by not appending
# to a dataframe in each loop, but working with lists inside the loop
    particle_ids=particles.particle_id.values.copy()
    np.random.shuffle(particle_ids)
    df=pd.DataFrame()
    for particle_id in tqdm(particle_ids):
        track_hits = truth[truth['particle_id'] == particle_id]
        if len(track_hits)<min_hits:
            continue
        track_hits = track_hits[['hit_id']].merge(hits, on='hit_id',how='left')
        d=get_features(track_hits)
        df=df.append(d,ignore_index=True)
    return df



for i in range(0,20):
    event='event000001{:03d}'.format(i)
    print('do event',event)
    hits, cells, particles, truth = load_event('input/train_1/{}'.format(event))
    
    # Need to have ready, and then load, actual runs of the binning algorithm. This is  
    # to get wrong tracks, which are as close as possible to the actual wrong tracks, which the 
    # the ML should give a small probability
    sub=pd.read_csv('input/train_1/res-a/binning_run_{}.csv'.format(event),index_col=False)
    

    #todo 1: Select all WRONG tracks as follows. Take tracks, which: 
    #      (1) have at least 4 hits, and at most 23 hits
    #      (2) not all hits belong to the same particle_id (eg by merging the particle_id onto the 
             # the submission dataframe, then count how many unique particle_ids each track has and only
             # consider those that have >= particle_ids
    
    #todo 2: Create a dataframe df_wrong, such that each row in this dataframe is one track. In addition
    #        add all features of this track (which will be used later in the ML algorithm). Each feature
    #        is one column
    df_wrong['target']=0
    df_true=get_true_tracks(hits,particles,truth)    
    df_true['target']=1
    df_both=pd.concat([df_true,df_wrong],ignore_index=False,sort=True)
    df_both['event_id']=i
    df_both=df_both.sample(frac=1).reset_index(drop=True)  # shuffle
    df_all_subs=df_all_subs.append(df_both	, ignore_index=True)


df_train=df_all_subs[df_all_subs[“event_id”]<=15]
df_test=df_all_subs[df_all_subs[“event_id”]>15]
    
save_obj(df_train,”df_train_v2-reduced.pkl”)
save_obj(df_test,”df_test_v1.pkl”)

