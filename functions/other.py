import pandas as pd
import numpy as np
import pickle

def calc_features(hits,hipos,phik,double_sided=False):
    
    if not 'rr' in list(hits.columns):
        hits['theta_']=np.arctan2(hits.y,hits.x)
        hits['rr']=np.sqrt(np.square(hits.x)+np.square(hits.y))
    ktrr=hits.rr*hipos.kt
    hits['dtheta']=np.where((np.abs(ktrr)<1),np.arcsin(ktrr,where=(np.abs(ktrr)<1) ),ktrr)
    hits['theta'] = hits.theta_+hits.dtheta
    hits['phi'] = np.arctan2((hits.z-hipos.z0) ,phik*hits.dtheta/hipos.kt)*2.0/np.pi
    hits['sint']=np.sin(hits['theta'])
    hits['cost']=np.cos(hits['theta'])
    hits['fault']=(np.abs(ktrr)>1).astype('int')
    if double_sided:
        hits['phi2'] = np.arctan2((hits.z-hipos.z0) ,phik*(np.pi-hits.dtheta)/hipos.kt)*2.0/np.pi
        hits['theta2'] = hits.theta_+np.pi-hits.dtheta
        hits['sint2']=np.sin(hits['theta2'])
        hits['cost2']=np.cos(hits['theta2'])
    return hits

def tag_bins(cat):
    un,inv,count = np.unique(cat,return_inverse=True, return_counts=True)
    bin_tag=inv
    bin_count=count[inv]
    return bin_tag,bin_count

def hit_score(res,truth):
    tt=res.merge(truth[['hit_id','particle_id','weight']],on='hit_id',how='left')
    un,inv,count = np.unique(tt['track_id'],return_inverse=True, return_counts=True)
    tt['track_len']=count[inv]
    un,inv,count = np.unique(tt['particle_id'],return_inverse=True, return_counts=True)
    tt['real_track_len']=count[inv]
    gp=tt.groupby('track_id')
    gp=gp['particle_id'].value_counts().rename('par_freq').reset_index()
    tt=tt.merge(gp,on=['track_id','particle_id'],how='left')
    gp=gp.groupby('track_id').head(1)
    gp=gp.rename(index=str, columns={'particle_id': 'common_particle_id'})
    tt = tt.merge(gp.drop(['par_freq'],axis=1),on='track_id',how='left')
    tt['to_score']=(2*tt['par_freq']>tt['track_len']) & (2*tt['par_freq']>tt['real_track_len'])
    tt['score']=tt['weight']*tt['to_score']
    return tt


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# the following 2 functions are taken from outrunner's kernel: https://www.kaggle.com/outrunner/trackml-2-solution-example
def get_event(path,event):
    hits= pd.read_csv(path+'%s-hits.csv'%event)
    cells= pd.read_csv(path+'%s-cells.csv'%event)
    truth= pd.read_csv(path+'%s-truth.csv'%event)
    particles= pd.read_csv(path+'%s-particles.csv'%event)
    return hits, cells, particles, truth

def score_event_fast(truth, submission):
    truth = truth[["hit_id", "particle_id", "weight"]].merge(submission, how='left', on='hit_id')
    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])
    
    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    particles = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].particle_id.unique()
    return score

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission
