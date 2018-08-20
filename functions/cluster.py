import time
from functions.other import calc_features, get_event, score_event_fast, tag_bins
import numpy as np
import pandas as pd
from ipywidgets import FloatProgress,FloatText
from tqdm import tqdm, tqdm_notebook

label_shift_M=1000000

def sparse_bin(features,bin_num,randomize=True,fault=None):
    err=np.random.rand(features.shape[1])*randomize
    cat=np.zeros(features.shape[0]).astype('int64')
    factore=1
    for i,feature in enumerate(features.columns):
        cat=cat+(features[feature]*bin_num._asdict()[feature]+err[i]).astype('int64')*factore
        factore=factore*(2*bin_num._asdict()[feature]+1)
    if not fault is None:
        cat=cat+(factore*features.index*fault).astype('int64')
    return tag_bins(cat)
    
    
def clustering(hits,stds,filters,phik=1.0,nu=500,weights=None,res=None,truth=None,history=None,pre_test_points=None):
    start = time.time()
    rest = hits.copy()
    if weights is None:
        weights={'phi':1, 'theta':0.15}
    
    # This part is only for printing while running
    calc_score = not truth is None
    if not history is None:
        hist_list=[]
    if calc_score:
        rest = rest.merge(truth[['hit_id','particle_id','weight']],on='hit_id',how='left')
        dum,rest['particle_track_len']=tag_bins(rest['particle_id'])
        score = 0 
        hit_num=0
        total_num=0
        frs=FloatText(value=0, description="full score:")
        display(frs)
        fs=FloatText(value=0, description="score:")
        display(fs)
        fss=FloatText(value=0, description="s rate:")
        display(fss)
        fsd=FloatText(value=0, description="add score:")
        display(fsd)    
    ft = FloatText(value=rest.shape[0], description="Rest size:")
    display(ft)
    fg = FloatText(value=rest.shape[0], description="Group size:")
    display(fg)
    fgss = FloatText(description="filter:")
    display(fgss)
    #End of printing part
    
    # if res in not None we continue to work on a partial solution: res otherwise - initialize
    if res is None:
        rest['track_len']=1
        rest['track_id']=-rest.index
        rest['kt']=1e-6
        rest['z0']=0
    else:
        rest=rest.merge(res[['hit_id','track_id','kt','z0']],on='hit_id',how='left')
        dum,rest['track_len']=tag_bins(rest['track_id'])

    res_list=[]
    rest['sensor']=rest.volume_id+rest.layer_id*100+100000*rest.module_id
    rest['layers']=rest.volume_id+rest.layer_id*100
    
    # This function can use random (z0,kt) pairs or get a predefined list
    if pre_test_points is None:
        maxprog= filters.npoints.sum()
    else:
        maxprog = filters.shape[0]*pre_test_points.shape[0]
    pbar = tqdm(total=maxprog,mininterval=5.0)
    rest['pre_track_id']=rest['track_id']
    p=-1
    feature_cols=['theta','sint','cost','phi','rr','theta_','dtheta','fault']
    
    # this is the main clustering loop.
    # a filter defines:
    # The bins' width: filter.theta , filter.phi
    # The minimal length of a track to be taken out of hits: min_group
    # npoint - is the number of times to us this filter
    for filt in filters.itertuples():
        if pre_test_points is None:
            test_points=pd.DataFrame()
            for col in stds:
                test_points[col] = np.random.normal(scale=stds[col],size=filt.npoints)
        else:
            test_points=pre_test_points.sample(frac=filt.npoints).reset_index(drop=True)
        
        for row in test_points.itertuples():
            p=p+1
            pbar.update()
            calc_features(rest,row,phik)
            # clustering using sparse_bin and the decide if to use the new track_id depending on the length of the track
            rest['new_track_id'],rest['new_track_len']=sparse_bin(rest[['phi','sint','cost']],filt,fault=rest.fault)
            rest['new_track_id']=rest['new_track_id']+(p+1)*label_shift_M
            better = (rest.new_track_len>rest.track_len) & (rest.new_track_len<19)
            rest['new_track_id']=rest['new_track_id'].where(better,rest.track_id)
            dum,rest['new_track_len']=tag_bins(rest['new_track_id'])
            better = (rest.new_track_len>rest.track_len) & (rest.new_track_len<19)
            rest['track_id']=rest['track_id'].where(~better,rest['new_track_id']) 
            rest['track_len']=rest['track_len'].where(~better,rest['new_track_len'])
            rest['kt']=rest['kt'].where(~better,row.kt)
            rest['z0']=rest['z0'].where(~better,row.z0)
            
            # every nu loops we will do a bit of outliner removal
            # and set a permanent tracks which are long enough bit removing their hits from rest
            if (((row.Index+1)%nu == 0) or (row.Index + 1 == test_points.shape[0])):
                dum,rest['track_len']=tag_bins(rest['track_id'])
                calc_features(rest,rest[['kt','z0']],phik)
                # If two hits in the same track are from the same detector, we choose the one closest to the tracks center of gravity
                # If 3 hits are from the same layer we choose the closest 2
                gp = rest.groupby(['track_id']).agg({'phi': np.mean , 
                    'sint':np.mean, 'cost':np.mean}).rename(columns={ 'phi': 'mean_phi', 
                                'sint':'mean_sint', 'cost':'mean_cost'}).reset_index()
                cols_to_drop = rest.columns.intersection(gp.columns).drop('track_id')
                rest = rest.drop(cols_to_drop,axis=1).reset_index().merge(gp,on=['track_id'],how = 'left').set_index('index')
                rest['dist'] = weights['theta']*np.square(rest.sint-rest.mean_sint)+ weights['theta']*np.square(rest.cost-rest.mean_cost)+ weights['phi']*np.square(rest.phi-rest.mean_phi)
                rest=rest.sort_values('dist')
                rest['closest']=rest.groupby(['track_id','sensor'])['dist'].cumcount()
                rest['closest2']=rest.groupby(['track_id','layers'])['dist'].cumcount()
                select = (rest['closest']!=0) | (rest['closest2']>2)  
                rest['track_id']=rest['track_id'].where(~select,rest['pre_track_id'])
                dum,rest['track_len']=tag_bins(rest['track_id'])
                
                fgss.value=filt.phi
                fg.value=filt.min_group
                ft.value = rest[rest.track_len<=filt.min_group].shape[0]

                select = (rest['track_len']>filt.min_group)
                
                #The next lines are just for printing
                if calc_score:
                    tm=rest[select]                   
                    gp = tm.groupby(['track_id','particle_id'])['hit_id'].count().rename('par_count').reset_index()
                    tm=tm.merge(gp,on=['track_id','particle_id'],how='left')
                    gp = rest.groupby(['track_id','particle_id'])['hit_id'].count().rename('par_count').reset_index()
                    rs=rest.merge(gp,on=['track_id','particle_id'],how='left')
                    to_full_score=(rs.weight*((rs.par_count*2>rs.track_len) & (rs.par_count*2>rs.particle_track_len)))
                    frs.value=to_full_score.sum()+fs.value
                    to_score=(tm.weight*((tm.par_count*2>tm.track_len) & (tm.par_count*2>tm.particle_track_len)))
                    hit_num=hit_num+(to_score>0).sum()
                    total_num=total_num+tm.weight.sum()
                    fs.value=fs.value+to_score.sum()
                    fss.value=fs.value/total_num
                    fsd.value=to_score.sum()
                    gp = rest.groupby(['track_id','particle_id'])['hit_id'].count().rename('par_count').reset_index()
                    rs=rest.merge(gp,on=['track_id','particle_id'],how='left')
                    to_full_score=(rs.weight*((rs.par_count*2>rs.track_len) & (rs.par_count*2>rs.particle_track_len)))
                    frs.value=to_full_score.sum()+fs.value-to_score.sum()
                    if not history is None:
                        hist_list.append(pd.DataFrame({'P':p,'ftheta':filt.phi,'added_score':to_score.sum(),'min_group':filt.min_group,
                                                    'full_score':frs.value,'score':fsd.value,'correct':fss.value,
                                                    'clustered':tm.shape[0],'left':rest.shape[0]-tm.shape[0]}, index=[0]))

                #end of printing part 
                
                tm=rest[select][['hit_id','track_id','kt','z0']]
                res_list.append(tm)
                rest = rest[~select]
                dum,rest['track_len']=tag_bins(rest['track_id'])
                rest['pre_track_id']=rest['track_id']

    ft.value = rest.shape[0]
    res_list.append(rest[['hit_id','track_id','kt','z0']].copy())
    res = pd.concat(res_list, ignore_index=True)
    pbar.close()
    rest['track_id'],dum=tag_bins(rest['track_id'])
     
    if not history is None:
        history.append(pd.concat(hist_list,ignore_index=False))
    print ('took {:.5f} sec'.format(time.time()-start))
    return res 
