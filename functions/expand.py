import numpy as np
import pandas as pd
from ipywidgets import FloatProgress,FloatText
from tqdm import tqdm, tqdm_notebook
from functions.other import calc_features, get_event, score_event_fast, tag_bins

# after finding the tracks we want to improve the (kt,z0) for a track by minimizing it's features std
# the optimization is done by changing the (z0,kt) pair by random steps 
def refine_hipos(res,hits,stds,nhipos,phik=3.3,weights=None): 
    cols=list(res.columns)
    if weights is None:
        weights={'theta':0.15, 'phi':1.0}

    groups = res.merge(hits,on='hit_id',how='left')
    if not groups.columns.contains('kt'):
        groups['kt']=0
        groups['z0']=0
        print("No kt's, calculating")
    calc_features(groups,groups[['kt','z0']],phik)

    gp=groups.groupby('track_id').agg({'phi': np.std , 'sint' : np.std,
            'cost' : np.std}).rename(columns={ 'phi': 'phi_std', 
            'sint' : 'sint_std', 'cost':'cost_std'}).reset_index()
    groups=groups.merge(gp,on='track_id',how='left')
    groups['theta_std']=np.sqrt(weights['theta']*np.square(groups.sint_std)+weights['theta']*np.square(groups.cost_std))
    hipos=pd.DataFrame()
    for col in stds:
        hipos[col]=np.random.normal(scale=stds[col],size=nhipos)

    for hipo in tqdm(hipos.itertuples(),total=nhipos):

        groups['kt_new']=groups['kt']+hipo.kt
        groups['z0_new']=groups['z0']+hipo.z0
        calc_features(groups,groups[['kt_new','z0_new']].rename(columns={"kt_new": "kt", "z0_new": "z0"}),phik)
        gp=groups.groupby('track_id').agg({'phi': np.std , 'sint' : np.std,
            'cost' : np.std}).rename(columns={ 'phi': 'new_phi_std', 
            'sint' : 'new_sint_std', 'cost':'new_cost_std'}).reset_index()
        groups=groups.merge(gp,on='track_id',how='left')
        groups['new_theta_std']=np.sqrt(weights['theta']*np.square(groups.new_sint_std)+weights['theta']*np.square(groups.new_cost_std))

        old_std=np.sqrt(np.square(groups.theta_std)+weights['phi']*np.square(groups.phi_std))
        new_std=np.sqrt(np.square(groups.new_theta_std)+np.square(groups.new_phi_std))
        cond=(old_std<=new_std) 
        groups['kt']=groups['kt'].where(cond,groups.kt_new)
        groups['z0']=groups['z0'].where(cond,groups.z0_new)
        groups['theta_std']=groups['theta_std'].where(cond,groups.new_theta_std)
        groups['sint_std']=groups['sint_std'].where(cond,groups.new_sint_std)
        groups['cost_std']=groups['cost_std'].where(cond,groups.new_cost_std)
        groups['phi_std']=groups['phi_std'].where(cond,groups.new_phi_std)
        groups=groups.drop(['new_theta_std','new_phi_std','new_sint_std','new_cost_std'],axis=1)

        #pdb.set_trace()
    to_return=groups[cols+['theta_std','phi_std','sint_std','cost_std']]
    return to_return
    

    
# expands tracks by adding close loose hits to a track 
# min_track_len: the minimum length of a track to be expanded
# max_tack_len: a track wouldn't be expanded beyond that length
# max_expand: the maximum hits that can be added to a track
# to_track_len: hits from track's shotrer then this length would be considered "loose"
# mstd/dstd - the closeness to a track wa be measured absolutly (dstd) or with relation to its std (mstd). if dstd>0 it is used
# max_dtheta - used to add hits to tracks that rotated more then 180 degrees, a track must already rotate more then max_theta (in rad)
# mstd_size - the size of the expand area can be changed depanding on the tracks length
# mstd_volume - the size of the expand area can be changed depanding on the hits volume_id
def expand_tracks(res,hits,min_track_len,max_track_len,max_expand,to_track_len,mstd=1.0,dstd=0.0,phik=3.3,
                                                      max_dtheta=10,mstd_size=None,mstd_vol=None,drop=0,nhipo=1000,weights=None):
    if weights is None:
        weights={'theta':0.25, 'phi':1.0}

    if mstd_size is None:
        mstd_size=[0 for i in range(20)]
    if mstd_vol is None:
        mstd_vol={7:0,8:0,9:0,12:0,13:0,14:0,16:0,17:0,18:0}
    gp=res.groupby('track_id').first().reset_index()
    orig_hipo=gp[['track_id','kt','z0']]
    eres=res.copy()
    res_list=[]
    stds={'kt':7e-5,'z0':0.8}
    eres=refine_hipos(eres,hits,stds,nhipo,phik=phik,weights=weights)
    dum,eres['track_len']=tag_bins(eres['track_id'])
    eres['max_track_len']=np.clip(eres.track_len+max_expand,0,max_track_len) 
    eres['max_track_len']=2*(  eres['max_track_len']/2).astype('int')+1
    eres=eres.sort_values('track_len')
    eres = eres.merge(hits,on='hit_id',how='left')
    eres['sensor']=eres.volume_id+eres.layer_id*100+100000*eres.module_id
    group_sensors=eres.groupby('track_id').sensor.unique()
    groups=eres[eres.track_len>min_track_len].groupby('track_id').first().reset_index().copy()
    groups['order']=-groups.track_len 
    groups=groups.sort_values('order').reset_index(drop=True)
    groups=groups.head(int((1.0-drop)*groups.shape[0])).copy()
    select=eres.track_len<to_track_len
    grouped=eres[~select]
    regrouped=eres[select].copy()
    regrouped['min_dist']=100
    regrouped['new_track_len']=0
    regrouped['new_track_id']=regrouped['track_id']
    regrouped['new_kt']=regrouped['kt']
    regrouped['new_z0']=regrouped['z0']
    regrouped['new_max_size'] = max_track_len

    f = FloatProgress(min=0, max=groups.shape[0], description='calculating:') # instantiate the bar
    display(f) # display the bar

    for group_tul in tqdm(groups.itertuples(),total=groups.shape[0]):
        if group_tul.Index%20 ==0: f.value=group_tul.Index
        if group_tul.track_len>=max_track_len: continue
        group=eres[eres.track_id==group_tul.track_id].copy()
        calc_features(group,group[['kt','z0']],phik)
        group['abs_z']=np.abs(group.z)
        group['abs_theta']=np.abs(group.theta)
        phi_mean=group.phi.mean()
        sint_mean=group.sint.mean()            
        cost_mean=group.cost.mean()
        max_z=group.abs_z.max()
        max_theta=group.abs_theta.max()
        regrouped['abs_z']=np.abs(regrouped.z)
        calc_features(regrouped,group_tul,phik,double_sided=True)
        regrouped['dist'] =np.sqrt(weights['theta']*np.square(regrouped.sint-sint_mean)+weights['theta']*np.square(regrouped.cost-cost_mean)+weights['phi']*np.square(regrouped.phi-phi_mean))
        regrouped['dist2'] =np.sqrt(weights['theta']*np.square(regrouped.sint2-sint_mean)+weights['theta']*np.square(regrouped.cost2-cost_mean)+weights['phi']*np.square(regrouped.phi2-phi_mean))
        select = (regrouped.abs_z>max_z)  & (max_dtheta >max_dtheta) & (regrouped.dist2<regrouped.dist)
        regrouped['dist']=regrouped['dist'].where(~select,regrouped['dist2'])    
        cmstd=regrouped.volume_id.map(mstd_vol)+mstd_size[group_tul.track_len]+mstd
        if (dstd==0.0):
            sdstd==group.dstd
        else:
            sdstd=dstd
        better =( regrouped.dist<cmstd*sdstd) & ( regrouped.dist<regrouped.min_dist) & (~regrouped.sensor.isin(group_sensors.loc[group_tul.track_id]))
        regrouped['min_dist']=np.where(better,regrouped.dist,regrouped.min_dist)
        regrouped['new_track_id']=np.where(better,group_tul.track_id,regrouped.new_track_id)
        regrouped['new_z0']=np.where(better,group_tul.z0,regrouped.new_z0)
        regrouped['new_kt']=np.where(better,group_tul.kt,regrouped.new_kt)
        regrouped['new_track_len']=np.where(better,group_tul.track_len,regrouped.new_track_len)
        regrouped['new_max_size']=np.where(better,group_tul.max_track_len,regrouped.new_max_size)
    f.value=group_tul.Index
    regrouped=regrouped.sort_values('min_dist')
    regrouped['closest']=regrouped.groupby('new_track_id')['min_dist'].cumcount()
    better=regrouped.closest+regrouped.new_track_len>=regrouped.new_max_size
    regrouped['track_id']=regrouped['track_id'].where(better,regrouped['new_track_id'])
    res_list.append(grouped[['hit_id','track_id']])
    res_list.append(regrouped[['hit_id','track_id']])
    to_return=pd.concat(res_list)
    to_return=to_return.merge(orig_hipo,on='track_id',how='left')
    return to_return
        
