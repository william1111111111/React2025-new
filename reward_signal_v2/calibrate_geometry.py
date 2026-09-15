"""Separately versioned geometry repair after saved-data bandwidth-only diagnosis."""
import numpy as np
from .audit import OLD,OUT,GROUPS,load_records,pair_data,analyze
from reward_policy.common import read,write

def main():
    old=read(OLD/'scales.json');cohort=read('runs/reaction_flow/mode_supervision_v1/cohort.json');cal=load_records('calibration');fit=read(OUT/'BANDWIDTH.json')['fit_indices'];hold=read(OUT/'BANDWIDTH.json')['hold_indices'];features=[]
    for i in fit:
        x=pair_data(cal[i],cal[i],old,cohort);s=x['scores'];pm=np.arange(96)//24<min(4,x['n']);features.extend(np.where(pm,s['phi'],np.nan));features.extend(np.where(x['mask'],s['target_phi'],np.nan))
    features=np.array(features);iqr=np.nanquantile(features,.75,axis=0)-np.nanquantile(features,.25,axis=0);sc=iqr.copy();floors={}
    for group,mask in GROUPS.items():
        positive=iqr[mask&(iqr>0)];floor=max(.001,.1*float(np.median(positive))) if len(positive) else .001;sc[mask]=np.maximum(sc[mask],floor);floors[group]=floor
    scales={**old,'phi':sc.tolist()};fr=[cal[i] for i in fit];hr=[cal[i] for i in hold]
    ds=[]
    for r in fr:
        x=pair_data(r,r,scales,cohort);ds.extend(x['d2'][x['eligible']])
    h2=max(1e-6,float(np.median(ds)/(2*np.log(2))));result=analyze(hr,hr,scales,cohort,h2);share=np.array(result['coordinate_contribution']).reshape(4,24).sum(0)
    passed=result['near_constant_column_fraction']<.9 and result['mean_rates']['eligible_pair_rate']>=.001 and float(share.max())<.5
    write(OUT/'GEOMETRY.json',dict(version='joint-generated-real-TRAIN-scale-v1',scales=scales,group_floors=floors,h2=h2,fit_indices=fit,hold_indices=hold,fit_feature_rows=len(features),heldout=result,physical_coordinate_shares=share.tolist(),passed=bool(passed),scope='changes feature geometry as well as bandwidth; not an identical objective or bandwidth-only comparison',old_geometry_decision='do not launch bandwidth-only arm: 79% distance from one physical transformed channel across cosine modes, despite individual-mode median share check passing'))
    print('joint geometry',passed,'h2',h2,'groups',result['group_contribution'],'maxphysicalshare',share.max(),'coverage',result['mean_rates']['coverage'],flush=True)
if __name__=='__main__':main()
