"""Explicit second geometry version: avoid majority-generated samples hiding target spread."""
import numpy as np
from .audit import *

def main():
    old=read(OLD/'scales.json');cohort=read('runs/reaction_flow/mode_supervision_v1/cohort.json');cal=load_records('calibration');prior=read(OUT/'GEOMETRY.json');fit=prior['fit_indices'];hold=prior['hold_indices'];pred=[];target=[]
    write(OUT/'STRATIFIED_SCALE_POLICY.json',dict(version='stratified-generated-real-IQR-v1',rule='max(generated IQR, real-target IQR, pooled IQR), then same group floor; same fit/hold split and bandwidth rule',reason='10 generated versus 4 target rows per episode can place real-target variation outside pooled IQR; preserve both marginal scales without deletion or group selection',new_hyperparameter_search=False))
    for i in fit:
        x=pair_data(cal[i],cal[i],old,cohort);s=x['scores'];pm=np.arange(96)//24<min(4,x['n']);pred.extend(np.where(pm,s['phi'],np.nan));target.extend(np.where(x['mask'],s['target_phi'],np.nan))
    iq=lambda x:np.nanquantile(x,.75,axis=0)-np.nanquantile(x,.25,axis=0)
    sc=np.maximum.reduce([iq(pred),iq(target),iq(pred+target)]);raw=sc.copy();floors={}
    for group,m in GROUPS.items():
        floor=max(.001,.1*float(np.median(sc[m&(sc>0)])));floors[group]=floor;sc[m]=np.maximum(sc[m],floor)
    scales={**old,'phi':sc.tolist()};ds=[]
    for i in fit:
        x=pair_data(cal[i],cal[i],scales,cohort);ds.extend(x['d2'][x['eligible']])
    h2=max(1e-6,float(np.median(ds)/(2*np.log(2))));hr=[cal[i] for i in hold];r=analyze(hr,hr,scales,cohort,h2);total=np.zeros(96)
    for i in hold:
        x=pair_data(cal[i],cal[i],scales,cohort);total+=x['sq'][x['eligible']].sum(0)
    total/=total.sum();maxchan=total.reshape(4,24).sum(0).max();groups={k:float(total[m].sum()) for k,m in GROUPS.items()}
    passed=maxchan<.5 and r['near_constant_column_fraction']<.9 and r['mean_rates']['eligible_pair_rate']>=.001
    write(OUT/'GEOMETRY_STRATIFIED.json',dict(version='stratified-generated-real-IQR-v1',scales=scales,h2=h2,group_floors=floors,heldout=r,qualified_groups=groups,qualified_max_channel_share=float(maxchan),passed=bool(passed),fit_indices=fit,hold_indices=hold,scope='geometry+bandwidth repair; not bandwidth-only'))
    print('stratified geometry',passed,'groups',groups,'max channel',maxchan,'h2',h2,flush=True)
if __name__=='__main__':main()
