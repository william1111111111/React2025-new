"""Additional baseline contrast from the already computed channel statistics."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
df=pd.read_csv(R/'ccc_channels.csv');rows=[]
for pairing in ['A_native','B_fixed_G0']:
    a=df[(df.pairing==pairing)&(df.model=='T0')].set_index(['source','candidate','channel'])
    b=df[(df.pairing==pairing)&(df.model=='G1-shared')].set_index(['source','candidate','channel']).loc[a.index]
    names=['official_r','scale_factor','mean_factor'];x=a[names].to_numpy();y=b[names].to_numpy();v=np.zeros_like(x)
    for order in itertools.permutations(range(3)):
        c=x.copy()
        for j in order:
            before=c.prod(1);c[:,j]=y[:,j];v[:,j]+=(c.prod(1)-before)/6
    assert np.max(np.abs(v.sum(1)-(b.ccc.to_numpy()-a.ccc.to_numpy())))<1e-12
    d=pd.DataFrame(v,index=a.index,columns=['r_contribution','scale_contribution','mean_contribution']).reset_index()
    for group,lo,hi in [('AU',0,15),('VA',15,17),('expression',17,25),('all',0,25)]:
        q=d[(d.channel>=lo)&(d.channel<hi)]
        for source,s in q.groupby('source'):
            rows.append(dict(pairing=pairing,comparison='G1-minus-T0',group=group,source=source,**s[['r_contribution','scale_contribution','mean_contribution']].mean().to_dict()))
pd.DataFrame(rows).to_csv(R/'ccc_G1_minus_T0_per_source.csv',index=False)
pd.DataFrame(rows).groupby(['pairing','comparison','group']).mean(numeric_only=True).drop(columns='source').to_csv(R/'ccc_G1_minus_T0_summary.csv')
