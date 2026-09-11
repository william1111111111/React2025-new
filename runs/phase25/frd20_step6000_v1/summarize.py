import csv,json,hashlib
from pathlib import Path
import numpy as np
from hirp.phase15_audit import canonical_hash
root=Path('runs/phase25/frd20_step6000_v1')
protocol=json.loads((root/'protocol.json').read_text());rows=[];verified=0
for arm in ['C0','C1','C2']:
 model=f'seed123_{arm}_step6000';values=[];elapsed=0
 for i in protocol['source_indices']:
  matrix=np.zeros((10,10))
  for k in range(10):
   for j in range(10):
    p=root/model/f'{i:03d}_{k:02d}_{j:02d}.json';x=json.loads(p.read_text())
    assert canonical_hash(x['eval_identity'])==x['eval_identity_sha256']
    assert canonical_hash(x['result'])==x['result_sha256']
    assert x['eval_identity']['source']==i and x['eval_identity']['candidate']==k and x['eval_identity']['target']==j
    assert x['eval_identity']['protocol_sha256']==hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    matrix[k,j]=x['result']['distance'];elapsed+=x['result']['seconds'];verified+=1
  values.append(dict(model=model,source_index=i,FRD=float(matrix.min(1).sum())))
 with open(root/f'{arm}_per_source.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 rows.append(dict(model=model,step=6000,sources=20,pairs=2000,FRD=float(np.mean([v['FRD'] for v in values])),pair_compute_seconds=elapsed))
old=list(csv.DictReader(open('runs/phase25/timescale_v1/final_seed123_v1/analysis/FRD20.csv')))
mam=float(old[-1]['FRD'])
for row,prev in zip(rows,old):
 row['FRD_step2000']=float(prev['FRD']);row['change_from_step2000_pct']=100*(row['FRD']/row['FRD_step2000']-1);row['gap_vs_MAM_pct']=100*(row['FRD']/mam-1)
with open(root/'comparison.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 6000-step exact FRD20','', '同一预先固定20条Development source、K10、10目标、全有效帧、原版无约束DTW。各模型6000/6000次DTW、2000/2000配对全部完成；没有截断、banded近似或部分均值。','', '| 模型 | 2000步FRD ↓ | 6000步FRD ↓ | 相对2000步变化 |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['model']} | {r['FRD_step2000']:.6f} | {r['FRD']:.6f} | {r['change_from_step2000_pct']:+.2f}% |")
lines+=['',f'MAM同协议归档参考：{mam:.6f}（直接复用已有完整结果，未重新生成）。','', '仅20条开发子集的距离型属性指标，不是视频真实性。单训练seed；MAM预训练/预算/原生后处理不同。6000步预测来自既有带hash的完整任务缓存，未修改采样或模型。','', '原2000步结果、checkpoint、预测与报告均未覆盖。运行命令/退出状态/每配对原始值与hash保存在本目录。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
(root/'verification.json').write_text(json.dumps(dict(verified_pairs=verified,completed=verified==6000,source_indices=protocol['source_indices'],mam_reference_sha256=hashlib.sha256(Path('runs/phase25/timescale_v1/final_seed123_v1/analysis/FRD20.csv').read_bytes()).hexdigest()),indent=2))
print('\n'.join(lines))
