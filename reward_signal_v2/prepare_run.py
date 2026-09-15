import numpy as np
from pathlib import Path
from .audit import OUT,OLD
from reward_policy.common import read,write,sha

def main():
    geometry=read(OUT/'GEOMETRY_STRATIFIED.json');assert geometry['passed']
    probes=geometry['hold_indices'][:2]
    inputs=[OLD/'PROTOCOL.json',OLD/'episodes.json',OLD/'scales.json',OUT/'GEOMETRY_STRATIFIED.json',OUT/'PREFLIGHT.json',Path('reward_policy/policy.py')]
    write(OUT/'PROTOCOL.json',dict(arms=['R0-baselined','R1-calibrated'],updates=100,B=2,K=10,seed=123,lr=1e-5,beta_KL=.01,clip_norm=1.,optimizer='AdamW weight_decay=0',parent_sha256=read(OLD/'PROTOCOL.json')['parent_sha256'],estimator='other-candidate quality baseline / original K; separate uncentered marginal coverage',R0_geometry='old scales, h2=1, valid-mode masks',R1_geometry='stratified-generated-real-IQR-v1 plus fitted h2; objective changes feature geometry and bandwidth, not a bandwidth-only causal contrast',geometry_selection='all saved-data versions retained; no DEV signals used; pooled-only IQR does not protect minority real-target spread, so stratified max-IQR is used',probe_indices=probes,probe_steps=[0,25,50,100],probe_visibility='independent TRAIN calibration episodes only',training_records='exact first200 original episodes, 100 updates',formal_action_rollouts_each=2000,probe_rollouts_each=80,fresh_reference_rollouts=0,old_reference_cost='original independent parent tables reused; do not claim they cost zero originally',dual=read(OLD/'PROTOCOL.json')['dual'],input_hashes_preserved=True,inputs={str(p):sha(p) for p in inputs},no_automatic_push=True,DEV_evaluation='not scheduled in this signal-chain phase; no final task-performance claim'))
    paths=list(Path('reward_signal_v2').glob('*.py'))+list(Path('reward_policy').glob('*.py'))+[Path('mode_supervision')/p for p in ('plans.py','data.py','prepare.py')]
    write(OUT/'CODE_IDENTITY.json',{str(p):sha(p) for p in paths})
if __name__=='__main__':main()
