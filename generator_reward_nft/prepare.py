import random,subprocess
from pathlib import Path
from .common import *

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'SCHEDULE.json').exists():return
    rows=read('runs/reaction_flow/mode_supervision_v1/cohort.json');eligible=[]
    for row in rows:
        blocks=[b for b in range((row['source_length']+749)//750) if min(row['source_length'],row['paired_length'])-b*750>=2]
        if blocks:eligible.append((row['index'],blocks))
    rng=random.Random(123)
    def schedule(n,offset):
        result=[]
        while len(result)<n:
            cycle=eligible.copy();rng.shuffle(cycle)
            for idx,bs in cycle:
                if len(result)==n:break
                result.append(dict(step=offset+len(result),session_id=rows[idx]['clip_id'].split('/')[1],source_indices=[idx],block_indices=[rng.choice(bs)],target_slots=[rng.randrange(4)]))
        return result
    write(OUT/'SCHEDULE.json',dict(training=schedule(3200,600000),replay=schedule(16000,700000),population=len(eligible),population_indices=[i for i,b in eligible]))
    write(OUT/'DESIGN.json',dict(parent_sha256=sha(PARENT),reviewed_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=123,arms=ARMS,rounds=200,groups_per_round=16,K=10,updates=4000,effective_batch=8,microbatch=1,replay_per_update=4,lr=2e-5,weight_decay=.01,warmup=100,clip=1.,beta=1.,old_EMA=.5,FM_weight=.25,ref_vector_field_weight=.01,tau=[.02,.98],precision='FP64 as native P2; FM reduction FP32 as existing protocol',dynamic_policy='domain-invalid penalty; TRAIN per-group motion references diagnostic only, no scenario-validated anomaly labels available',reference_revision='cbb14f84b8312b620390dfcbe2fab69c1383b104',reference_repository='https://github.com/NVlabs/DiffusionNFT',source='scripts/train_nft_sd3.py endpoint normalization; time converted data->noise to noise->data',evaluation=dict(DEV80=[1000,4000],exact_FRD20=[4000],K=10,source_only=True),no_auto_push=True))
if __name__=='__main__':main()
