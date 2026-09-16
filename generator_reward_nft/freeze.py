from pathlib import Path
from .common import *
def main():
    assert read(OUT/'ACCEPTANCE.json')['passed']
    paths=[PARENT,OUT/'DESIGN.json',OUT/'SCHEDULE.json',OUT/'CALIBRATION.json',V2/'GEOMETRY_STRATIFIED.json',OLD/'scales.json',Path('runs/reaction_flow/mode_supervision_v1/cohort.json'),Path('runs/reaction_flow/bert_semantic_v1/content/index.json'),Path('runs/reaction_flow/bert_semantic_v1/content/vectors.npy')]
    paths += [Path('generator_reward_nft/vendor/train_nft_sd3.py'),Path('generator_reward_nft/vendor/DiffusionNFT_LICENSE'),Path('generator_reward_nft/vendor/generator_reward_redesign/CODEX_GENERATOR_NFT.md')]
    protocol=dict(version='generator-reward-nft-v1',inputs={str(p):sha(p) for p in paths},acceptance_sha256=sha(OUT/'ACCEPTANCE.json'),objective='endpoint self-normalized NFT; core only analytical diagnostics; beta1; native noise->data',dynamic_scope='only domain-invalid penalty; per-group empirical motion diagnostic; no unvalidated speed anomaly classifier',budget=dict(updates=4000,rounds=200,groups_per_round=16,K=10,real_replay=16000),reference_cost='each new source-episode ref bank K10 shared by arms; creation and reuse logged separately',checkpoint_steps=[0,500,1000,2000,4000],DEV80=[1000,4000],exact_FRD20=[4000])
    code={str(p):sha(p) for folder in ['generator_reward_nft','reward_policy','reaction_flow','semantic_supervision/bert_experiments','mode_supervision'] for p in sorted(Path(folder).glob('*.py'))}
    for name,value in [('PROTOCOL.json',protocol),('CODE_IDENTITY.json',code)]:
        p=OUT/name
        if p.exists():assert read(p)==value
        else:write(p,value)
if __name__=='__main__':main()
