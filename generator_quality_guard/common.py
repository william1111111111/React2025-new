from pathlib import Path
from generator_reward_nft.common import read,write,sha,PARENT,digest,save,cpu_state
OUT=Path('runs/reaction_flow/generator_quality_guard_v1')
NFT=Path('runs/reaction_flow/generator_reward_nft_v1')
OLD=Path('runs/reaction_flow/reward_policy_v1')
ARMS=('Q0-quality-guarded','Q1-coverage-guarded')
