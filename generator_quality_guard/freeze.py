from .common import *
def main():
 assert read(OUT/'ACCEPTANCE.json')['passed'];assert read(OUT/'monitor/P2_reference/summary.json')['exact_pairs']==1600
 paths=[PARENT,OUT/'DESIGN.json',OUT/'SCHEDULE.json',OUT/'CALIBRATION.json',OUT/'monitor/manifest.json',OUT/'monitor/P2_reference/summary.json',Path('runs/reaction_flow/reward_signal_v2/GEOMETRY_STRATIFIED.json'),OLD/'scales.json',Path('runs/reaction_flow/bert_semantic_v1/content/index.json'),Path('runs/reaction_flow/bert_semantic_v1/content/vectors.npy'),Path('/home/zhengshiyi/react2025/framework/metrics/FRC.py'),Path('/home/zhengshiyi/react2025/framework/metrics/FRD.py'),Path('/home/zhengshiyi/react2025/framework/modules/post_processor.py')]
 protocol=dict(version='absolute-quality-guarded-NFT-v1',inputs={str(p):sha(p) for p in paths},fixed_mu_C=1.,fixed_mu_D=1.,gradient_diagnostics='first proposal and every20; logs missing diagnostics as null, never fabricated zeros',no_LR_adaptation=True,monitor_tolerances=dict(FRC=0.,FRD=0.),max_proposed=1000,stop_after_consecutive_rejections=3)
 code={str(p):sha(p) for folder in ['generator_quality_guard','generator_reward_nft','reward_policy','reaction_flow','semantic_supervision/bert_experiments'] for p in sorted(Path(folder).glob('*.py'))}
 for name,obj in [('PROTOCOL.json',protocol),('CODE_IDENTITY.json',code)]:
  path=OUT/name
  if path.exists():assert read(path)==obj
  else:write(path,obj)
if __name__=='__main__':main()
