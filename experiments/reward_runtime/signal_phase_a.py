"""No-update fixed-input replay of old identity/final policies; keep scope explicit."""
import numpy as np,torch
from reward_signal_v2.train import Run
from reward_signal_v2.audit import OUT,OLD,load_records
from reward_policy.common import read,write
from reaction_flow.train import configure_flow

def main():
    configure_flow();torch.set_num_threads(4);probes=[read(OLD/'episodes.json')['calibration'][i] for i in read(OUT/'PROTOCOL.json')['probe_indices']]
    for arm in ('R-quality','R-distance','R-coverage'):
        torch.manual_seed(123);r=Run('R0-baselined');dest=OUT/'phase_a_replay'/arm;dest.mkdir(parents=True,exist_ok=True);r.probe(probes,dest);s=torch.load(OLD/'training'/arm/'checkpoints/step_000500.pt',map_location='cpu',weights_only=True);r.policy.load_state_dict(s['policy']);r.dual=np.array(s['dual']);r.step=500;r.probe(probes,dest)
        write(dest/'SCOPE.json',dict(no_updates=True,parameter_action_noise_output_metrics='old checkpoint versus matching identity; same fixed TRAIN inputs',gradient_scope='recomputed with B1 control-variate for this replay; original estimator gradients remain in archived KL_gradient_diagnostic_1789456439.json',extra_action_rollouts=40,reference_draws_reused=40))
        print('phase A replay',arm,flush=True)
    old=read(OLD/'scales.json');before=[];after=[]
    for row in load_records('reference_table'):
        s=row['scores'];q=np.max(s['ccc'],1)/old['c']-np.min(s['dtw'],1)/old['d'];k=len(q);before.extend((q/k)**2);after.extend(((q-(q.sum()-q)/(k-1))/k)**2)
    write(OUT/'B1_CREDIT_SCALE_AUDIT.json',dict(scope='independent saved parent references; squared credit magnitude, not a measured gradient-variance or SNR estimate',old_mean_squared_quality_credit=float(np.mean(before)),B1_mean_squared_quality_credit=float(np.mean(after)),ratio=float(np.mean(after)/np.mean(before)),episodes=1000))
if __name__=='__main__':main()
