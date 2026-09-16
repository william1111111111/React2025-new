import unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from reference_contracts import (
    PreferenceEvidence, preference_loss, pareto_preferences,
    contiguous_shift_pair, anchored_score,
)

class Contracts(unittest.TestCase):
    def test_unknown_has_zero_gradient(self):
        a = torch.tensor([0.2, 9.0], requires_grad=True)
        b = torch.tensor([-0.1, -9.0], requires_grad=True)
        loss = preference_loss(a,b,torch.tensor([1.,float('nan')]),
                               torch.tensor([True,False]),torch.ones(2))
        loss.backward()
        self.assertEqual(a.grad[1].item(), 0.)
        self.assertEqual(b.grad[1].item(), 0.)
        self.assertLess(a.grad[0].item(), 0.)

    def test_swap_symmetry(self):
        a,b = torch.tensor([.2,.4]),torch.tensor([-.3,.6])
        p = torch.tensor([1.,0.]); k=torch.ones(2,dtype=torch.bool); w=torch.ones(2)
        torch.testing.assert_close(preference_loss(a,b,p,k,w),preference_loss(b,a,1-p,k,w))

    def test_conflicting_metrics_abstain(self):
        p=pareto_preferences([.20,.15],[9.,8.],.01,.2)
        self.assertTrue(np.isnan(p).all())

    def test_pareto_orientation(self):
        p=pareto_preferences([.20,.15],[7.,8.],.01,.2)
        self.assertEqual(p[0,1],1.); self.assertEqual(p[1,0],0.)
        self.assertTrue(np.isnan(np.diag(p)).all())

    def test_contiguous_no_wrap(self):
        y=np.arange(30)[:,None]
        a,b=contiguous_shift_pair(y,6,8,3)
        np.testing.assert_array_equal(a[:,0],np.arange(6,14))
        np.testing.assert_array_equal(b[:,0],np.arange(9,17))
        with self.assertRaises(ValueError): contiguous_shift_pair(y,0,8,-3)

    def test_donor_leak_rejected(self):
        row=PreferenceEvidence('RM_fit','source',('paired',),('donor',),'cross','context',1.,.5)
        with self.assertRaises(ValueError):
            row.validate({'source':'RM_fit','paired':'RM_fit','donor':'RM_audit'})

    def test_weak_time_and_fake_tie_rejected(self):
        assignments={'source':'RM_fit','a':'RM_fit','b':'RM_fit'}
        with self.assertRaises(ValueError):
            PreferenceEvidence('RM_fit','source',('a',),('b',),'weak_same_session','time',1.,.25).validate(assignments)
        with self.assertRaises(ValueError):
            PreferenceEvidence('RM_fit','source',('a',),('b',),'metric_disagreement','context',.5,1.).validate(assignments)
        PreferenceEvidence('RM_fit','source',('a',),('b',),'metric_disagreement','context',None,0.).validate(assignments)

    def test_reference_cancels_source_offset(self):
        s=np.array([.1,.8]);r=np.array([.3,.4,.5])
        np.testing.assert_allclose(anchored_score(s,r,1.2),anchored_score(s+12,r+12,1.2),atol=1e-13)

    def test_2x2_cancels_additive_shortcut(self):
        source=np.array([1.,4.]); listener=np.array([-2.,3.])
        scores=source[:,None]+listener[None,:]
        interaction=scores[0,0]+scores[1,1]-scores[0,1]-scores[1,0]
        self.assertEqual(interaction,0.)

    def test_budget_and_stage_contract(self):
        cfg=yaml.safe_load((Path(__file__).parent/'experiment.yaml').read_text())
        b=cfg['candidate_bank'];ws=b['max_windows'];k=b['candidate_count']
        budget=ws['RM_fit']*2*k+ws['RM_cal']*2*k+ws['RM_audit']*3*k
        self.assertEqual(budget,b['candidate_window_budget_max'])
        self.assertFalse(cfg['stage_B']['enabled'])
        self.assertFalse(cfg['inputs']['learned_P2_encoder_weights'])
        self.assertEqual(sum(cfg['training']['family_slots'].values()),cfg['training']['batch_records'])

if __name__=='__main__': unittest.main(verbosity=2)
