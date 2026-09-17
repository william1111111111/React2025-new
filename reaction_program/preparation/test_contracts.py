import copy,unittest
import numpy as np
from .evaluation import load_manifest,load_processed_targets,validate_prediction
from .validate import reject_identifiers,validate_support

class ContractTests(unittest.TestCase):
    def test_frozen_ten_references(self):
        m=load_manifest();self.assertEqual(len(m['sources']),1142)
        self.assertTrue(all(len(s['targets'])==10 for s in m['sources']))
    def test_missing_cache_fails_closed(self):
        with self.assertRaises(FileNotFoundError):load_processed_targets(0,{'sources':[{'length':5,'processed_target':{'path':'/nonexistent/reaction-program-test.npy','sha256':'missing'}}]})
    def test_prediction_requires_native_k10_and_finite(self):
        m={'sources':[{'length':5}]}
        validate_prediction(np.zeros((10,5,25)),0,m)
        for p in [np.zeros((11,5,25)),np.zeros((10,4,25)),np.full((10,5,25),np.nan)]:
            with self.assertRaises(ValueError):validate_prediction(p,0,m)
    def test_teacher_identifier_leak_rejected(self):
        for p in [{'session_id':'s3'},{'media':'/data/train/listener/clip'},{'local_context':{'target':[1]}}]:
            with self.assertRaises(ValueError):reject_identifiers(p)
        reject_identifiers({'transcript':'What happened?','media_assets':['asset_123']})
    def test_pending_observation_cannot_be_observed_support(self):
        s={'event_id':'e','status':'supported','observed_support':[{'observation_id':'o','association_evidence':'a'}]}
        e={'e':{'split':'train','fold':'RM_fit'}}
        o={'o':{'status':'pending_observation','actions':None,'relation_status':'UNKNOWN','split':'train','fold':'RM_fit'}}
        with self.assertRaises(ValueError):validate_support(s,e,o)
    def test_cross_fold_support_rejected(self):
        s={'event_id':'e','status':'supported','observed_support':[{'observation_id':'o','association_evidence':'a'}]}
        e={'e':{'split':'train','fold':'RM_fit'}}
        o={'o':{'status':'observed','actions':[{'description':'motion'}],'relation_status':'temporal_support','split':'train','fold':'RM_audit'}}
        with self.assertRaises(ValueError):validate_support(s,e,o)
    def test_missing_support_not_positive(self):
        with self.assertRaises(ValueError):validate_support({'event_id':'e','status':'supported','observed_support':[]},{'e':{'split':'train','fold':'RM_fit'}},{})
if __name__=='__main__':unittest.main()
