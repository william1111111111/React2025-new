import unittest
import numpy as np
from .selectors import select

class SelectorTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        self.y=rng.normal(size=(16,20,25));self.a=rng.normal(size=(16,4));self.b=rng.normal(size=(16,4))
    def test_determinism_unique_k(self):
        x=select(self.y,self.a,self.b,[0,0],[1,1])
        self.assertEqual(x,select(self.y,self.a,self.b,[0,0],[1,1]))
        self.assertTrue(all(len(r['selected'])==len(set(r['selected']))==10 for r in x.values()))
    def test_gate_empty_and_full(self):
        self.b[:,2]=-1;self.b[:,3]=1
        self.assertEqual(select(self.y,self.a,self.b,[0,0],[1,1])['vector-train']['fallback_slots'],10)
        self.b[:,2]=1;self.b[:,3]=-1
        self.assertFalse(select(self.y,self.a,self.b,[0,0],[1,1])['vector-train']['fallback'])
    def test_both_quality_constraints_required(self):
        self.b[:,2]=1;self.b[:,3]=1
        self.assertEqual(select(self.y,self.a,self.b,[0,0],[1,1])['vector-train']['gate_accepted'],[])
    def test_diversity_definition(self):
        f=self.y[:10].reshape(10,-1)
        self.assertAlmostEqual(float(((f[:,None]-f[None,:])**2).mean(-1).sum()/90),float(2*10/9*np.var(f,axis=0).mean()))
    def test_selector_has_no_target_or_io_interface(self):
        import inspect,ast
        tree=ast.parse(inspect.getsource(select))
        forbidden={'open','read','load','target','eval','exec'}
        self.assertFalse(any(isinstance(n,ast.Name) and n.id in forbidden for n in ast.walk(tree)))
        self.assertEqual(list(inspect.signature(select).parameters),['y','raw_a','raw_b','mean','std'])
if __name__=='__main__':unittest.main()
