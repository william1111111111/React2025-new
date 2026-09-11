import torch,itertools
from mam_refine.losses import one_to_one,objective
from mam_target.losses import objective as original

def test_exact_assignment_and_gradients():
    torch.manual_seed(123);c=torch.rand(3,4,4,dtype=torch.double,requires_grad=True)
    expected=torch.stack([torch.stack([x[torch.arange(4),torch.tensor(p)].mean() for p in itertools.permutations(range(4))]).min() for x in c]).mean()
    torch.testing.assert_close(one_to_one(c),expected,atol=0,rtol=0)
    g=torch.autograd.grad(one_to_one(c),c)[0]
    torch.testing.assert_close(g.sum(2),torch.full((3,4),1/12,dtype=torch.double));torch.testing.assert_close(g.sum(1),torch.full((3,4),1/12,dtype=torch.double))
    assert torch.autograd.gradcheck(one_to_one,(c,),atol=1e-5)

def test_duplicates_and_permutation():
    c=torch.tensor([[[1.,2.,1.,2.]]*4],requires_grad=True)
    assert one_to_one(c).item()==1.5
    g=torch.autograd.grad(one_to_one(c),c)[0];assert (g.sum(1)==.25).all() and (g.sum(2)==.25).all()
    torch.manual_seed(4);c=torch.rand(2,4,4)
    torch.testing.assert_close(one_to_one(c),one_to_one(c[:,[3,0,2,1]][:,:,[2,0,3,1]]))

def test_control_exact_objective_and_gradient():
    torch.manual_seed(6);p=torch.rand(1,4,8,25,requires_grad=True);y=torch.rand(1,4,8,25);n=torch.tensor([[6,7,7,7]])
    b=dict(source_lengths=torch.tensor([7]),pair_lengths=torch.tensor([6]),paired_target=y[:,0]);w=dict(a=1,b=.15813484622234084,beta=.25,gamma=.25,eta=.5,grid=4)
    a=original(p,b,y,n,w);z=objective(p,b,y,n,w,'softmin')
    for k in a:torch.testing.assert_close(a[k],z[k],atol=0,rtol=0)
    ga=torch.autograd.grad(a['loss'],p,retain_graph=True)[0];gz=torch.autograd.grad(z['loss'],p,retain_graph=True)[0];torch.testing.assert_close(ga,gz,atol=0,rtol=0)
    q=objective(p,b,y,n,w,'one_to_one')
    for k in ('valid','paired','preserve','ccc_cost','sdtw_cost'):torch.testing.assert_close(a[k],q[k],atol=0,rtol=0)
    g=torch.autograd.grad(q['loss'],p)[0];assert g[:,:,7:].abs().sum()==0 and (g.abs().sum((2,3))>0).all()

def test_short_reference_pool_retains_slots():
    from types import SimpleNamespace
    from mam_target.data import TaskData
    d=TaskData.__new__(TaskData);d.device='cpu'
    raw=torch.rand(8,25)
    d.pool=SimpleNamespace(references={'s':['listener/s/a','listener/s/b']},reference_paths={'listener/s/b':'b'},dataset=SimpleNamespace(_load=lambda _:raw))
    batch=dict(clip_id=['speaker/s/a'],paired_target=raw[None],pair_lengths=torch.tensor([8]),crop_start=torch.tensor([0]),source_total_length=torch.tensor([8]),source_lengths=torch.tensor([8]))
    d.batch=lambda rec:(batch,None)
    a=d.task_batch(dict(session_id='s',step=6001));b=d.task_batch(dict(session_id='s',step=6001))
    assert a[4]==[['listener/s/a','listener/s/b','listener/s/b','listener/s/b']]
    torch.testing.assert_close(a[2],b[2]);assert a[2].shape==(1,4,8,25)
