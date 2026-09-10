import json
from copy import deepcopy
from pathlib import Path
import pytest
from hirp.phase24 import dependencies,guarded,official_selection,validate_candidate
from hirp.phase23_cache import cached,save_cached
from hirp.phase15_audit import sha256_file
import random


def test_unlisted_generator_and_population_dependency_changes(tmp_path):
    root=tmp_path/'hirp';(root/'model').mkdir(parents=True)
    import shutil
    actual=Path(__file__).resolve().parents[1]
    for name in ('model/temporal_decoder.py','session_data.py'):shutil.copyfile(actual/name,root/name)
    identity=dependencies(root);cache=tmp_path/'cache.json';save_cached(cache,identity,{'score':1})
    for name in ('model/temporal_decoder.py','session_data.py'):
        p=root/name;old=p.read_text();p.write_text(old+'\n# temporary dependency mutation\n');changed=dependencies(root)
        assert changed!=identity
        with pytest.raises(ValueError):cached(cache,changed)
        p.write_text(old)
    assert cached(cache,dependencies(root))=={'score':1}


def test_normalization_blocks_generation_and_cache(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);directory=Path('external/FaceVerse');directory.mkdir(parents=True)
    for name in ('mean_face.npy','std_face.npy'):(directory/name).write_bytes(b'frozen')
    plan={'normalization':[dict(path=str(p),sha256=sha256_file(p)) for p in directory.iterdir()]}
    called=[]
    cache=tmp_path/'result.json';save_cached(cache,{'version':'frozen'},{'score':42})
    assert guarded(plan,lambda:cached(cache,{'version':'frozen'}))=={'score':42}
    (directory/'std_face.npy').write_bytes(b'changed')
    for operation in ('generate','read_export','read_result'):
        with pytest.raises(ValueError,match='content mismatch'):guarded(plan,lambda:called.append(operation))
    assert not called
    with pytest.raises(ValueError,match='content mismatch'):guarded(plan,lambda:cached(cache,{'version':'frozen'}))


def test_official_target_count_order_short_pool():
    assert official_selection('p',['p'],random.Random(3))==['p']*10
    x=official_selection('p',['p','a','b'],random.Random(3));assert len(x)==10 and x[0]=='p' and set(x[1:])<= {'a','b'}
    x=official_selection('p',['p']+list(range(12)),random.Random(3));assert len(set(x))==10


def test_lambda_label_not_just_path():
    m=dict(arm='C1',train_config=dict(init_seed=123,lambda_group=.03),global_step=2000,prior_mode='standard_normal',output_config={},config={},scales={})
    p=dict(arm='C1',seed=123,lambda_group=.03,stored_lambda_group=.03,steps=2000,**{k:m[k] for k in ('prior_mode','output_config','config','scales')})
    validate_candidate(p,m)
    for key,value in [('arm','C2'),('seed',42),('steps',128),('lambda_group',.1),('stored_lambda_group',.1)]:
        changed={**p,key:value}
        with pytest.raises(ValueError):validate_candidate(changed,m)


def test_selection_matches_literal_official_ast():
    import ast
    from types import SimpleNamespace
    source=Path('/home/zhengshiyi/react2025/dataset/react_2025.py')
    if not source.exists():pytest.skip('locked local official loader unavailable')
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReactionDataset')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    branch=next(n for n in method.body if isinstance(n,ast.If) and ast.unparse(n.test)=="self._split == 'test'")
    code=compile(ast.fix_missing_locations(ast.Module(body=branch.body,type_ignores=[])),'official-selection','exec')
    for pool in [['p'],['p','a','b'],['p']+list(range(15))]:
        scope=dict(self=SimpleNamespace(gt_path_list=[pool]),index=0,listener_path='p',total_length=750,random=random.Random(24100))
        exec(code,scope)
        assert scope['listener_paths']==official_selection('p',pool,random.Random(24100))
