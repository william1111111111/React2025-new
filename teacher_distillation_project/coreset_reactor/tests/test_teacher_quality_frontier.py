import torch
from coreset_reactor.teacher_quality_frontier import combine, recipe_grid
from coreset_reactor.evaluate import official_prediction


def test_fixed_recipes_preserve_anchor_and_inputs():
    torch.manual_seed(13)
    a,b=torch.rand(10,12,25),torch.rand(10,12,25)
    ac,bc=a.clone(),b.clone()
    for recipe in recipe_grid():
        out=combine(a,b,recipe)
        assert torch.equal(out[0],official_prediction(a)[0])
        assert out.shape==a.shape and torch.isfinite(out).all()
        assert torch.equal(out[...,:15],out[...,:15].round())
        if recipe['kind']=='replace':
            rank=recipe['rank']
            assert torch.equal(out[rank],official_prediction(b)[rank])
            for i in range(10):
                if i!=rank: assert torch.equal(out[i],official_prediction(a)[i])
    assert torch.equal(a,ac) and torch.equal(b,bc)


def test_mix_then_round_is_not_mixing_rounded_au():
    a=torch.full((10,3,25),.51); b=torch.full_like(a,.01)
    out=combine(a,b,dict(kind='blend',alpha=.1))
    assert torch.equal(out[1:,:,:15],torch.zeros_like(out[1:,:,:15]))
    assert torch.equal(out[0,:,:15],torch.ones_like(out[0,:,:15]))
