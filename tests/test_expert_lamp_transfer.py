"""Contract tests for the expert-lamp transfer adapter (not yet executed)."""
import torch
from model.blocks import build_model, ParameterProbe
from model.losses import curve_loss, param_loss, smoothness_loss
from training.dataset import SyntheticCompressorDataset, collate

def test_dataset_determinism_and_pooling_grid():
    ds1=SyntheticCompressorDataset(n_samples=2,duration_s=0.25,seed=19,precompute=True)
    ds2=SyntheticCompressorDataset(n_samples=2,duration_s=0.25,seed=19,precompute=False)
    a,b=ds1[0],ds2[0]
    assert set(a)=={"x","y","g","params","idx"}
    assert torch.equal(a["x"],b["x"])
    assert torch.equal(a["y"],b["y"])
    assert torch.equal(a["g"],b["g"])
    assert torch.equal(a["params"],b["params"])
    assert a["x"].shape==(1,11025)
    assert a["y"].shape==a["x"].shape
    assert a["g"].shape==(1,44)
    assert a["params"].shape==(4,)
    assert int(a["idx"])==0

def test_collate_and_model_four_return_contract():
    ds=SyntheticCompressorDataset(n_samples=2,duration_s=0.25,seed=5)
    batch=collate([ds[0],ds[1]])
    model=build_model(n_bands=1)
    y,g,bands,extras=model(batch["x"],return_extras=True)
    probe=ParameterProbe(extras["latent"].shape[-1],4)
    p=probe(extras["latent"])
    assert y.shape==batch["y"].shape
    assert g.shape==batch["g"].shape
    assert bands.shape==(2,1,1,44)
    assert p.shape==batch["params"].shape
    assert "latent" in extras
    assert torch.isfinite(y).all() and torch.isfinite(g).all() and torch.isfinite(p).all()()

def test_loss_gradients_reach_model_and_probe():
    ds=SyntheticCompressorDataset(n_samples=2,duration_s=0.25,seed=11)
    batch=collate([ds[0],ds[1]])
    model=build_model(n_bands=1)
    _,g,_,extras=model(batch["x"],return_extras=True)
    probe=ParameterProbe(extras["latent"].shape[-1],4)
    p=probe(extras["latent"])
    loss=curve_loss(g,batch["g"])+param_loss(p,batch["params"])+smoothness_loss(g)
    loss.backward()
    model_grads=[q.grad for q in model.parameters() if q.grad is not None]
    probe_grads=[q.grad for q in probe.parameters() if q.grad is not None]
    assert model_grads and probe_grads
    assert all(torch.isfinite(q).all() for q in model_grads+probe_grads)
    assert any(torch.count_nonzero(q).item()>0 for q in model_grads)
    assert any(torch.count_nonzero(q).item()>0 for q in probe_grads)

def test_band_count_is_explicitly_single_band():
    try:
        build_model(n_bands=4)
    except ValueError as exc:
        assert "single-band" in str(exc)
    else:
        raise AssertionError("multi-band model was unexpectedly accepted")
