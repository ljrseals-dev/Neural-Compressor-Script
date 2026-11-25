import math
import torch
import pytest

from model.blocks import ParameterProbe, build_model, rms_envelope
from model.losses import CompressorLoss, curve_loss
from training.dataset import SyntheticCompressorDataset, collate


def test_model_contract_and_gradients():
    torch.manual_seed(7)
    model = build_model(n_bands=1)
    probe = ParameterProbe(model.stem.out_channels, 4)
    x = torch.randn(2, 1, 4096) * 0.1
    assert rms_envelope(x).shape == x.shape
    y, g, bands, extras = model(x, return_extras=True)
    p = probe(extras["latent"])
    assert y.shape == x.shape
    assert g.shape == (2, 1, math.ceil(4096 / 256))
    assert bands.shape == (2, 1, 1, math.ceil(4096 / 256))
    assert p.shape == (2, 4)
    assert torch.isfinite(y).all() and torch.isfinite(g).all()
    target_y = x * 0.8
    target_g = torch.full_like(g, -1.0)
    target_p = torch.full_like(p, 0.5)
    loss, logs = CompressorLoss()(y, target_y, g, target_g, p, target_p)
    assert torch.isfinite(loss)
    assert {"total", "curve", "stft", "param", "smooth"} <= set(logs)
    loss.backward()
    for name, parameter in list(model.named_parameters()) + [
        ("probe." + n, p_) for n, p_ in probe.named_parameters()
    ]:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("sample_rate,duration", [(44100, 0.25), (48000, 0.5), (32000, 0.4)])
def test_dataset_teacher_student_grid_and_repeatability(sample_rate, duration):
    a = SyntheticCompressorDataset(3, duration, sample_rate, seed=31, precompute=True)
    b = SyntheticCompressorDataset(3, duration, sample_rate, seed=31, precompute=False)
    batch = collate([a[0], a[1]])
    assert torch.equal(a[0]["x"], b[0]["x"])
    assert torch.equal(a[0]["g"], b[0]["g"])
    expected = math.ceil(round(duration * sample_rate) / 256)
    assert batch["g"].shape == (2, 1, expected)
    model = build_model()
    _, predicted_gain, _, _ = model(batch["x"])
    assert predicted_gain.shape == batch["g"].shape
    assert batch["sample_id"].shape == (2, 2)


def test_curve_loss_rejects_temporal_mismatch():
    with pytest.raises(ValueError, match="gain curve shape mismatch"):
        curve_loss(torch.zeros(1, 1, 10), torch.zeros(1, 1, 9))


def test_toy_fit_reduces_curve_objective():
    prediction = torch.nn.Parameter(torch.zeros(2, 1, 16))
    target = torch.full_like(prediction, -1.0)
    optimizer = torch.optim.Adam([prediction], lr=0.02)
    initial = float(curve_loss(prediction, target))
    for _ in range(50):
        optimizer.zero_grad()
        loss = curve_loss(prediction, target)
        loss.backward()
        optimizer.step()
    assert float(curve_loss(prediction, target)) < initial * 0.2
