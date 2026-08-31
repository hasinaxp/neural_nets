import pytest

from nanollm.config import ModelConfig, TrainConfig, apply_overrides, load_config


def test_reference_model_is_under_200m():
    cfg = ModelConfig()
    assert cfg.estimate_params() < 200_000_000
    assert cfg.estimate_params() > 150_000_000


def test_head_dim_hits_the_fast_kernels():
    # 32/64/128 are the sizes flash attention is tuned for; anything else
    # silently falls back to the math kernel.
    assert ModelConfig().head_dim in (32, 64, 128)


def test_estimate_matches_built_model():
    from nanollm.model import Transformer
    cfg = ModelConfig(vocab_size=256, n_dim=64, n_layer=2, n_head=2, n_kv_head=1)
    assert Transformer.from_config(cfg).get_param_count() == cfg.estimate_params()


def test_validate_rejects_bad_geometry():
    with pytest.raises(ValueError):
        ModelConfig(n_dim=100, n_head=7).validate()
    with pytest.raises(ValueError):
        ModelConfig(n_dim=896, n_head=14, n_kv_head=3).validate()


def test_overrides_are_typed():
    cfg = TrainConfig()
    apply_overrides(cfg, ["optim.peak_lr=3e-4", "model.n_layer=6",
                          "runtime.compile=false"])
    assert cfg.optim.peak_lr == 3e-4 and isinstance(cfg.optim.peak_lr, float)
    assert cfg.model.n_layer == 6 and isinstance(cfg.model.n_layer, int)
    assert cfg.runtime.compile is False


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        apply_overrides(TrainConfig(), ["optim.nonexistent=1"])


def test_roundtrip_through_dict():
    cfg = TrainConfig()
    cfg.optim.peak_lr = 1.23e-4
    assert TrainConfig.from_dict(cfg.to_dict()).optim.peak_lr == 1.23e-4


def test_shipped_configs_load():
    import os
    for name in ("configs/base.yaml", "configs/debug.yaml"):
        if os.path.exists(name):
            load_config(name, []).validate()
