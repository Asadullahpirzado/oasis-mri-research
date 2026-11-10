"""Configuration loading + path management."""
import os, yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(os.path.dirname(_HERE), "configs", "default.yaml")


class Cfg(dict):
    """dict with attribute access (nested)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e
    def __setattr__(self, k, v):
        self[k] = v


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    return x


def load_config(path=None, overrides=None):
    """overrides: {'paths.data_dir': '/x', 'train.epochs': 5}"""
    with open(path or DEFAULT_CFG) as f:
        cfg = _wrap(yaml.safe_load(f))
    for k, v in (overrides or {}).items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = _wrap(v)
    return cfg


class Paths:
    """All input/output locations. Every stage folder is created on first use."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg.paths.data_dir
        self.out_dir = cfg.paths.out_dir
        self.local_dir = cfg.paths.local_dir
        for d in (self.out_dir, self.local_dir):
            os.makedirs(d, exist_ok=True)

    def out(self, *parts):
        p = os.path.join(self.out_dir, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def local(self, *parts):
        p = os.path.join(self.local_dir, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    @property
    def audit(self): return self.out("step1_audit")
    @property
    def cache_drive(self): return self.out("step2_cache")
    @property
    def cache_local(self): return self.local("cache")
    @property
    def extracted(self): return self.local("extracted")
    @property
    def splits(self): return self.out("step3_splits")
    @property
    def baselines(self): return self.out("step4_baselines")
    def models(self, name): return self.out("step5_models", name)
    @property
    def evaluation(self): return self.out("step6_evaluation")
    @property
    def ood(self): return self.out("step7_ood")
    @property
    def selective(self): return self.out("step8_selective_saliency")
    @property
    def bundle(self): return self.out("step9_bundle")
