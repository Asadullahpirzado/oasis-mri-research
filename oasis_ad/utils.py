"""Small shared helpers."""
import os, json, random, time
import numpy as np


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (set, tuple)): return list(o)
        return str(o)


def _clean(o):
    """make dict keys and values JSON-safe (numpy ints / floats -> builtin int/float/str)."""
    if isinstance(o, dict):
        clean_dict = {}
        for k, v in o.items():
            if isinstance(k, np.integer):
                ck = int(k)
            elif isinstance(k, np.floating):
                ck = float(k)
            elif isinstance(k, np.bool_):
                ck = bool(k)
            elif isinstance(k, (str, int, float, bool)) or k is None:
                ck = k
            else:
                ck = str(k)
            clean_dict[ck] = _clean(v)
        return clean_dict
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=2, cls=_Enc)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def participant_id(session_id):
    """OAS1_0001_MR1 -> OAS1_0001"""
    s = str(session_id).strip().upper()
    return s.rsplit("_MR", 1)[0] if "_MR" in s else s


def age_band(age, bands):
    for lo, hi in bands:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "other"
