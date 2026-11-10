"""Training one fold + the repeated participant-grouped CV loop.

Resumable: a `last.pt` checkpoint is written (atomically) after every epoch, `best.pt` whenever validation
NLL improves, and `fold_done.json` only after the whole fold (incl. calibration outputs) is finished.
After a Colab disconnect simply re-run the cell - finished folds are skipped, an unfinished fold resumes
from its last epoch.
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import set_seed, save_json, load_json, log
from .data.dataset import VolumeStore, VolDataset
from .data.splits import get_ids
from .models.zoo import build_model, count_params
from .uncertainty import fit_temperature, softmax, entropy


# ───────────────────────── inference helpers ─────────────────────────
@torch.no_grad()
def predict_logits(model, store, ids, device, bs=4):
    model.eval()
    ds = VolDataset(store, ids, None)
    out = []
    for x, _ in DataLoader(ds, batch_size=bs, shuffle=False):
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2))


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_logits(model, store, ids, device, passes, bs=4):
    """(passes, N, C) logits with dropout active, everything else (norm layers) in eval mode."""
    enable_mc_dropout(model)
    ds = VolDataset(store, ids, None)
    res = []
    for _ in range(passes):
        out = [model(x.to(device)).float().cpu() for x, _ in DataLoader(ds, batch_size=bs, shuffle=False)]
        res.append(torch.cat(out).numpy())
    model.eval()
    return np.stack(res)


# ───────────────────────── one fold ─────────────────────────
def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_model(model, store, labels, fit_ids, val_ids, ckpt_dir, cfg, device, seed):
    """Early stopping on validation NLL. Returns dict with training summary. Leaves `best.pt` in ckpt_dir."""
    tc = cfg.train
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)
    model.to(device)
    y_fit = np.array([labels[p] for p in fit_ids])
    n_cls = model.head[-1].out_features
    cnt = np.bincount(y_fit, minlength=n_cls).astype(float)
    w = torch.tensor(len(y_fit) / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32) if tc.class_weighting else None
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device) if w is not None else None)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.epochs)
    use_amp = bool(tc.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    last_p, best_p = os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt")
    state = {"epoch": 0, "best": np.inf, "bad": 0, "history": []}
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(f"    resumed from epoch {state['epoch']}")

    ds = VolDataset(store, fit_ids, labels, aug_cfg=tc.augment, seed=seed)
    loader = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, drop_last=len(ds) > tc.batch_size,
                        generator=torch.Generator().manual_seed(seed))
    y_val = np.array([labels[p] for p in val_ids])
    t0 = time.time()
    while state["epoch"] < tc.epochs and state["bad"] < tc.patience:
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y) / tc.accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % tc.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item() * tc.accum_steps)
        sched.step()
        z = predict_logits(model, store, val_ids, device)
        p = softmax(z)
        v_nll = float(-np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean())
        v_acc = float((p.argmax(1) == y_val).mean())
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": float(np.mean(losses)), "val_nll": v_nll, "val_acc": v_acc})
        if v_nll < state["best"] - 1e-4:
            state["best"], state["bad"] = v_nll, 0
            _atomic_save(model.state_dict(), best_p)
        else:
            state["bad"] += 1
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state}, last_p)
        if state["epoch"] % 10 == 0 or state["epoch"] == 1:
            log(f"    epoch {state['epoch']:3d} loss={np.mean(losses):.4f} val_nll={v_nll:.4f} val_acc={v_acc:.3f}")
    if not os.path.exists(best_p):                       # degenerate case: never improved
        _atomic_save(model.state_dict(), best_p)
    model.load_state_dict(torch.load(best_p, map_location=device))
    pd.DataFrame(state["history"]).to_csv(os.path.join(ckpt_dir, "history.csv"), index=False)
    return {"epochs_run": state["epoch"], "best_val_nll": state["best"], "train_seconds": time.time() - t0,
            "n_params": count_params(model)}


def _fold_outputs(model, store, labels, ids_by_role, T, cfg, device):
    """Deterministic + MC-dropout outputs for val / calib / test participants of one fold."""
    rows = []
    for role in ("val", "calib", "test"):
        ids = ids_by_role[role]
        if not ids:
            continue
        z = predict_logits(model, store, ids, device)
        pc = softmax(z, T)
        mcz = mc_logits(model, store, ids, device, cfg.train.mc_passes)
        mcp = softmax(mcz.reshape(-1, mcz.shape[-1]), T).reshape(mcz.shape)       # (P,N,C) calibrated
        mean_p = mcp.mean(0)
        ent = entropy(mean_p)
        mi = ent - entropy(mcp.reshape(-1, mcp.shape[-1])).reshape(mcp.shape[:2]).mean(0)   # mutual information
        for i, pid in enumerate(ids):
            d = {"participant_id": pid, "role": role, "label": int(labels[pid]),
                 "mc_entropy": float(ent[i]), "mc_mutual_info": float(mi[i])}
            for c in range(z.shape[1]):
                d[f"logit_{c}"] = float(z[i, c]); d[f"p_cal_{c}"] = float(pc[i, c]); d[f"p_mc_{c}"] = float(mean_p[i, c])
            rows.append(d)
    return pd.DataFrame(rows)


def run_cv(model_name, cfg, paths, labeled, splits, store, device, repeats=None, folds=None):
    """Train + calibrate for every (repeat, fold). Writes into step5_models/<model_name>/r{R}_f{F}/ ."""
    labels = dict(zip(labeled["participant_id"], labeled["label"].astype(int)))
    n_cls = 2 if cfg.task.mode == "binary" else 3
    out_root = paths.models(model_name)
    R = range(cfg.cv.n_repeats) if repeats is None else repeats
    F = range(cfg.cv.n_splits) if folds is None else folds
    for r in R:
        for f in F:
            d = os.path.join(out_root, f"r{r}_f{f}")
            done = os.path.join(d, "fold_done.json")
            if os.path.exists(done):
                log(f"[{model_name} r{r} f{f}] already done - skipped"); continue
            ids = {k: get_ids(splits, r, f, k) for k in ("fit", "val", "calib", "test")}
            log(f"[{model_name} r{r} f{f}] fit={len(ids['fit'])} val={len(ids['val'])} calib={len(ids['calib'])} test={len(ids['test'])}")
            seed = cfg.cv.seed + 1000 * r + f
            set_seed(seed)
            model = build_model(model_name, cfg, n_cls, paths)
            info = train_model(model, store, labels, ids["fit"], ids["val"], d, cfg, device, seed)
            zv = predict_logits(model, store, ids["val"], device)
            T = fit_temperature(zv, [labels[p] for p in ids["val"]])
            out = _fold_outputs(model, store, labels, ids, T, cfg, device)
            out.insert(0, "repeat", r); out.insert(1, "fold", f)
            out.to_csv(os.path.join(d, "fold_outputs.csv"), index=False)
            save_json({**info, "temperature": T, "repeat": r, "fold": f, "model": model_name,
                       "n_fit": len(ids["fit"]), "n_val": len(ids["val"]), "n_calib": len(ids["calib"]), "n_test": len(ids["test"])}, done)
            log(f"[{model_name} r{r} f{f}] done: T={T:.3f} best_val_nll={info['best_val_nll']:.4f} ({info['train_seconds']/60:.1f} min)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out_root


def load_model_for_fold(model_name, cfg, paths, repeat, fold, device):
    n_cls = 2 if cfg.task.mode == "binary" else 3
    m = build_model(model_name, cfg, n_cls, paths)
    p = os.path.join(paths.models(model_name), f"r{repeat}_f{fold}", "best.pt")
    m.load_state_dict(torch.load(p, map_location=device))
    return m.to(device).eval()
