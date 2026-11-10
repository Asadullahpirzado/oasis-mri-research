"""Deployment bundle + Predictor (the same code path is used by the notebooks and the web backend)."""
import os, io, base64, shutil, json
import numpy as np
import pandas as pd
import torch

from . import DISCLAIMER
from .utils import save_json, load_json
from .data.preprocess import preprocess_volume, raw_qc_metrics, qc_flags
from .data.audit import LABEL_NAMES
from .models.zoo import build_model
from .ood import Mahalanobis, ScoreNormalizer
from .train import enable_mc_dropout
from .uncertainty import softmax, entropy, conformal_qhat
from . import evaluation as E
from . import saliency as S


# ───────────────────────── export ─────────────────────────
def export_bundle(model_name, cfg, paths, labeled, qc_stats, repeat=None):
    """Collect the 5 fold-models of one CV repeat (+ calibration + OOD stats) into a self-contained folder."""
    repeat = cfg.app.bundle_repeat if repeat is None else repeat
    bdir = paths.bundle
    os.makedirs(os.path.join(bdir, "models"), exist_ok=True)
    folds = []
    mdir = paths.models(model_name)
    odir = os.path.join(paths.ood, model_name)
    for f in range(cfg.cv.n_splits):
        d = os.path.join(mdir, f"r{repeat}_f{f}")
        meta = load_json(os.path.join(d, "fold_done.json"))
        shutil.copy2(os.path.join(d, "best.pt"), os.path.join(bdir, "models", f"f{f}.pt"))
        entry = {"fold": f, "temperature": meta["temperature"], "ckpt": f"models/f{f}.pt"}
        mz, nj = os.path.join(odir, f"r{repeat}_f{f}_maha.npz"), os.path.join(odir, f"r{repeat}_f{f}_norms.json")
        if os.path.exists(mz) and os.path.exists(nj):
            shutil.copy2(mz, os.path.join(bdir, "models", f"f{f}_maha.npz"))
            entry["maha"] = f"models/f{f}_maha.npz"; entry["norms"] = load_json(nj)
        folds.append(entry)

    # Use the held-out inner calibration participants for deployment thresholds.
    # Outer-test predictions are reserved for unbiased performance evaluation.
    df = E.load_fold_outputs(mdir)
    df = df[(df["repeat"] == repeat)]
    calib = df[df["role"] == "calib"]
    cols = E.prob_cols(calib)
    if calib.empty or not cols:
        raise ValueError(f"No calibration predictions available for repeat {repeat} in {mdir}")
    alpha = 1 - cfg.uncertainty.target_coverage
    qhat = conformal_qhat(calib[cols].values, calib["label"].values.astype(int), alpha)
    conf_tau = float(np.quantile(calib[cols].values.max(1), 1 - cfg.uncertainty.selective_coverage, method="higher"))

    # age+sex baseline (shown next to the MRI model so users can see whether the image adds anything)
    from .baselines import make_estimator
    lab = labeled.dropna(subset=["age", "sex_num"])
    est = make_estimator("logreg", 0).fit(lab[["age", "sex_num"]].values, lab["label"].astype(int).values)
    import joblib
    joblib.dump(est, os.path.join(bdir, "baseline_age_sex.joblib"))

    n_cls = 2 if cfg.task.mode == "binary" else 3
    bundle = {"model_name": model_name, "task_mode": cfg.task.mode, "n_classes": n_cls,
              "class_names": [LABEL_NAMES[cfg.task.mode][i] for i in range(n_cls)],
              "target_shape": list(cfg.preprocess.target_shape), "expected_shape": list(cfg.data.expected_shape),
              "folds": folds, "qhat": qhat, "alpha": alpha, "conf_tau": conf_tau,
              "qc_cfg": {"brain_frac_range": list(cfg.qc.brain_frac_range), "robust_z_limit": cfg.qc.robust_z_limit},
              "qc_stats": qc_stats, "mc_passes": 20, "disclaimer": DISCLAIMER,
              "cfg_models": json.loads(json.dumps(cfg.models)),
              "conformal_note": "q_hat and rejection threshold use only inner calibration participants from repeat %d. The exported fold ensemble approximates this calibration." % repeat}
    save_json(bundle, os.path.join(bdir, "bundle.json"))
    return bundle


# ───────────────────────── predictor ─────────────────────────
class _CfgShim(dict):
    __getattr__ = dict.get


class Predictor:
    def __init__(self, bundle_dir, device=None):
        self.dir = bundle_dir
        self.b = load_json(os.path.join(bundle_dir, "bundle.json"))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models, self.mahas, self.norms, self.temps = [], [], [], []
        cm = self.b["cfg_models"]
        shim = _CfgShim(models=_wrap(cm), paths=_wrap({"medicalnet_weights": None, "foundation_encoder": None}))
        for fd in self.b["folds"]:
            m = build_model(self.b["model_name"], shim, self.b["n_classes"])
            m.load_state_dict(torch.load(os.path.join(bundle_dir, fd["ckpt"]), map_location="cpu"))
            self.models.append(m.to(self.device).eval())
            self.temps.append(fd["temperature"])
            if "maha" in fd:
                self.mahas.append(Mahalanobis.from_dict(dict(np.load(os.path.join(bundle_dir, fd["maha"])))))
                self.norms.append({k: ScoreNormalizer.from_dict(v) for k, v in fd["norms"].items()})
            else:                                   # OOD statistics missing for this fold-model -> skipped in OOD scoring
                self.mahas.append(None); self.norms.append(None)
        bp = os.path.join(bundle_dir, "baseline_age_sex.joblib")
        import joblib
        self.baseline = joblib.load(bp) if os.path.exists(bp) else None

    # -- helpers
    def _logits(self, m, x, mc=False):
        with torch.no_grad():
            if mc:
                enable_mc_dropout(m)
            else:
                m.eval()
            z = m(x).float().cpu().numpy()
            m.eval()
        return z

    def predict_raw(self, raw_vol, age=None, sex=None, with_cam=True):
        qc = raw_qc_metrics(raw_vol, self.b["expected_shape"])
        vol, mask = preprocess_volume(raw_vol, self.b["target_shape"])
        return self.predict_preprocessed(vol, mask, qc, age, sex, with_cam)

    def predict_preprocessed(self, vol, mask, qc=None, age=None, sex=None, with_cam=True):
        b = self.b
        qc = qc or {"shape_ok": True, "finite_frac": 1.0, "brain_frac": float(mask.mean())}
        class _Q:
            brain_frac_range = b["qc_cfg"]["brain_frac_range"]; robust_z_limit = b["qc_cfg"]["robust_z_limit"]
        hard, soft, why = qc_flags(qc, _Q, b["qc_stats"])
        x = torch.from_numpy(vol.astype(np.float32))[None, None].to(self.device)
        P, MC, maha_z, maha_thr, ent_z = [], [], [], [], []
        for i, m in enumerate(self.models):
            T = self.temps[i]
            P.append(softmax(self._logits(m, x), T)[0])
            mcz = np.stack([self._logits(m, x, mc=True)[0] for _ in range(b["mc_passes"])])
            MC.append(softmax(mcz, T))
            if self.mahas[i] is not None:
                with torch.no_grad():
                    m.eval(); emb = m.embed(x).float().cpu().numpy()
                s = self.mahas[i].score(emb)[0]
                nz = self.norms[i]["mahalanobis"]
                maha_z.append(float(nz.z(s))); maha_thr.append(float(nz.z(nz.thr)))
        P = np.array(P); ens = P.mean(0)
        mc_all = np.concatenate(MC); mc_mean = mc_all.mean(0)
        sets = np.where(1 - ens <= b["qhat"])[0].tolist()
        conf = float(ens.max()); pred = int(ens.argmax())
        ood_flag = bool(maha_z and (np.mean(maha_z) > np.mean(maha_thr)))
        reasons = []
        if hard: reasons.append("QC failure: " + why)
        if soft: reasons.append("QC warning: " + why)
        if ood_flag: reasons.append("scan looks unusual compared with the training data (feature-space distance above threshold)")
        if conf < b["conf_tau"]: reasons.append(f"confidence {conf:.2f} below rejection threshold {b['conf_tau']:.2f}")
        if len(sets) != 1: reasons.append(f"conformal set contains {len(sets)} classes ({', '.join(b['class_names'][k] for k in sets) or 'none'})")
        abstain = hard or ood_flag or conf < b["conf_tau"] or len(sets) != 1
        out = {"decision": "ABSTAIN - refer for expert review" if abstain else f"Predicted group: {b['class_names'][pred]}",
               "abstain": bool(abstain), "reasons": reasons, "predicted_class": pred if not abstain else None,
               "class_names": b["class_names"], "probabilities": {b["class_names"][k]: float(ens[k]) for k in range(len(ens))},
               "confidence": conf, "model_disagreement_sd": float(P.std(0).max()),
               "mc_dropout_entropy": float(entropy(mc_mean[None])[0]),
               "conformal_set": [b["class_names"][k] for k in sets], "conformal_target_coverage": 1 - b["alpha"],
               "ood": {"checked": bool(maha_z), "flag": ood_flag,
                       "mahalanobis_z_mean": float(np.mean(maha_z)) if maha_z else None,
                       "flag_threshold_z_mean": float(np.mean(maha_thr)) if maha_thr else None},
               "qc": {"hard_fail": bool(hard), "soft_flag": bool(soft), "detail": why or "ok",
                      "metrics": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in qc.items()}},
               "n_models": len(self.models), "disclaimer": b["disclaimer"]}
        if self.baseline is not None and age is not None and sex in ("M", "F"):
            pb = self.baseline.predict_proba(np.array([[float(age), 1.0 if sex == "M" else 0.0]]))[0]
            out["age_sex_baseline"] = {"probabilities": {b["class_names"][k]: float(pb[k]) for k in range(len(pb))},
                                       "note": "Prediction from age and sex alone. If the MRI model does not differ from this, the image adds little."}
        if with_cam and getattr(self.models[0], "has_feature_map", False):
            cam, _, _ = S.gradcam(self.models[0], vol.astype(np.float32), class_idx=pred, device=self.device)
            fig = S.overlay_png(vol, cam, title=f"Grad-CAM for '{b['class_names'][pred]}' (inspection only, not a validated biomarker)")
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); import matplotlib.pyplot as plt; plt.close(fig)
            out["gradcam_png_base64"] = base64.b64encode(buf.getvalue()).decode()
        return out


def _wrap(x):
    from .config import _wrap as w
    return w(x)
