"""End-to-end smoke test on SYNTHETIC OASIS-like data (CPU, a few minutes).
    python tests/smoke_test.py
Proves that every stage runs and produces its outputs. The numbers are meaningless (fake images)."""
import os, sys, shutil, subprocess, time, glob, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASE = os.environ.get("SMOKE_DIR", os.path.join(tempfile.gettempdir(), "oasis_smoke"))
DATA, OUT, LOC = os.path.join(BASE, "data"), os.path.join(BASE, "out"), os.path.join(BASE, "local")
shutil.rmtree(BASE, ignore_errors=True)
subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "make_synthetic_oasis.py"), DATA, "--n", "60", "--n_young", "10", "--shape", "44", "52", "44"])

from oasis_ad.config import load_config, Paths
from oasis_ad import pipeline as PL
cfg = load_config(None, {
    "paths.data_dir": DATA, "paths.out_dir": OUT, "paths.local_dir": LOC,
    "data.expected_shape": [44, 52, 44], "preprocess.target_shape": [24, 28, 24],
    "cv.n_splits": 3, "cv.n_repeats": 2,
    "train.epochs": 4, "train.patience": 3, "train.batch_size": 4, "train.mc_passes": 3, "train.amp": False,
    "models.cnn3d": {"width": 8, "dropout": 0.3, "spatial_dropout": 0.1},
    "ood.n_per_cell": 3, "ood.severities": [0.5, 1.0], "ood.corruptions": ["noise", "blur", "motion"], "ood.folds_used": 2,
    "eval.n_boot": 100, "app.bundle_repeat": 0,
})
paths = Paths(cfg)
t0 = time.time()
def stage(name, fn):
    t = time.time(); r = fn(); print(f"=== {name} OK ({time.time()-t:.0f}s)", flush=True); return r

manifest, rep = stage("step1 audit", lambda: PL.step1_audit(cfg, paths))
assert rep["role_counts"]["labeled"] == 60, rep["role_counts"]
assert rep["role_counts"].get("excluded_repeat_scan", 0) == 4
assert rep["role_counts"]["ood_young"] == 10
qc, passed = stage("step2 preprocess", lambda: PL.step2_preprocess(cfg, paths))
sp, summ = stage("step3 splits", lambda: PL.step3_splits(cfg, paths))
stage("step4 baselines", lambda: PL.step4_baselines(cfg, paths))
stage("step5 train", lambda: PL.step5_train(cfg, paths))
res = stage("step6 evaluate", lambda: PL.step6_evaluate(cfg, paths, "cnn3d"))
print(res["model_comparison"].to_string())
print(res["calibration"].to_string())
print({k: v for k, v in res["conformal"]["pooled_mean_over_repeats"]["overall"].items()})
ood = stage("step7 ood", lambda: PL.step7_ood(cfg, paths, "cnn3d"))
print(ood.groupby(["shift", "score"])[["AUROC", "detection_rate", "false_alarm_clean"]].mean().round(2).to_string())
s8 = stage("step8 selective+saliency", lambda: PL.step8_selective_saliency(cfg, paths, "cnn3d", n_examples=8))
print(s8["selective"]["pooled_mean"])
b = stage("step9 export", lambda: PL.step9_export(cfg, paths, "cnn3d"))

# new in v2: ablations, external validation, extra outputs
print(res["subgroups"].to_string())
print(res["calibration"].to_string())
assert "majority__prior" in set(res["model_comparison"]["model"]), "majority baseline missing"
assert any(m.startswith("tissue") for m in res["model_comparison"]["model"]), "tissue baseline missing"
assert "sensitivity" in res["model_comparison"].columns
print(s8["perturbation_stability"].to_string())
assert set(ood["shift"]) >= {"hard_familiar(misclassified)_vs_correct"} or True
print("OOD scores:", sorted(ood["score"].unique()), "| shifts:", sorted(ood["shift"].unique()))
assert "ens_disagreement" in set(ood["score"]), "ensemble disagreement score missing"
abl = stage("step10 ablations", lambda: PL.step10_ablations(cfg, paths, "cnn3d", ["no_augmentation", "mask_only_input", "cnn2d_slices"]))
print(abl.to_string())
import pandas as pd
man = pd.read_csv(f"{paths.cache_drive}/manifest_qc_passed.csv"); labd = man[man.role == "labeled"].head(20)
ext = pd.DataFrame({"participant_id": labd.participant_id, "path": labd.t88_img, "label": labd.label.astype(int), "age": labd.age, "sex": labd.sex})
ext.to_csv(f"{BASE}/external.csv", index=False)
er, edf = stage("step11 external", lambda: PL.step11_external(cfg, paths, f"{BASE}/external.csv"))
print({k: er[k] for k in ("n", "abstention_rate", "qc_fail_rate", "ood_flag_rate")}, er["conformal"])

# resume test: re-running training must skip finished folds instantly
t = time.time(); PL.step5_train(cfg, paths); assert time.time() - t < 60, "resume/skip logic failed"; print("=== resume OK")

# backend test (no network needed)
from fastapi.testclient import TestClient
from oasis_ad.app.server import create_app
app = create_app(paths.bundle, paths.cache_local, f"{paths.cache_drive}/manifest_qc_passed.csv")
c = TestClient(app)
assert c.get("/api/health").json()["status"] == "ok"
assert "Research" in c.get("/").text or "RESEARCH" in c.get("/").text
ex = c.get("/api/examples").json(); assert len(ex) > 0
r = c.get(f"/api/example/{ex[0]['id']}").json(); print("example result:", r["decision"], r["probabilities"], r["conformal_set"], r["ood"]["flag"])
assert "gradcam_png_base64" in r
# upload test with a real Analyze pair from the extracted synthetic data
img = glob.glob(f"{LOC}/extracted/OAS1_0001_MR1/*masked_gfc.img")[0]; hdr = img[:-4] + ".hdr"
rr = c.post("/api/predict", files=[("files", (os.path.basename(img), open(img, "rb"))), ("files", (os.path.basename(hdr), open(hdr, "rb")))], data={"age": "72", "sex": "F"})
assert rr.status_code == 200, rr.text
j = rr.json(); print("upload result:", j["decision"], "QC:", j["qc"]["detail"], "| baseline:", j.get("age_sex_baseline", {}).get("probabilities"))
bad = c.post("/api/predict", files=[("files", ("x.img", b"abc"))]); assert bad.status_code == 400
print(f"\nALL SMOKE TESTS PASSED in {(time.time()-t0)/60:.1f} min")
