"""Create a SMALL FAKE dataset with the same layout as the OASIS-1 download (tar.gz discs of Analyze .hdr/.img pairs +
clinical csv) so the whole pipeline can be smoke-tested without the real data. The images are synthetic ellipsoids:
results on them mean NOTHING scientifically - they only prove that the code runs end to end.

    python tools/make_synthetic_oasis.py /tmp/fake_oasis --n 70 --shape 44 52 44
"""
import os, sys, argparse, tarfile, shutil
import numpy as np
import pandas as pd
import nibabel as nib


def brain(shape, atrophy, rng):
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    r = (x / 0.75) ** 2 + (y / 0.85) ** 2 + (z / 0.7) ** 2
    mask = r < 1.0
    img = np.where(mask, 100 + 12 * rng.standard_normal(shape), 0.0)
    vent = ((x / (0.12 + 0.25 * atrophy)) ** 2 + (y / (0.2 + 0.3 * atrophy)) ** 2 + (z / (0.18 + 0.2 * atrophy)) ** 2) < 1
    img[mask & vent] = 25 + 5 * rng.standard_normal(int((mask & vent).sum()))
    sulc = r > (0.78 - 0.25 * atrophy)               # widening of cortical CSF with atrophy
    img[mask & sulc & (rng.random(shape) < 0.4 + 0.4 * atrophy)] = 30
    seg = np.zeros(shape, dtype=np.int16)
    seg[mask] = 3; seg[mask & (r > 0.6)] = 2; seg[mask & (vent | (sulc & (img < 40)))] = 1
    return np.clip(img, 0, None).astype(np.int16), seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--n", type=int, default=70); ap.add_argument("--n_young", type=int, default=12)
    ap.add_argument("--shape", type=int, nargs=3, default=[44, 52, 44]); ap.add_argument("--discs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    shape = tuple(a.shape)
    stage = os.path.join(a.out, "_stage"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    rows = []
    n_cn, n_vm, n_m = int(a.n * 0.55), int(a.n * 0.32), a.n - int(a.n * 0.55) - int(a.n * 0.32)
    cdrs = [0.0] * n_cn + [0.5] * n_vm + [1.0] * n_m
    sid = 1
    plan = []
    for c in cdrs:
        age = float(np.clip(rng.normal(70 + 6 * (c > 0), 7), 60, 94))
        plan.append((sid, age, c)); sid += 1
    for _ in range(a.n_young):
        plan.append((sid, float(rng.uniform(19, 58)), np.nan)); sid += 1
    for _ in range(3):                                   # older, but no CDR -> excluded
        plan.append((sid, float(rng.uniform(62, 80)), np.nan)); sid += 1
    sessions = []
    for (i, age, cdr) in plan:
        sessions.append((f"OAS1_{i:04d}_MR1", i, age, cdr))
    for (i, age, cdr) in plan[:4]:                       # repeat scans
        sessions.append((f"OAS1_{i:04d}_MR2", i, age, cdr))
    for s, i, age, cdr in sessions:
        atrophy = float(np.clip(0.15 + 0.012 * max(age - 20, 0) / 3 + (0.0 if np.isnan(cdr) else 0.35 * cdr) + rng.normal(0, 0.06), 0, 1))
        img, seg = brain(shape, atrophy, rng)
        disc = f"disc{(i % a.discs) + 1}"
        d = os.path.join(stage, disc, s, "PROCESSED", "MPRAGE", "T88_111")
        os.makedirs(d, exist_ok=True)
        aff = np.eye(4)
        nib.save(nib.AnalyzeImage(img[..., None], aff), os.path.join(d, f"{s}_mpr_n4_anon_111_t88_masked_gfc.img"))
        d2 = os.path.join(stage, disc, s, "FSL_SEG"); os.makedirs(d2, exist_ok=True)
        nib.save(nib.AnalyzeImage(seg[..., None], aff), os.path.join(d2, f"{s}_mpr_n4_anon_111_t88_masked_gfc_fseg.img"))
        rawd = os.path.join(stage, disc, s, "RAW"); os.makedirs(rawd, exist_ok=True)
        with open(os.path.join(rawd, f"{s}_mpr-1_anon.txt"), "w") as f:
            f.write("not needed\n")
        brainvol = float((img > 0).mean())
        rows.append({"ID": s, "M/F": "F" if rng.random() < 0.6 else "M", "Hand": "R", "Age": round(age), "Educ": 3, "SES": 2,
                     "MMSE": 30 - (0 if np.isnan(cdr) else int(4 * cdr)) if not np.isnan(cdr) else np.nan, "CDR": cdr,
                     "eTIV": round(1400 + rng.normal(0, 120)), "ASF": round(1.2 + rng.normal(0, 0.1), 3),
                     "nWBV": round(0.84 - 0.002 * max(age - 20, 0) / 2 - 0.05 * (0 if np.isnan(cdr) else cdr) + rng.normal(0, 0.012), 3),
                     "Delay": np.nan})
    os.makedirs(a.out, exist_ok=True)
    for k in range(1, a.discs + 1):
        with tarfile.open(os.path.join(a.out, f"oasis_cross-sectional_disc{k}.tar.gz"), "w:gz") as tf:
            tf.add(os.path.join(stage, f"disc{k}"), arcname=f"disc{k}")
    shutil.rmtree(stage)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "oasis_cross-sectional.csv"), index=False)
    print(f"synthetic OASIS-like data written to {a.out}: {len(rows)} sessions, {a.discs} archives")


if __name__ == "__main__":
    main()
