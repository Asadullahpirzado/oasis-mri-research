"""OASIS-1 input/output.

The official OASIS-1 download consists of 12 disc archives (oasis_cross-sectional_disc1..12.tar.gz).
Each session folder contains several versions of the scan. We only need, per session:

  * the processed, atlas-registered, brain-masked volume   ..._mpr_n4_anon_111_t88_masked_gfc.(hdr|img)
  * (optional) the FSL tissue segmentation                  ..._masked_gfc_fseg.(hdr|img)

The files are Analyze 7.5 (.hdr/.img pairs), which nibabel reads directly.
This module streams through each archive ONCE and extracts only those files to fast local disk
(about 6 GB for all 436 sessions instead of the full ~20 GB). Archives that were already unzipped
(loose folders on Drive) and .zip archives are handled as well.
"""
import os, re, shutil, tarfile, zipfile, glob, json, io, gzip
import numpy as np
import pandas as pd

from ..utils import log

ARCHIVE_EXT = (".tar.gz", ".tgz", ".tar", ".zip")
SESSION_RE = re.compile(r"(OAS1_\d{4}_MR\d+)", re.IGNORECASE)


# ─────────────────────────────── clinical table ───────────────────────────────
def find_clinical_file(data_dir, explicit=None):
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"clinical_file not found: {explicit}")
    cands = []
    for root, _, files in os.walk(data_dir):
        depth = root[len(data_dir):].count(os.sep)
        if depth > 2:
            continue
        for f in files:
            fl = f.lower()
            if fl.endswith((".csv", ".xlsx", ".xls")) and ("oasis" in fl or "cross" in fl):
                cands.append(os.path.join(root, f))
    if not cands:
        raise FileNotFoundError(
            f"No clinical file (oasis_cross-sectional*.csv/.xlsx) found under {data_dir}. "
            "Download it from the OASIS-1 page and put it next to the disc archives, "
            "or set paths.clinical_file in the config.")
    cands.sort(key=lambda p: (not p.lower().endswith(".csv"), len(p)))
    return cands[0]


_COLMAP = {"id": "session_id", "m/f": "sex", "sex": "sex", "gender": "sex", "hand": "handedness",
           "age": "age", "educ": "educ", "education": "educ", "ses": "ses", "mmse": "mmse",
           "cdr": "cdr", "etiv": "etiv", "asf": "asf", "nwbv": "nwbv", "delay": "delay"}


def read_clinical(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: _COLMAP[c.lower()] for c in df.columns if c.lower() in _COLMAP})
    need = {"session_id", "age", "cdr"}
    if not need.issubset(df.columns):
        raise ValueError(f"Clinical file is missing columns {need - set(df.columns)}; found {list(df.columns)}")
    df["session_id"] = df["session_id"].astype(str).str.strip().str.upper()
    for c in ("age", "cdr", "mmse", "etiv", "asf", "nwbv", "educ", "ses", "delay"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.strip().str.upper().str[0]
    return df


# ─────────────────────────────── archives → local files ───────────────────────────────
def list_archives(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(ARCHIVE_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)


def classify_member(name, cfg):
    """Return (session_id, kind, ext) if this archive member is a file we need, else None."""
    base = os.path.basename(name)
    bl = base.lower()
    if not bl.endswith((".img", ".hdr")):
        return None
    m = SESSION_RE.search(base)
    if not m:
        return None
    ext = bl[-4:]
    sid = m.group(1).upper()
    seg_suffix = cfg.data.seg_suffix.lower()
    t88_suffix = cfg.data.t88_suffix.lower()
    stem = bl[:-4]
    if stem.endswith(seg_suffix):
        return sid, "seg", ext
    if stem.endswith(t88_suffix) and "t88" in name.lower():
        return sid, "t88", ext
    return None


def _marker(local_extracted, archive):
    d = os.path.join(local_extracted, ".done")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, os.path.basename(archive) + ".json")


def _write_member(src, out_path, size=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(src, f, 1 << 20)
    os.replace(tmp, out_path)


def extract_archive(archive, local_extracted, cfg):
    """Stream one archive once and write only the needed files. Returns number of files written."""
    n = 0
    low = archive.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                c = classify_member(info.filename, cfg)
                if c is None:
                    continue
                sid, kind, ext = c
                out = os.path.join(local_extracted, sid, os.path.basename(info.filename))
                if os.path.exists(out) and os.path.getsize(out) == info.file_size:
                    continue
                with zf.open(info) as src:
                    _write_member(src, out)
                n += 1
    else:
        with tarfile.open(archive, mode="r|*") as tf:      # streaming: one pass, low memory
            for m in tf:
                if not m.isfile():
                    continue
                c = classify_member(m.name, cfg)
                if c is None:
                    continue
                sid, kind, ext = c
                out = os.path.join(local_extracted, sid, os.path.basename(m.name))
                if os.path.exists(out) and os.path.getsize(out) == m.size:
                    continue
                src = tf.extractfile(m)
                if src is None:
                    continue
                _write_member(src, out)
                n += 1
    return n


def ensure_extracted(paths, cfg, force=False):
    """Idempotent: extract needed files from every archive that has not been processed in this session."""
    archives = list_archives(paths.data_dir)
    ex = paths.extracted
    if not archives:
        log(f"No archives found under {paths.data_dir} - assuming scans are already unpacked there.")
        return 0
    total = 0
    for i, a in enumerate(archives, 1):
        mk = _marker(ex, a)
        if os.path.exists(mk) and not force:
            log(f"[{i}/{len(archives)}] already extracted: {os.path.basename(a)}")
            continue
        log(f"[{i}/{len(archives)}] extracting needed files from {os.path.basename(a)} ...")
        n = extract_archive(a, ex, cfg)
        with open(mk, "w") as f:
            json.dump({"archive": a, "files_written": n}, f)
        total += n
        log(f"    wrote {n} files")
    return total


def build_file_index(roots, cfg):
    """Walk folders and return DataFrame: session_id, t88_img, seg_img (only complete hdr+img pairs)."""
    found = {}
    for root in roots:
        for dp, _, files in os.walk(root):
            if os.sep + ".done" in dp:
                continue
            for f in files:
                full = os.path.join(dp, f)
                c = classify_member(full, cfg)
                if c is None:
                    continue
                sid, kind, ext = c
                found.setdefault(sid, {}).setdefault(kind, {})[ext] = full
    rows = []
    for sid, d in sorted(found.items()):
        row = {"session_id": sid, "t88_img": None, "seg_img": None}
        for kind in ("t88", "seg"):
            e = d.get(kind, {})
            if ".img" in e and ".hdr" in e:
                row[f"{kind}_img"] = e[".img"]
        rows.append(row)
    return pd.DataFrame(rows, columns=["session_id", "t88_img", "seg_img"])


def get_file_index(paths, cfg, extract=True):
    if extract:
        ensure_extracted(paths, cfg)
    return build_file_index([paths.extracted, paths.data_dir], cfg)


# ─────────────────────────────── volume loading ───────────────────────────────
def load_volume(path):
    """Load .img/.hdr (Analyze), .nii or .nii.gz as a 3-D float32 array."""
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {arr.shape} from {path}")
    return arr


def load_volume_bytes(data, filename, header_data=None):
    """Load a NIfTI file or Analyze pair from uploaded bytes without writing to disk."""
    import nibabel as nib
    low = filename.lower()
    if low.endswith((".nii", ".nii.gz")):
        raw = gzip.decompress(data) if low.endswith(".nii.gz") else data
        img = nib.Nifti1Image.from_bytes(raw)
    elif low.endswith(".img") and header_data is not None:
        fmap = nib.AnalyzeImage.make_file_map()
        fmap["header"].fileobj = io.BytesIO(header_data)
        fmap["image"].fileobj = io.BytesIO(data)
        img = nib.AnalyzeImage.from_file_map(fmap)
    else:
        raise ValueError("Upload either one .nii/.nii.gz file, or an Analyze .img with its matching .hdr file.")
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {arr.shape} from {filename}")
    return arr
