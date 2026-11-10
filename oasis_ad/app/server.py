"""FastAPI backend + static frontend.

    GET  /                       web page
    GET  /api/health
    GET  /api/model-info         what the loaded bundle is
    GET  /api/examples           participants available in the local cache (for a quick demo)
    POST /api/predict            multipart: files (.hdr+.img pair, or .nii / .nii.gz), age?, sex?
    GET  /api/example/{pid}      run the pipeline on a cached OASIS-1 participant (IN-SAMPLE - demo only)

The frontend uses RELATIVE URLs so it also works behind the Colab port proxy.
"""
import os, io, threading, time
import numpy as np

from .. import DISCLAIMER

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def create_app(bundle_dir, cache_dir=None, manifest_csv=None):
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from ..inference import Predictor
    from ..data.oasis_io import load_volume_bytes
    import pandas as pd

    app = FastAPI(title="OASIS-1 trustworthy MRI classifier (research prototype)")
    pred = Predictor(bundle_dir)
    manifest = pd.read_csv(manifest_csv) if manifest_csv and os.path.exists(manifest_csv) else None
    lock = threading.Lock()                       # one GPU/CPU inference at a time

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models": pred.b["n_classes"], "n_models": len(pred.models), "device": str(pred.device),
                "multimodal_available": False}

    @app.get("/api/model-info")
    def info():
        b = pred.b
        return {"model": b["model_name"], "task": b["task_mode"], "classes": b["class_names"], "n_models": len(pred.models),
                "conformal_target_coverage": 1 - b["alpha"], "rejection_confidence_tau": b["conf_tau"],
                "input": "OASIS-1 processed, atlas-registered, brain-masked T1 volume (T88_111 grid)",
                "disclaimer": b["disclaimer"], "note": b.get("conformal_note", ""),
                "multimodal_available": False}

    @app.get("/api/examples")
    def examples():
        if manifest is None:
            return []
        m = manifest[manifest["role"] == "labeled"]
        out = []
        for _, r in m.head(200).iterrows():
            if cache_dir and os.path.exists(os.path.join(cache_dir, f"{r['participant_id']}_vol.npy")):
                out.append({"id": r["participant_id"], "age": float(r["age"]), "sex": r["sex"], "reference_cdr": float(r["cdr"]),
                            "mmse": float(r["mmse"]) if pd.notna(r.get("mmse")) else None,
                            "nwbv": float(r["nwbv"]) if pd.notna(r.get("nwbv")) else None})
        return out

    @app.get("/api/example/{pid}")
    def example(pid: str):
        if not cache_dir:
            raise HTTPException(404, "cache not configured")
        f = os.path.join(cache_dir, f"{pid}_vol.npy")
        if not os.path.exists(f) or "/" in pid or ".." in pid:
            raise HTTPException(404, "unknown participant")
        vol = np.load(f).astype(np.float32)
        mk = os.path.join(cache_dir, f"{pid}_mask.npy")
        mask = np.load(mk).astype(bool) if os.path.exists(mk) else vol != 0
        age = sex = mmse = nwbv = etiv = asf = None
        if manifest is not None:
            r = manifest[manifest["participant_id"] == pid]
            if len(r):
                row = r.iloc[0]
                age, sex = float(row["age"]), row["sex"]
                mmse = float(row["mmse"]) if pd.notna(row.get("mmse")) else 29.0
                nwbv = float(row["nwbv"]) if pd.notna(row.get("nwbv")) else 0.70
                etiv = float(row["etiv"]) if pd.notna(row.get("etiv")) else 1450.0
                asf = float(row["asf"]) if pd.notna(row.get("asf")) else 1.10
        qc_dict = None
        qc_f = os.path.join(cache_dir, "qc_json", f"{pid}.json")
        if os.path.exists(qc_f):
            try:
                import json
                with open(qc_f) as jf:
                    qc_dict = json.load(jf)
            except Exception:
                pass
        with lock:
            res = pred.predict_preprocessed(vol, mask, qc_dict, age, sex)
        res["warning"] = ("This participant was part of the study data (in-sample for most fold-models): "
                          "use for checking the interface only, not as an accuracy estimate.")
        if manifest is not None and len(r):
            res["reference_cdr_from_metadata"] = float(r.iloc[0]["cdr"])
        return res

    @app.post("/api/predict")
    async def predict(files: list[UploadFile] = File(...), age: str = Form(""), sex: str = Form("")):
        uploads = {}
        for uf in files:
            name = os.path.basename(uf.filename or "upload")
            uploads[name.lower()] = (name, await uf.read())
        target = next((v for v in uploads.values() if v[0].lower().endswith((".nii", ".nii.gz"))), None)
        header_data = None
        if target is None:
            img = next((v for v in uploads.values() if v[0].lower().endswith(".img")), None)
            hdr = next((v for v in uploads.values() if v[0].lower().endswith(".hdr")), None)
            if img is None or hdr is None or os.path.splitext(img[0])[0].lower() != os.path.splitext(hdr[0])[0].lower():
                raise HTTPException(400, "Upload either one .nii/.nii.gz file, or BOTH files of an Analyze pair (same name, .hdr and .img).")
            target, header_data = img, hdr[1]
        try:
            raw = load_volume_bytes(target[1], target[0], header_data)
        except Exception as e:
            raise HTTPException(400, f"Could not read the image: {e}")
        try:
            a = float(age) if age.strip() else None
            s = sex.strip().upper()[:1] if sex.strip() else None
            with lock:
                res = pred.predict_raw(raw, a, s)
            res["input_shape"] = list(raw.shape)

            return JSONResponse(res)
        except ValueError as e:
            raise HTTPException(400, f"Invalid patient metadata: {e}")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def run_in_thread(bundle_dir, port=8000, cache_dir=None, manifest_csv=None):
    """Start the server in a background thread (Colab notebooks). Returns the uvicorn server object."""
    import uvicorn
    app = create_app(bundle_dir, cache_dir, manifest_csv)
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    return server


if __name__ == "__main__":
    import argparse, uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--cache"); ap.add_argument("--manifest")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(create_app(a.bundle, a.cache, a.manifest), host="0.0.0.0", port=a.port)
