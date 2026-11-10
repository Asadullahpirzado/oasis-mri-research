"""Run the OASIS-1 MRI Classifier Web App locally on your laptop.

Usage:
    python run_app.py
    python run_app.py --bundle /path/to/downloaded/step9_bundle
    python run_app.py --port 8080
"""
import os
import sys
import argparse
import webbrowser
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from oasis_ad.app.server import create_app


def find_default_bundle():
    candidates = [
        os.path.join(ROOT, "step9_bundle"),
        os.path.join(ROOT, "OASIS1_outputs", "step9_bundle"),
        os.path.join(ROOT, "outputs", "step9_bundle"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "bundle.json")):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Run OASIS-1 MRI Classifier on Localhost")
    parser.add_argument("--bundle", default=None, help="Path to exported step9_bundle folder")
    parser.add_argument("--cache", default=None, help="Optional local volume cache directory")
    parser.add_argument("--manifest", default=None, help="Optional manifest_qc_passed.csv path")
    parser.add_argument("--port", type=int, default=8000, help="Local port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    bundle_dir = args.bundle or find_default_bundle()
    if not bundle_dir or not os.path.exists(os.path.join(bundle_dir, "bundle.json")):
        print("\n" + "=" * 70)
        print("ERROR: Deployment bundle not found!")
        print("To run the model on your laptop, you need the trained bundle:")
        print("1. Train the model on Google Colab through NB09.")
        print("2. Download the 'step9_bundle' folder from Google Drive.")
        print("3. Put 'step9_bundle' into this project folder or run:")
        print(f"       python run_app.py --bundle <path/to/step9_bundle>")
        print("=" * 70 + "\n")
        sys.exit(1)

    manifest = args.manifest
    if not manifest:
        cand = [
            os.path.join(ROOT, "OASIS1_outputs", "step2_cache", "manifest_qc_passed.csv"),
            os.path.join(ROOT, "outputs", "step2_cache", "manifest_qc_passed.csv"),
            os.path.join(bundle_dir, "manifest_qc_passed.csv"),
        ]
        manifest = next((c for c in cand if os.path.exists(c)), None)

    cache = args.cache
    if not cache:
        cand = [
            os.path.join(ROOT, "OASIS1_outputs", "step2_cache"),
            os.path.join(ROOT, "outputs", "step2_cache"),
            os.path.join(bundle_dir, "cache"),
        ]
        cache = next((c for c in cand if os.path.exists(c)), None)

    print("=" * 70)
    print("Starting OASIS-1 MRI Classifier Web App")
    print(f"Bundle directory : {bundle_dir}")
    print(f"Cache directory  : {cache or '(none - upload mode only)'}")
    print(f"Manifest         : {manifest or '(none)'}")
    print(f"URL              : http://localhost:{args.port}")
    print("=" * 70)

    if not args.no_browser:
        webbrowser.open_new_tab(f"http://localhost:{args.port}")

    app = create_app(bundle_dir, cache_dir=cache, manifest_csv=manifest)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
