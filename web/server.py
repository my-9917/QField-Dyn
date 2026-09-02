import cgi
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


PROJECT_ROOT = Path(os.environ.get("QFIELD_ROOT", Path(__file__).resolve().parents[1]))
WEB_ROOT = Path(__file__).resolve().parent
JOB_ROOT = Path(os.environ.get("QFIELD_WEB_JOBS", PROJECT_ROOT / "web_jobs"))
sys.path.insert(0, str(PROJECT_ROOT))
HOST = os.environ.get("QFIELD_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("QFIELD_WEB_PORT", "8765"))
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
SEED = 20260825
SPECS = {
    "T1": {"n_obs": 10, "n_pred": 10, "dt_ps": 80.0, "ligand_resname": "MOL"},
    "T2": {"n_obs": 80, "n_pred": 20, "dt_ps": 80.0, "ligand_resname": "MOL"},
    "T3": {"n_obs": 20, "n_pred": 80, "dt_ps": 80.0, "ligand_resname": "MOL"},
    "T4": {"n_obs": 10, "n_pred": 490, "dt_ps": 1000.0, "ligand_resname": "LIG"},
}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
JOBS = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def set_job(job_id, **updates):
    with LOCK:
        JOBS[job_id].update(updates)


def extract_dataset(archive_path, data_root, tier):
    with zipfile.ZipFile(archive_path) as archive:
        entries = {info.filename: info for info in archive.infolist()}
        ids_path = f"{tier}/ids.txt"
        if ids_path not in entries:
            raise ValueError(f"Missing {ids_path} at the ZIP root")
        ids = [line.strip() for line in archive.read(ids_path).decode("utf-8").splitlines() if line.strip()]
        if not ids or len(ids) > 100:
            raise ValueError("ids.txt must contain 1–100 system IDs")
        if len(ids) != len(set(ids)) or any(not ID_PATTERN.fullmatch(value) for value in ids):
            raise ValueError("ids.txt contains duplicate or invalid system IDs")

        required = [ids_path]
        for complex_id in ids:
            prefix = f"{tier}/{complex_id}/{complex_id}"
            required.extend((f"{prefix}.pdb", f"{prefix}_obs.xtc", f"{tier}/{complex_id}/meta.json"))
        missing = [name for name in required if name not in entries]
        if missing:
            raise ValueError(f"Missing required file: {missing[0]}")

        for name in required:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe ZIP path")
            info = entries[name]
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not accepted")
            target = data_root.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    spec = SPECS[tier]
    for complex_id in ids:
        meta_path = data_root / tier / complex_id / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"id": complex_id, "tier": tier, **spec}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{complex_id}: meta.json field {key} must be {value}")
    return ids


def validate_predictions(data_root, prediction_root, tier, ids):
    import numpy as np

    from tools.public_io import _read_xtc

    for complex_id in ids:
        meta = json.loads((data_root / tier / complex_id / "meta.json").read_text(encoding="utf-8"))
        output = prediction_root / tier / f"{complex_id}_pred.xtc"
        coordinates, _, times = _read_xtc(output)
        if coordinates.shape != (meta["n_pred"], meta["n_atoms"], 3):
            raise ValueError(f"{complex_id}: generated trajectory shape is invalid")
        expected_times = (meta["n_obs"] + np.arange(meta["n_pred"])) * meta["dt_ps"]
        if not np.allclose(times, expected_times):
            raise ValueError(f"{complex_id}: generated trajectory timestamps are invalid")


def run_job(job_id, tier, archive_path):
    job_dir = JOB_ROOT / job_id
    data_root = job_dir / "data"
    prediction_root = job_dir / "predictions"
    log_path = job_dir / "inference.log"
    try:
        set_job(job_id, state="running", label="Validating input", progress=8, message="Checking package structure and temporal specification…")
        ids = extract_dataset(archive_path, data_root, tier)
        set_job(job_id, systems=len(ids), label="Queued on GPU", progress=15, message=f"Validated {len(ids)} system(s). Waiting for the inference worker…")
        command = [
            sys.executable, "-m", "tools.predict_public", "--data", str(data_root),
            "--output", str(prediction_root), "--tiers", tier, "--device", "cuda",
            "--seed", str(SEED),
        ]
        inference_environment = os.environ.copy()
        inference_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=inference_environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                match = re.search(r"predicted (\d+)/(\d+)", line)
                if match:
                    done, total = map(int, match.groups())
                    progress = 15 + round(70 * done / total)
                    set_job(job_id, label="Running inference", progress=progress, message=f"Predicted {done}/{total} system(s).")
            exit_code = process.wait()
        if exit_code:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-8:])
            raise RuntimeError(tail or f"Inference exited with code {exit_code}")

        set_job(job_id, label="Validating output", progress=90, message="Checking XTC frame counts, atom counts and timestamps…")
        validate_predictions(data_root, prediction_root, tier, ids)
        result_path = job_dir / f"QField-Dyn_{tier}_{job_id}.zip"
        with zipfile.ZipFile(result_path, "w", compression=zipfile.ZIP_DEFLATED) as result:
            for output in sorted((prediction_root / tier).glob("*_pred.xtc")):
                result.write(output, f"{tier}/{output.name}")
        set_job(
            job_id, state="completed", label="Completed", progress=100,
            message=f"Generated and validated {len(ids)} trajectory file(s).",
            download_url=f"/api/jobs/{job_id}/download",
        )
    except Exception as error:
        set_job(job_id, state="failed", label="Failed", progress=0, message=str(error))


class Handler(BaseHTTPRequestHandler):
    server_version = "QFieldDynWeb/1.0"

    def json_response(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = (WEB_ROOT / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            self.json_response(HTTPStatus.OK, {"status": "ok", "device": "cuda", "queue": "single-worker"})
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(/download)?", path)
        if not match:
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        job_id, download = match.groups()
        with LOCK:
            job = JOBS.get(job_id)
            payload = dict(job) if job else None
        if payload is None:
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "Unknown job"})
            return
        if download:
            if payload["state"] != "completed":
                self.json_response(HTTPStatus.CONFLICT, {"error": "Result is not ready"})
                return
            result_path = JOB_ROOT / job_id / f"QField-Dyn_{payload['tier']}_{job_id}.zip"
            body = result_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{result_path.name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.json_response(HTTPStatus.OK, payload)

    def do_POST(self):
        if urlparse(self.path).path != "/api/jobs":
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self.json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "ZIP must be smaller than 512 MiB"})
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        tier = form.getfirst("tier", "")
        upload = form["dataset"] if "dataset" in form else None
        if tier not in SPECS or upload is None or not upload.filename.lower().endswith(".zip"):
            self.json_response(HTTPStatus.BAD_REQUEST, {"error": "Provide a ZIP and one T1–T4 specification"})
            return
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True)
        archive_path = job_dir / "dataset.zip"
        with archive_path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)
        with LOCK:
            JOBS[job_id] = {
                "job_id": job_id, "tier": tier, "state": "queued", "label": "Queued",
                "progress": 5, "message": "Upload complete. Prediction job entered the GPU queue.",
                "systems": None, "download_url": None,
            }
        EXECUTOR.submit(run_job, job_id, tier, archive_path)
        self.json_response(HTTPStatus.ACCEPTED, {"job_id": job_id})

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}", flush=True)


if __name__ == "__main__":
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"QField-Dyn web service: http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
