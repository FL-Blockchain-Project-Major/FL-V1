"""
Federated Learning Aggregator — server.py
==========================================
Run from the project root:
    python -m aggregator.server

Current workflow:
  - Listens on HTTP port 8090 for .pt file uploads from clients
  - Clients may train locally first or connect with an already-trained model
  - Server verifies the SHA-256 hash, saves the file, and logs the result
  - After all expected clients have uploaded valid models, the server exits automatically

Override defaults with environment variables:
    FL_NUM_CLIENTS=3 python -m aggregator.server
    FL_UPLOAD_PORT=8090 python -m aggregator.server

NOTE: Binds on 0.0.0.0:8090 (all interfaces). Clients connect using
      the server machine's actual IP on their shared network.
"""

# ── Suppress all unnecessary warnings and logs BEFORE any imports ──────────
import logging
import os
import warnings
from dotenv import load_dotenv

load_dotenv(".env")

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import json
import socket
import sys
import threading
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path

from flask import Flask, request, jsonify


# =========================================================
# CONFIGURATION  (override via environment variables)
# =========================================================

# Number of clients expected before server auto-shuts down
NUM_CLIENTS   = int(os.environ.get("FL_NUM_CLIENTS",  "3"))

# HTTP upload port — clients POST their .pt files here
UPLOAD_PORT   = int(os.environ.get("FL_UPLOAD_PORT", "8090"))
BIND_ADDRESS  = "0.0.0.0"

LOG_DIR       = Path("aggregator/logs")
MODELS_DIR    = Path("aggregator/received_models")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID    = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_LOG   = LOG_DIR / f"session_{SESSION_ID}.json"

# Shared state (thread-safe via lock)
_state_lock          = threading.Lock()
_accepted_uploads    = 0       # count of successfully verified uploads
_session_log_data    = {
    "session_id": SESSION_ID,
    "started_at": datetime.now().isoformat(),
    "expected_clients": NUM_CLIENTS,
    "uploads": [],
}


# =========================================================
# HELPERS
# =========================================================

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _sep(char: str = "═", width: int = 68) -> str:
    return char * width


def _progress_bar(current: int, total: int, width: int = 36) -> str:
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total else 0
    return f"[{bar}] {current}/{total} ({pct:.0f}%)"


def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file on disk."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _save_session_log():
    with open(SESSION_LOG, "w", encoding="utf-8") as f:
        json.dump(_session_log_data, f, indent=4)


def _get_all_ips() -> list:
    """Return all non-loopback IPv4 addresses for this machine."""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if ":" not in addr and not addr.startswith("127."):
                ips.append(addr)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary = s.getsockname()[0]
            if primary not in ips:
                ips.insert(0, primary)
    except Exception:
        pass
    return list(dict.fromkeys(ips))


def _shutdown_server():
    """Gracefully shut down the process after a short delay."""
    time.sleep(2)
    os._exit(0)


# =========================================================
# HTTP APP (FLASK)
# =========================================================

app = Flask(__name__)
# Allow large uploads (1GB)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint — GET / returns server status."""
    with _state_lock:
        accepted = _accepted_uploads
    return jsonify({
        "status":           "running",
        "session_id":       SESSION_ID,
        "accepted_uploads": accepted,
        "expected_clients": NUM_CLIENTS,
    })

@app.route("/upload", methods=["POST"])
def upload_model():
    global _accepted_uploads

    if 'model' not in request.files:
        return jsonify({"error": "No 'model' file field in request"}), 400

    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    client_id     = request.form.get("client_id",    "unknown")
    reported_hash = request.form.get("model_hash",   None)
    train_time    = request.form.get("train_time_s", "0")
    num_examples  = request.form.get("num_examples", "0")
    epochs        = request.form.get("epochs",       "1")
    imgsz         = request.form.get("imgsz",        "640")

    received_at   = datetime.now().isoformat()

    print(f"\n{_sep()}")
    print(f"  UPLOAD RECEIVED  —  {client_id.upper()}")
    print(_sep("─"))
    print(f"  Client ID    : {client_id}")
    print(f"  Train time   : {train_time}s  |  Images : {num_examples}  |  Epochs : {epochs}")
    print(f"  Reported hash: {reported_hash[:32] if reported_hash else 'NONE'}…")

    # ── Save to disk ───────────────────────────────────────────────────
    save_path = MODELS_DIR / f"{client_id}_{SESSION_ID}.pt"
    model_file.save(str(save_path))
    file_size = save_path.stat().st_size
    print(f"  File size    : {_human_bytes(file_size)}")

    # ── Verify hash ────────────────────────────────────────────────────
    calculated_hash = _hash_file(save_path)
    hash_valid = (reported_hash is not None) and (reported_hash == calculated_hash)
    status_sym = "✔" if hash_valid else "✘"
    status_lbl = "ACCEPTED" if hash_valid else "REJECTED (hash mismatch)"

    print(f"  Calculated   : {calculated_hash[:32]}…")
    print(f"  Hash check   : {status_sym}  {status_lbl}")

    # ── Update shared state ────────────────────────────────────────────
    upload_record = {
        "client_id":       client_id,
        "received_at":     received_at,
        "file_size_bytes": file_size,
        "save_path":       str(save_path),
        "train_time_s":    train_time,
        "num_examples":    num_examples,
        "epochs":          epochs,
        "imgsz":           imgsz,
        "reported_hash":   reported_hash,
        "calculated_hash": calculated_hash,
        "status":          "accepted" if hash_valid else "rejected",
    }

    with _state_lock:
        _session_log_data["uploads"].append(upload_record)
        if hash_valid:
            _accepted_uploads += 1
            current_accepted = _accepted_uploads
        else:
            current_accepted = _accepted_uploads

    _save_session_log()

    if not hash_valid:
        # Remove the rejected file
        save_path.unlink(missing_ok=True)

    progress = _progress_bar(current_accepted, NUM_CLIENTS)
    print(f"\n  Progress     : {progress}")
    print(f"  📄  Log → {SESSION_LOG.name}")
    print(_sep() + "\n")

    # ── Check if all clients have uploaded — shutdown ──────────────────
    if hash_valid and current_accepted >= NUM_CLIENTS:
        _session_log_data["completed_at"] = datetime.now().isoformat()
        _session_log_data["status"]       = "complete"
        _save_session_log()

        print(f"\n{_sep()}")
        print(f"  🎉  ALL {NUM_CLIENTS} CLIENT(S) UPLOADED SUCCESSFULLY.")
        print(f"  Models saved in : {MODELS_DIR.resolve()}")
        print(f"  Session log     : {SESSION_LOG.resolve()}")
        print(f"  Aggregator shutting down…")
        print(_sep() + "\n")

        # Shutdown in a background thread so HTTP response is sent first
        threading.Thread(target=_shutdown_server, daemon=True).start()

    # ── Send response ──────────────────────────────────────────────────
    if hash_valid:
        return jsonify({
            "message":   f"Model accepted and saved as {save_path.name}",
            "client_id": client_id,
            "status":    "accepted",
            "progress":  f"{current_accepted}/{NUM_CLIENTS}",
        }), 200
    else:
        return jsonify({
            "error":     "Hash mismatch — model rejected",
            "client_id": client_id,
            "status":    "rejected",
        }), 422


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    all_ips = _get_all_ips()

    print("\n" + _sep())
    print("  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n")
    print(_sep())
    print(f"  Session ID      : {SESSION_ID}")
    print(f"  Upload port     : {UPLOAD_PORT}  (HTTP — all interfaces)")
    print(f"  Expected clients: {NUM_CLIENTS}")
    print(f"  Models saved to : {MODELS_DIR.resolve()}")
    print(f"  Log directory   : {LOG_DIR.resolve()}")
    print()
    print(f"  ► Clients should connect using ONE of these IPs:")
    if all_ips:
        for ip in all_ips:
            print(f"      --server {ip}:{UPLOAD_PORT}")
    else:
        print(f"      --server <THIS_MACHINE_IP>:{UPLOAD_PORT}")
    print()
    print(f"  ► Server auto-shuts down after all {NUM_CLIENTS} client(s) upload.")
    print(f"\n  Waiting for client uploads…")
    print(_sep() + "\n")

    app.run(host=BIND_ADDRESS, port=UPLOAD_PORT, debug=False, use_reloader=False)
