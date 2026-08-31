"""
Federated Learning Client — client.py
======================================
New workflow:
  1. Train the YOLO model locally (completely offline)
  2. Connect to the aggregator server
  3. Send the trained .pt file ONCE via HTTP
  4. Exit automatically

Configure via environment variables (.env.local) OR command-line args:
    python client.py --server <SERVER_IP>:8090 --id client1 --data data/client1/client1.yaml
    FL_SERVER_ADDRESS=<SERVER_IP>:8090 FL_CLIENT_ID=client2 python client.py

NOTE: The server IP should be the aggregator machine's IP on the shared network.
      The upload port is 8090 (HTTP), separate from the Flower gRPC port (8080).
"""

# ── Suppress all unnecessary warnings and logs ─────────────────────────────
import logging
import os
import warnings
from dotenv import load_dotenv

load_dotenv(".env")

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import contextlib
import hashlib
import sys
import threading
import time
from pathlib import Path

import requests
from ultralytics import YOLO


# =========================================================
# ARGUMENT PARSING
# =========================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FL YOLO Client — Train then Upload")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a local YOLO model and then upload it to the aggregator.",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect to the aggregator and upload an already-trained model.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FL_MODEL_PATH"),
        help="Path to an existing .pt model to upload when using --connect.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("FL_SERVER_ADDRESS", "localhost:8090"),
        help="Aggregator HTTP upload address (host:port). Default: localhost:8090",
    )
    parser.add_argument(
        "--id",
        default=os.environ.get("FL_CLIENT_ID", "client1"),
        help="Unique client identifier",
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("FL_DATASET_YAML", "data/client1/client1.yaml"),
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("FL_LOCAL_EPOCHS", "1")),
        help="Number of local training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.environ.get("FL_IMAGE_SIZE", "640")),
        help="Training image size",
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get("FL_WEIGHTS", "yolo11n.pt"),
        help="Path to initial YOLO weights (.pt)",
    )
    args = parser.parse_args(argv)
    if args.train and args.connect:
        parser.error("Choose only one mode: --train or --connect.")
    if not args.train and not args.connect:
        args.train = True
    return args


# =========================================================
# HELPERS
# =========================================================

def resolve_model_path(project_root: Path | str = None, run_name: str = None, preferred_path: str | Path = None) -> Path | None:
    """Look for a trained YOLO .pt model in the usual results folders."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parent
    prefer = Path(preferred_path).expanduser() if preferred_path else None
    if prefer and prefer.exists():
        return prefer

    candidate_names = []
    if run_name:
        candidate_names.extend([run_name, f"{run_name}_local", f"{run_name}_training", f"{run_name}_trained"])
    candidate_names.extend(["client1_local", "client1_training", "client2_local", "client2_training", "client3_local", "client3_training"])

    search_roots = []
    for base in [root, root.parent, Path.cwd().resolve()]:
        if base not in search_roots:
            search_roots.append(base)

    for base in search_roots:
        for name in candidate_names:
            candidates = [
                base / "results" / name / "weights" / "best.pt",
                base / "results" / name / "weights" / "last.pt",
                base / "results" / name / "best.pt",
                base / "results" / name / "last.pt",
                base / name / "weights" / "best.pt",
                base / name / "weights" / "last.pt",
                base / name / "best.pt",
                base / name / "last.pt",
                base / "results" / f"{name}.pt",
                base / f"{name}.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate

    return None

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _sep(char: str = "═", width: int = 60) -> str:
    return char * width


def count_training_images(data_dir: str) -> int:
    image_dir = Path(data_dir)
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        total += len(list(image_dir.glob(ext)))
    return total


def hash_file(path: Path) -> str:
    """SHA-256 hash of a file on disk."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# =========================================================
# LIVE SPINNER
# =========================================================

class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Working"):
        self._label   = label
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._elapsed = 0.0

    def _spin(self):
        start = time.time()
        from itertools import cycle
        for frame in cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self._elapsed = time.time() - start
            sys.__stdout__.write(f"\r  {frame}  {self._label} … ({self._elapsed:.0f}s)")
            sys.__stdout__.flush()
            time.sleep(0.1)
        sys.__stdout__.write(
            f"\r  ✔  {self._label} done  ({self._elapsed:.1f}s)          \n"
        )
        sys.__stdout__.flush()

    def start(self):
        self._thread.start()
        return self

    def stop(self, success: bool = True, final_msg: str = ""):
        self._stop.set()
        self._thread.join()
        if not success:
            sys.__stdout__.write(
                f"\r  ✘  {self._label} failed.                             \n"
            )
            sys.__stdout__.flush()
        elif final_msg:
            sys.__stdout__.write(f"\r  ✔  {final_msg}          \n")
            sys.__stdout__.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# =========================================================
# STEP 1 — LOCAL TRAINING
# =========================================================

def train_locally(args) -> Path:
    """
    Run YOLO training entirely on the local machine.
    Returns the path to the best trained weights (.pt file).
    """
    run_name   = f"{args.id}_local"
    run_dir    = Path("results") / run_name

    print(f"\n{_sep()}")
    print(f"  FL CLIENT  —  {args.id.upper()}")
    print(_sep("─"))
    print(f"  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}  |  Img size : {args.imgsz}")
    print(f"  Weights  : {args.weights}")
    print(f"  Output   : {run_dir}/weights/best.pt")
    print(_sep())
    print()

    # ── Validate inputs ─────────────────────────────────────────────────────
    if not Path(args.data).exists():
        print(f"  ✘  Dataset YAML not found: {args.data}")
        sys.exit(1)
    if not Path(args.weights).exists():
        print(f"  ✘  Weights file not found: {args.weights}")
        sys.exit(1)

    # ── Count images ─────────────────────────────────────────────────────────
    yaml_dir     = str(Path(args.data).parent / "images")
    num_examples = count_training_images(yaml_dir)
    print(f"  [1/3] Starting local training  "
          f"({num_examples:,} images × {args.epochs} epoch(s))…")

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    spinner = Spinner(f"Training [{args.id}]").start()
    yolo = YOLO(args.weights, verbose=False)
    try:
        with open(os.devnull, "w") as f, \
             contextlib.redirect_stdout(f), \
             contextlib.redirect_stderr(f):
            yolo.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                project="results",
                name=run_name,
                exist_ok=True,
                verbose=False,
            )
    except Exception as e:
        spinner.stop(success=False)
        print(f"  ✘  Training failed: {e}")
        sys.exit(1)
    finally:
        spinner.stop()

    train_time = time.time() - t0
    print(f"  [1/3] ✔  Training complete  ({train_time:.1f}s)")

    # ── Locate the saved .pt file ─────────────────────────────────────────────
    pt_path = resolve_model_path(
        project_root=Path.cwd().resolve(),
        run_name=args.id,
        preferred_path=(run_dir / "weights" / "best.pt"),
    )
    if pt_path is None:
        pt_path = resolve_model_path(project_root=Path.cwd().resolve(), run_name=f"{args.id}_local")
    if pt_path is None:
        pt_path = resolve_model_path(project_root=Path.cwd().resolve(), run_name=f"{args.id}_training")

    if pt_path is None:
        print(f"  ✘  Could not find trained weights in {run_dir}/weights/ or the YOLO results folder.")
        sys.exit(1)

    size = pt_path.stat().st_size
    print(f"  [2/3] Model saved  →  {pt_path}  ({_human_bytes(size)})")

    # ── Hash the file ─────────────────────────────────────────────────────────
    file_hash = hash_file(pt_path)
    print(f"  [3/3] SHA-256  : {file_hash[:32]}…")
    print(_sep() + "\n")

    return pt_path, file_hash, num_examples, train_time


# =========================================================
# STEP 2 — UPLOAD TO AGGREGATOR
# =========================================================

def upload_to_aggregator(args, pt_path: Path, file_hash: str,
                          num_examples: int, train_time: float):
    """
    Connect to the aggregator's HTTP endpoint and upload the .pt file once.
    Exits the process after a successful upload (or after a fatal error).
    """
    server_host = args.server.split(":")[0]
    server_port = args.server.split(":")[1] if ":" in args.server else "8090"
    upload_url  = f"http://{server_host}:{server_port}/upload"

    print(_sep())
    print(f"  UPLOADING TO AGGREGATOR")
    print(_sep("─"))
    print(f"  Endpoint : {upload_url}")
    print(f"  File     : {pt_path.name}  ({_human_bytes(pt_path.stat().st_size)})")
    print(f"  Hash     : {file_hash[:32]}…")
    print()

    metadata = {
        "client_id":    args.id,
        "model_hash":   file_hash,
        "train_time_s": round(train_time, 2),
        "num_examples": num_examples,
        "epochs":       args.epochs,
        "imgsz":        args.imgsz,
        "weights_used": args.weights,
    }

    # ── Retry loop (3 attempts) ───────────────────────────────────────────────
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  ► Connecting…  (attempt {attempt}/{MAX_RETRIES})")
            t_up = time.time()

            with open(pt_path, "rb") as pt_file:
                response = requests.post(
                    upload_url,
                    files={"model": (pt_path.name, pt_file, "application/octet-stream")},
                    data=metadata,
                    timeout=300,  # 5 min timeout for large files
                )

            elapsed = time.time() - t_up

            if response.status_code == 200:
                resp_data = response.json()
                print(f"  ✔  Upload successful  ({elapsed:.1f}s)")
                print(f"  Server response : {resp_data.get('message', 'OK')}")
                print(_sep() + "\n")
                print(f"  🎉  Training complete. Process exiting.")
                print(_sep() + "\n")
                sys.exit(0)

            else:
                print(f"  ✘  Server returned HTTP {response.status_code}: {response.text}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s…")
                    time.sleep(RETRY_DELAY)

        except requests.exceptions.ConnectionError:
            print(f"  ✘  Could not connect to {upload_url}")
            if attempt < MAX_RETRIES:
                print(f"  ⏳  Waiting {RETRY_DELAY}s before retry…")
                time.sleep(RETRY_DELAY)

        except requests.exceptions.Timeout:
            print(f"  ✘  Upload timed out on attempt {attempt}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  ✘  Unexpected error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(f"\n  ✘  All {MAX_RETRIES} upload attempts failed.")
    print(f"  Trained model is saved locally at: {pt_path}")
    print(_sep() + "\n")
    sys.exit(1)


def connect_and_upload(args):
    """Upload a model that already exists on disk without re-training."""
    project_root = Path(__file__).resolve().parent
    data_file = Path(args.data)
    if not data_file.is_absolute():
        candidate_data = project_root / data_file
        if candidate_data.exists():
            data_file = candidate_data

    pt_path = Path(args.model).expanduser() if args.model else None
    if pt_path is None or not pt_path.exists():
        pt_path = resolve_model_path(project_root=project_root, run_name=f"{args.id}_local")
    if pt_path is None or not pt_path.exists():
        pt_path = resolve_model_path(project_root=project_root, run_name=f"{args.id}_training")
    if pt_path is None or not pt_path.exists():
        print(f"  ✘  No trained model found for client {args.id}. Train it first or pass --model <path>.")
        sys.exit(1)

    file_hash = hash_file(pt_path)
    num_examples = count_training_images(str(data_file.parent / "images")) if data_file.exists() else 0
    train_time = 0.0
    print(f"  [connect] Found trained model → {pt_path}")
    upload_to_aggregator(args, pt_path, file_hash, num_examples, train_time)


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    args = parse_args()

    if args.connect:
        connect_and_upload(args)
    else:
        pt_path, file_hash, num_examples, train_time = train_locally(args)
        print(f"  Connecting to aggregator at {args.server}…")
        print(f"  (Trained model will be uploaded once and the process will close)\n")
        upload_to_aggregator(args, pt_path, file_hash, num_examples, train_time)
