"""
Federated Learning Client — client.py
======================================
Copy this file (and the security/ + utils/ folders) to each client machine.

Configure via environment variables OR command-line args:
    python client.py --server 10.5.70.249:8080 --id client1 --data data/client1/client1.yaml
    FL_SERVER_ADDRESS=10.5.70.249:8080 FL_CLIENT_ID=client2 python client.py

Required directory layout on each client machine:
    client_project/
    ├── client.py              ← this file
    ├── yolo11n.pt             ← YOLO base model weights
    ├── security/
    │   ├── __init__.py
    │   └── hashing.py
    ├── utils/
    │   ├── __init__.py
    │   └── model_utils.py
    └── data/
        └── client<N>/
            ├── images/        ← training images
            ├── labels/        ← YOLO-format label files
            └── client<N>.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import flwr as fl
from ultralytics import YOLO

from utils.model_utils import get_parameters, set_parameters
from security.hashing import hash_parameters


# =========================================================
# ARGUMENT PARSING  (env vars → CLI args → defaults)
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning YOLO Client"
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("FL_SERVER_ADDRESS", "localhost:8080"),
        help="Aggregator address  e.g. 10.5.70.249:8080",
    )
    parser.add_argument(
        "--id",
        default=os.environ.get("FL_CLIENT_ID", "client1"),
        help="Unique client identifier  e.g. client1 / client2 / client3",
    )
    parser.add_argument(
        "--data",
        default=os.environ.get(
            "FL_DATASET_YAML",
            "data/client1/client1.yaml",
        ),
        help="Path to the YOLO dataset YAML for this client",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("FL_LOCAL_EPOCHS", "1")),
        help="Local training epochs per FL round",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.environ.get("FL_IMAGE_SIZE", "640")),
        help="YOLO image size",
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get("FL_WEIGHTS", "yolo11n.pt"),
        help="YOLO model weights file",
    )
    return parser.parse_args()


# =========================================================
# HELPER UTILITIES
# =========================================================

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total else 0
    return f"[{bar}] {pct:.0f}%"


def count_training_images(data_dir: str) -> int:
    """Count images in a given directory (jpg/jpeg/png)."""
    image_dir = Path(data_dir)
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        total += len(list(image_dir.glob(ext)))
    return total


# =========================================================
# FEDERATED CLIENT CLASS
# =========================================================

class FLClient(fl.client.NumPyClient):

    def __init__(self, args):
        self.args        = args
        self.client_id   = args.id
        self.server_addr = args.server

        print("\n" + "=" * 60)
        print(f"  FEDERATED LEARNING CLIENT  —  {self.client_id.upper()}")
        print("=" * 60)
        print(f"  Server   : {self.server_addr}")
        print(f"  Dataset  : {args.data}")
        print(f"  Epochs   : {args.epochs} per round")
        print(f"  Img size : {args.imgsz}")
        print(f"  Weights  : {args.weights}")

        # ----- Validate dataset YAML exists -----
        if not Path(args.data).exists():
            print(f"\n  ✘  Dataset YAML not found: {args.data}")
            print(f"     Make sure you ran the annotation converter first.")
            sys.exit(1)

        # ----- Load YOLO model -----
        print(f"\n  ► Loading YOLO model from {args.weights}…")
        if not Path(args.weights).exists():
            print(f"  ✘  Weights file not found: {args.weights}")
            sys.exit(1)

        self.yolo  = YOLO(args.weights)
        self.model = self.yolo.model

        # ----- Count images -----
        yaml_dir    = str(Path(args.data).parent / "images")
        self.num_examples = count_training_images(yaml_dir)

        print(f"  Training images  : {self.num_examples:,}")
        print(f"\n  ✔  {self.client_id} ready. Connecting to {self.server_addr}…")
        print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 0 — Send initial parameters to server (called once at start)
    # ------------------------------------------------------------------
    def get_parameters(self, config):
        print(f"\n  [{self.client_id}] Sending initial model parameters to server…")
        params  = get_parameters(self.model)
        payload = sum(a.nbytes for a in params)
        print(f"  ► Payload: {_human_bytes(payload)}")
        return params

    # ------------------------------------------------------------------
    # STEP 1-5 — Core FL round
    # ------------------------------------------------------------------
    def fit(self, parameters, config):
        server_round = int(config.get("server_round", "?"))

        print(f"\n{'='*60}")
        print(f"  [{self.client_id}]  ROUND {server_round}  —  FIT")
        print(f"{'='*60}")

        # ── 1. Load global model from server ──
        print(f"\n  [1/5] Receiving global model from aggregator…")
        t0 = time.time()
        set_parameters(self.model, parameters)
        payload_in = sum(a.nbytes for a in parameters)
        print(f"       ✔  Received  ({_human_bytes(payload_in)}, {time.time()-t0:.2f}s)")

        # ── 2. Train locally ──
        print(f"\n  [2/5] Starting local YOLO training…")
        print(f"       Dataset : {self.args.data}")
        print(f"       Epochs  : {self.args.epochs}")
        print(f"       Images  : {self.num_examples:,}")
        t1 = time.time()

        self.yolo.train(
            data=self.args.data,
            epochs=self.args.epochs,
            imgsz=self.args.imgsz,
            project="results",
            name=f"{self.client_id}_round{server_round}",
            exist_ok=True,
            verbose=False,   # suppress YOLO verbosity; our wrapper shows progress
        )

        train_time = time.time() - t1
        print(f"\n       ✔  Local training complete ({train_time:.1f}s)")

        # ── 3. Extract updated parameters ──
        print(f"\n  [3/5] Extracting updated model parameters…")
        updated_params = get_parameters(self.model)
        payload_out    = sum(a.nbytes for a in updated_params)
        n_tensors      = len(updated_params)
        print(f"       Tensors  : {n_tensors:,}")
        print(f"       Payload  : {_human_bytes(payload_out)}")

        # ── 4. Compute SHA-256 hash ──
        print(f"\n  [4/5] Computing SHA-256 integrity hash…")
        t2 = time.time()
        model_hash = hash_parameters(updated_params)
        print(f"       Hash     : {model_hash[:32]}… ({time.time()-t2:.2f}s)")

        # ── 5. Send back to aggregator ──
        print(f"\n  [5/5] Sending update to aggregator…")
        print(f"       Payload  : {_human_bytes(payload_out)}")

        metrics = {
            "client_id":    self.client_id,
            "model_hash":   model_hash,
            "train_time_s": round(train_time, 2),
            "num_images":   self.num_examples,
        }

        print(f"       ✔  Update ready for transmission")
        print(f"{'─'*60}")
        print(f"  Round {server_round} complete  |  "
              f"samples={self.num_examples:,}  |  "
              f"time={train_time:.0f}s")
        print(f"{'='*60}\n")

        return (updated_params, self.num_examples, metrics)

    # ------------------------------------------------------------------
    # EVALUATE (minimal stub — server handles real evaluation)
    # ------------------------------------------------------------------
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        return 0.0, self.num_examples, {"client_id": self.client_id}


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    args   = parse_args()
    client = FLClient(args)

    print(f"\n  Connecting to aggregator at {args.server}…\n")

    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )
