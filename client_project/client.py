"""
Federated Learning Client — client.py
======================================
Configure via environment variables (.env.local) OR command-line args:
    python client.py --server <SERVER_IP>:8080 --id client1 --data data/client1/client1.yaml
    FL_SERVER_ADDRESS=<SERVER_IP>:8080 FL_CLIENT_ID=client2 python client.py

NOTE: The server IP should be the aggregator machine's IP on the shared network
(WiFi, hotspot, etc.). The server binds on all interfaces automatically.
"""

# ── Suppress all unnecessary warnings and logs ─────────────────────────────
import logging
import os
import warnings
from dotenv import load_dotenv

load_dotenv(".env.local")

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
import threading
import time
import contextlib
from itertools import cycle
from pathlib import Path

import flwr as fl
from flwr.client import start_client
from ultralytics import YOLO

from utils.model_utils import get_parameters, set_parameters
from security.hashing import hash_parameters


# =========================================================
# ARGUMENT PARSING
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FL YOLO Client")
    parser.add_argument("--server",  default=os.environ.get("FL_SERVER_ADDRESS", "localhost:8080"))
    parser.add_argument("--id",      default=os.environ.get("FL_CLIENT_ID",      "client1"))
    parser.add_argument("--data",    default=os.environ.get("FL_DATASET_YAML",   "data/client1/client1.yaml"))
    parser.add_argument("--epochs",  type=int, default=int(os.environ.get("FL_LOCAL_EPOCHS", "1")))
    parser.add_argument("--imgsz",   type=int, default=int(os.environ.get("FL_IMAGE_SIZE",   "640")))
    parser.add_argument("--weights", default=os.environ.get("FL_WEIGHTS",         "yolo11n.pt"))
    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================

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
        for frame in cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self._elapsed = time.time() - start
            sys.__stdout__.write(f"\r  {frame}  {self._label} … ({self._elapsed:.0f}s)")
            sys.__stdout__.flush()
            time.sleep(0.1)
        sys.__stdout__.write(f"\r  ✔  {self._label} done  ({self._elapsed:.1f}s)          \n")
        sys.__stdout__.flush()

    def start(self):
        self._thread.start()
        return self

    def stop(self, success: bool = True, final_msg: str = ""):
        self._stop.set()
        self._thread.join()
        if not success:
            sys.__stdout__.write(f"\r  ✘  {self._label} failed.                             \n")
            sys.__stdout__.flush()
        elif final_msg:
            sys.__stdout__.write(f"\r  ✔  {final_msg}          \n")
            sys.__stdout__.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# =========================================================
# FEDERATED CLIENT CLASS
# =========================================================

class FLClient(fl.client.NumPyClient):

    def __init__(self, args):
        self.args        = args
        self.client_id   = args.id
        self.server_addr = args.server

        # ── Validate dataset YAML ───────────────────────────────────────
        if not Path(args.data).exists():
            print(f"  ✘  Dataset YAML not found: {args.data}")
            sys.exit(1)

        if not Path(args.weights).exists():
            print(f"  ✘  Weights file not found: {args.weights}")
            sys.exit(1)

        # ── Load YOLO model (suppress YOLO's own stdout) ────────────────
        self.yolo  = YOLO(args.weights, verbose=False)
        self.model = self.yolo.model

        # ── Count training images ───────────────────────────────────────
        yaml_dir          = str(Path(args.data).parent / "images")
        self.num_examples = count_training_images(yaml_dir)

        # ── Print clean startup banner ──────────────────────────────────
        print("\n" + _sep())
        print(f"  FL CLIENT  —  {self.client_id.upper()}")
        print(_sep("─"))
        print(f"  Server   : {self.server_addr}")
        print(f"  Dataset  : {args.data}")
        print(f"  Images   : {self.num_examples:,}")
        print(f"  Epochs   : {args.epochs} / round   |  Img size : {args.imgsz}")
        print(_sep())
        print(f"  ✔  Ready — connecting independently (no waiting for other clients)")
        print(_sep() + "\n")

    # ── Send initial parameters to server ──────────────────────────────
    def get_parameters(self, config):
        params  = get_parameters(self.model)
        payload = sum(a.nbytes for a in params)
        print(f"  [{self.client_id}] ► Sending initial params to server  "
              f"({_human_bytes(payload)})")
        return params

    # ── Core FL round ───────────────────────────────────────────────────
    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))

        print(f"\n{_sep()}")
        print(f"  [{self.client_id}]  ROUND {server_round}  —  FIT")
        print(_sep("─"))

        # 1. Load global model
        t0 = time.time()
        set_parameters(self.model, parameters)
        payload_in = sum(a.nbytes for a in parameters)
        print(f"  [1/5] Global model received  "
              f"({_human_bytes(payload_in)}, {time.time()-t0:.2f}s)")

        # 2. Local training
        print(f"  [2/5] Local training  "
              f"({self.num_examples:,} images × {self.args.epochs} epoch(s))…")
        t1 = time.time()

        spinner = Spinner(f"Training  [{self.client_id}]").start()
        try:
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.yolo.train(
                    data=self.args.data,
                    epochs=self.args.epochs,
                    imgsz=self.args.imgsz,
                    project="results",
                    name=f"{self.client_id}_round{server_round}",
                    exist_ok=True,
                    verbose=False,
                )
        finally:
            spinner.stop()

        train_time = time.time() - t1
        print(f"  [2/5] ✔  Training complete  ({train_time:.1f}s)")

        # 3. Extract updated parameters
        updated_params = get_parameters(self.model)
        payload_out    = sum(a.nbytes for a in updated_params)
        print(f"  [3/5] Params extracted  "
              f"({len(updated_params):,} tensors  {_human_bytes(payload_out)})")

        # 4. SHA-256 hash
        t2 = time.time()
        model_hash = hash_parameters(updated_params)
        print(f"  [4/5] Hash : {model_hash[:32]}…  ({time.time()-t2:.2f}s)")

        # 5. Send to aggregator
        metrics = {
            "client_id":    self.client_id,
            "model_hash":   model_hash,
            "train_time_s": round(train_time, 2),
            "num_images":   self.num_examples,
        }
        print(f"  [5/5] Sending update to aggregator  ({_human_bytes(payload_out)})")
        print(_sep("─"))
        print(f"  [{self.client_id}] Round {server_round} complete  |  "
              f"images={self.num_examples:,}  time={train_time:.0f}s")
        print(f"  ✔  Task complete. Client will auto-disconnect.")
        print(_sep() + "\n")

        # Schedule auto-shutdown so the client stops after sending data
        def auto_shutdown():
            time.sleep(2)
            os._exit(0)
        threading.Thread(target=auto_shutdown, daemon=True).start()

        return (updated_params, self.num_examples, metrics)

    # ── Evaluate stub ───────────────────────────────────────────────────
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        return 0.0, self.num_examples, {"client_id": self.client_id}


# =========================================================
# ENTRYPOINT
# =========================================================

# Max gRPC message size (512 MB) — must match server setting
GRPC_MAX_MSG_LEN = 536_870_912


if __name__ == "__main__":
    args   = parse_args()
    client = FLClient(args)

    print(f"  Connecting to aggregator at {args.server}…")
    print(f"  (Make sure the server IP matches your shared network interface)\n")

    start_client(
        server_address=args.server,
        client=client.to_client(),
        grpc_max_message_length=GRPC_MAX_MSG_LEN,
    )
