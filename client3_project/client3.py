"""
Federated Learning Client — client3.py
=======================================
Run from inside the client3_project/ directory:

    python client3.py              # train then connect (default)
    python client3.py --train      # local training only
    python client3.py --connect    # connect and send pre-trained model
    python client3.py --train --connect  # same as default

All defaults are set via the .env file in this directory.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

from ultralytics import YOLO
import flwr as fl

from utils.model_utils import get_parameters, set_parameters
from security.hashing import hash_parameters


# =========================================================
# CONFIGURATION  (all defaults come from .env)
# =========================================================

CLIENT_ID     = os.environ.get("FL_CLIENT_ID",    "client3")
SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "localhost:8080")
DATASET_YAML  = os.environ.get("FL_DATASET_YAML", "data/client3/client3.yaml")
LOCAL_EPOCHS  = int(os.environ.get("FL_LOCAL_EPOCHS", "1"))
IMAGE_SIZE    = int(os.environ.get("FL_IMAGE_SIZE",   "640"))
WEIGHTS       = os.environ.get("FL_WEIGHTS",      "yolo11n.pt")

BASE_DIR      = Path(__file__).resolve().parent
DATASET_PATH  = BASE_DIR / DATASET_YAML
CLIENT_DATA_DIR = BASE_DIR.parent / "clients" / CLIENT_ID
IMAGES_DIR    = CLIENT_DATA_DIR / "images"
LABELS_DIR    = CLIENT_DATA_DIR / "labels"
RESULTS_DIR   = BASE_DIR / "results"


# =========================================================
# HELPERS
# =========================================================


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _sep(char="─", width=60):
    return char * width


def _count_images(images_dir: Path) -> int:
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        total += len(list(images_dir.glob(ext)))
    return total


def _count_labels(labels_dir: Path) -> int:
    if not labels_dir.exists():
        return 0
    return len(list(labels_dir.glob("*.txt")))


def _verify_paths():
    """Raise FileNotFoundError if required paths are missing."""
    for path, label in [
        (DATASET_PATH.parent, "Dataset directory"),
        (IMAGES_DIR,          "Images directory"),
        (LABELS_DIR,          "Labels directory"),
        (DATASET_PATH,        "Dataset YAML"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


# =========================================================
# TRAINING
# =========================================================

def run_training() -> list:
    """
    Train YOLO locally and return updated model parameters.
    Returns the list of NumPy arrays (model weights).
    """
    _verify_paths()

    num_images = _count_images(IMAGES_DIR)
    num_labels = _count_labels(LABELS_DIR)

    if num_images == 0:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")
    if num_labels == 0:
        raise RuntimeError(f"No label files found in {LABELS_DIR}")

    print(_sep("═", 60))
    print(f"  FL CLIENT  —  {CLIENT_ID.upper()}")
    print(_sep())
    print(f"  Dataset   : {DATASET_PATH}")
    print(f"  Images    : {num_images:,}  |  Labels : {num_labels:,}")
    print(f"  Epochs    : {LOCAL_EPOCHS}  |  Img size: {IMAGE_SIZE}")
    print(f"  Weights   : {WEIGHTS}")
    print(_sep("═", 60))

    yolo  = YOLO(WEIGHTS)
    model = yolo.model

    # Ensure all parameters are trainable
    for p in model.parameters():
        p.requires_grad = True

    print(f"\n  ► Training locally ({num_images:,} images × {LOCAL_EPOCHS} epoch(s))…\n")

    yolo.train(
        data=str(DATASET_PATH),
        epochs=LOCAL_EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(RESULTS_DIR),
        name=f"{CLIENT_ID}_training",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n  ✔  Local training complete.")

    # Extract and hash parameters
    params = get_parameters(model)
    model_hash = hash_parameters(params)

    print(f"  ✔  SHA-256 hash : {model_hash[:32]}…")
    print(_sep("═", 60) + "\n")

    # Persist the hash so --connect can reuse it
    hash_file = BASE_DIR / ".last_hash"
    hash_file.write_text(model_hash)

    return params, model_hash, num_images


# =========================================================
# FEDERATED CLIENT
# =========================================================

class FLClient(fl.client.NumPyClient):
    """Flower client that skips local training and sends pre-trained weights."""

    def __init__(self, parameters: list, model_hash: str, num_examples: int):
        self._parameters  = parameters
        self._model_hash  = model_hash
        self._num_examples = num_examples

    def get_parameters(self, config):
        return self._parameters

    def fit(self, parameters, config):
        # Server sends global weights but we send our local update directly.
        # Ignore the server's global model — we already trained locally.
        metrics = {
            "client_id":  CLIENT_ID,
            "model_hash": self._model_hash,
        }

        print(f"  ► Sending update to aggregator  ({self._num_examples:,} images)")
        print(f"    Hash : {self._model_hash[:32]}…")

        # Schedule shutdown after sending
        def _shutdown():
            time.sleep(2)
            os._exit(0)
        threading.Thread(target=_shutdown, daemon=True).start()

        return self._parameters, self._num_examples, metrics

    def evaluate(self, parameters, config):
        # Not performing evaluation — return neutral values
        return 0.0, self._num_examples, {}


# =========================================================
# CONNECT
# =========================================================

def run_connect(parameters: list, model_hash: str, num_examples: int):
    """Connect to the aggregator and send the trained model."""
    print(_sep("═", 60))
    print(f"  FL CLIENT  —  {CLIENT_ID.upper()}")
    print(_sep())
    print(f"  Aggregator: {SERVER_ADDRESS}")
    print(f"  Images    : {num_examples:,}")
    print(f"  Hash      : {model_hash[:32]}…")
    print(_sep("═", 60))
    print(f"\n  Connecting to aggregator at {SERVER_ADDRESS}…\n")

    client = FLClient(parameters, model_hash, num_examples)
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


# =========================================================
# ENTRYPOINT
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning Client 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (no flags)          Train then connect to the aggregator (default)
  --train             Local training only — does NOT connect to aggregator
  --connect           Connect and send the last trained model (skip training)
  --train --connect   Same as default: train then connect

All configuration defaults are loaded from the .env file.
        """
    )
    parser.add_argument("--train",   action="store_true", help="Run local YOLO training")
    parser.add_argument("--connect", action="store_true", help="Connect and send model to aggregator")
    return parser.parse_args()


def main():
    args = parse_args()

    # Default: both train + connect
    do_train   = args.train   or (not args.train and not args.connect)
    do_connect = args.connect or (not args.train and not args.connect)

    parameters  = None
    model_hash  = None
    num_examples = 0

    if do_train:
        parameters, model_hash, num_examples = run_training()
    
    if do_connect and not do_train:
        # Load a previously trained model to send
        yolo  = YOLO(WEIGHTS)
        model = yolo.model
        parameters   = get_parameters(model)
        model_hash   = hash_parameters(parameters)
        num_examples = _count_images(IMAGES_DIR)
        print(f"  ► Using current model weights (no training requested).")
        print(f"    Hash : {model_hash[:32]}…\n")

    if do_connect:
        run_connect(parameters, model_hash, num_examples)


if __name__ == "__main__":
    main()