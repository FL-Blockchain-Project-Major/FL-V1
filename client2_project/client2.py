import flwr as fl
from ultralytics import YOLO
from pathlib import Path

from utils.model_utils import get_parameters, set_parameters
from security.hashing import hash_parameters
import threading
import time
import os


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data" / "client2"
DATASET_PATH = DATASET_DIR / "client2.yaml"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"


# =========================================================
# CONFIGURATION
# =========================================================

CLIENT_ID = "client2"

LOCAL_EPOCHS = 1
IMAGE_SIZE = 64

SERVER_ADDRESS = "10.5.70.249:8080"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def count_training_images():
    """Count images in Client 2's local dataset."""

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    total = 0

    for extension in extensions:
        total += len(list(IMAGES_DIR.glob(extension)))

    return total


def count_label_files():
    """Count YOLO label files."""

    if not LABELS_DIR.exists():
        return 0

    return len(list(LABELS_DIR.glob("*.txt")))


def count_model_parameters(model):
    """
    Count total and trainable model parameters.

    IMPORTANT:
    We use requires_grad, NOT grad.

    grad is normally None before training starts.
    """

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_params = total_params - trainable_params

    return (
        total_params,
        trainable_params,
        frozen_params
    )


# =========================================================
# FEDERATED CLIENT
# =========================================================

class Client2(fl.client.NumPyClient):

    def __init__(self):

        print("\n" + "=" * 60)
        print("INITIALIZING CLIENT 2")
        print("=" * 60)

        # -------------------------------------------------
        # DATASET PATHS
        # -------------------------------------------------

        print(f"\nProject directory:")
        print(BASE_DIR)

        print(f"\nDataset directory:")
        print(DATASET_DIR)

        print(f"\nImages directory:")
        print(IMAGES_DIR)

        print(f"\nLabels directory:")
        print(LABELS_DIR)

        print(f"\nDataset YAML:")
        print(DATASET_PATH)

        # -------------------------------------------------
        # VERIFY PATHS
        # -------------------------------------------------

        if not DATASET_DIR.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{DATASET_DIR}"
            )

        if not IMAGES_DIR.exists():
            raise FileNotFoundError(
                f"Images directory not found:\n{IMAGES_DIR}"
            )

        if not LABELS_DIR.exists():
            raise FileNotFoundError(
                f"Labels directory not found:\n{LABELS_DIR}"
            )

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found:\n{DATASET_PATH}"
            )

        # -------------------------------------------------
        # COUNT DATASET
        # -------------------------------------------------

        self.num_examples = count_training_images()
        self.num_labels = count_label_files()

        if self.num_examples == 0:
            raise RuntimeError(
                f"No images found in:\n{IMAGES_DIR}"
            )

        if self.num_labels == 0:
            raise RuntimeError(
                f"No YOLO label files found in:\n{LABELS_DIR}"
            )

        # -------------------------------------------------
        # DATASET INFORMATION
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("CLIENT 2 DATASET INFORMATION")
        print("=" * 60)

        print(
            f"Images found             : "
            f"{self.num_examples:,}"
        )

        print(
            f"YOLO label files         : "
            f"{self.num_labels:,}"
        )

        print("=" * 60)

        # -------------------------------------------------
        # LOAD YOLO MODEL
        # -------------------------------------------------

        print("\nLoading YOLOv11 Nano model...")

        self.yolo = YOLO("yolo11n.pt")

        # Underlying PyTorch model
        self.model = self.yolo.model

        # -------------------------------------------------
        # ENSURE PARAMETERS ARE TRAINABLE
        # -------------------------------------------------

        for parameter in self.model.parameters():
            parameter.requires_grad = True

        # -------------------------------------------------
        # COUNT MODEL PARAMETERS
        # -------------------------------------------------

        (
            self.total_params,
            self.trainable_params,
            self.frozen_params
        ) = count_model_parameters(self.model)

        # -------------------------------------------------
        # MODEL INFORMATION
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("CLIENT 2 MODEL INFORMATION")
        print("=" * 60)

        print(
            f"Client ID               : "
            f"{CLIENT_ID}"
        )

        print(
            f"Local training images   : "
            f"{self.num_examples:,}"
        )

        print(
            f"YOLO label files        : "
            f"{self.num_labels:,}"
        )

        print(
            f"Total model parameters  : "
            f"{self.total_params:,} "
            f"({self.total_params / 1_000_000:.2f}M)"
        )

        print(
            f"Trainable parameters    : "
            f"{self.trainable_params:,} "
            f"({self.trainable_params / 1_000_000:.2f}M)"
        )

        print(
            f"Frozen parameters       : "
            f"{self.frozen_params:,}"
        )

        print(
            f"Local epochs / round    : "
            f"{LOCAL_EPOCHS}"
        )

        print(
            f"Image size              : "
            f"{IMAGE_SIZE}"
        )

        print(
            f"Aggregator server       : "
            f"{SERVER_ADDRESS}"
        )

        print("=" * 60)

        print("\nClient 2 initialized successfully.")

    # =====================================================
    # GET INITIAL MODEL PARAMETERS
    # =====================================================

    def get_parameters(self, config):

        print("\n" + "=" * 60)
        print("CLIENT 2: CONNECTED TO AGGREGATOR")
        print("=" * 60)

        print(
            f"Aggregator server: "
            f"{SERVER_ADDRESS}"
        )

        print(
            "Successfully connected and "
            "sending initial model parameters..."
        )

        print(
            f"Parameters being sent: "
            f"{self.total_params:,}"
        )

        print("=" * 60)

        return get_parameters(self.model)

    # =====================================================
    # FEDERATED TRAINING
    # =====================================================

    def fit(self, parameters, config):

        print("\n" + "=" * 60)
        print("CLIENT 2: NEW FEDERATED TRAINING ROUND")
        print("=" * 60)

        # -------------------------------------------------
        # RECEIVE GLOBAL MODEL
        # -------------------------------------------------

        print("\n[1] Receiving global model...")

        set_parameters(
            self.model,
            parameters
        )

        print("Global model loaded successfully.")

        # -------------------------------------------------
        # TRAINING INFORMATION
        # -------------------------------------------------

        print("\n" + "-" * 60)
        print("LOCAL TRAINING INFORMATION")
        print("-" * 60)

        print(
            f"Client                  : {CLIENT_ID}"
        )

        print(
            f"Training images         : "
            f"{self.num_examples:,}"
        )

        print(
            f"Trainable parameters    : "
            f"{self.trainable_params:,}"
        )

        print(
            f"Local epochs            : "
            f"{LOCAL_EPOCHS}"
        )

        print(
            f"Image size              : "
            f"{IMAGE_SIZE}"
        )

        print("-" * 60)

        # -------------------------------------------------
        # LOCAL YOLO TRAINING
        # -------------------------------------------------

        print("\n[2] Starting local YOLO training...")

        self.yolo.train(
            data=str(DATASET_PATH),
            epochs=LOCAL_EPOCHS,
            imgsz=IMAGE_SIZE,

            project=str(
                BASE_DIR / "results"
            ),

            name=f"{CLIENT_ID}_training",

            exist_ok=True,

            verbose=True
        )

        print("\nLocal training completed.")

        # -------------------------------------------------
        # UPDATED PARAMETERS
        # -------------------------------------------------

        print(
            "\n[3] Extracting updated model parameters..."
        )

        updated_parameters = get_parameters(
            self.model
        )

        updated_count = len(updated_parameters)

        print(
            f"Parameter tensors extracted: "
            f"{updated_count}"
        )

        # -------------------------------------------------
        # HASH
        # -------------------------------------------------

        print(
            "\n[4] Generating SHA-256 hash..."
        )

        model_hash = hash_parameters(
            updated_parameters
        )

        print(
            "\nCLIENT 2 MODEL HASH:"
        )

        print(model_hash)

        # -------------------------------------------------
        # SEND TO AGGREGATOR
        # -------------------------------------------------

        print(
            "\n[5] Sending updated model "
            "parameters to aggregator..."
        )

        metrics = {
            "client_id": CLIENT_ID,
            "model_hash": model_hash,

            # Dataset information
            "num_images": self.num_examples,
            "num_labels": self.num_labels,

            # Model information
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,

            # Training information
            "local_epochs": LOCAL_EPOCHS
        }

        print(
            "\n" + "=" * 60
        )

        print(
            "CLIENT 2: MODEL UPDATE READY"
        )

        print(
            f"Images trained          : "
            f"{self.num_examples:,}"
        )

        print(
            f"Local epochs            : "
            f"{LOCAL_EPOCHS}"
        )

        print(
            f"Trainable parameters    : "
            f"{self.trainable_params:,}"
        )

        print(
            f"Model hash              : "
            f"{model_hash}"
        )

        print(
            "Sending update to aggregator..."
        )

        print("✔  Task complete. Client will auto-disconnect.")

        print("=" * 60)

        # Schedule auto-shutdown so the client stops after sending data
        def auto_shutdown():
            time.sleep(2)
            os._exit(0)

        threading.Thread(
            target=auto_shutdown,
            daemon=True
        ).start()

        return (
            updated_parameters,
            self.num_examples,
            metrics
        )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self, parameters, config):

        print(
            "\nCLIENT 2: Evaluating global model"
        )

        set_parameters(
            self.model,
            parameters
        )

        return (
            0.0,
            self.num_examples,
            {}
        )


# =========================================================
# START CLIENT
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("STARTING FEDERATED LEARNING CLIENT 2")
    print("=" * 60)

    print(
        f"\nAttempting connection to aggregator:"
    )

    print(
        f"SERVER: {SERVER_ADDRESS}"
    )

    client = Client2()

    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=client
    )