import flwr as fl
from ultralytics import YOLO
from pathlib import Path

from utils.model_utils import get_parameters, set_parameters
from security.hashing import hash_parameters


# =========================================================
# CONFIGURATION
# =========================================================

CLIENT_ID = "client1"

# Path to Client 1's YOLO dataset YAML file
DATASET_PATH = "data/client1/client1.yaml"

# Number of epochs Client 1 trains during each FL round
LOCAL_EPOCHS = 1

# Image size for YOLO
IMAGE_SIZE = 640

# Change this to the IP address of the aggregator laptop
SERVER_ADDRESS = "192.168.1.10:8080"


# =========================================================
# HELPER FUNCTION
# =========================================================

def count_training_images():

    """
    Count the number of images used by Client 1.
    This number is sent to the aggregator for FedAvg.
    """

    image_dir = Path(
        "data/client1/images"
    )

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    total = 0

    for extension in extensions:
        total += len(
            list(image_dir.glob(extension))
        )

    return total


# =========================================================
# FEDERATED CLIENT
# =========================================================

class Client1(fl.client.NumPyClient):

    def __init__(self):

        print("\n" + "=" * 60)
        print("INITIALIZING CLIENT 1")
        print("=" * 60)

        # -------------------------------------------------
        # Load YOLOv11 Nano model
        # -------------------------------------------------

        self.yolo = YOLO("yolo11n.pt")

        # Access underlying PyTorch model
        self.model = self.yolo.model

        # Count local training samples
        self.num_examples = count_training_images()

        print(f"Client ID: {CLIENT_ID}")
        print(f"Local training images: {self.num_examples}")
        print("Client 1 initialized successfully.")


    # =====================================================
    # GET INITIAL MODEL PARAMETERS
    # =====================================================

    def get_parameters(self, config):

        print("\nCLIENT 1: Sending initial parameters")

        return get_parameters(
            self.model
        )


    # =====================================================
    # FEDERATED TRAINING
    # =====================================================

    def fit(self, parameters, config):

        print("\n" + "=" * 60)
        print("CLIENT 1: NEW FEDERATED ROUND")
        print("=" * 60)

        # -------------------------------------------------
        # STEP 1
        # Receive global model parameters from aggregator
        # -------------------------------------------------

        print(
            "\n[1] Receiving global model from aggregator..."
        )

        set_parameters(
            self.model,
            parameters
        )

        print("Global model loaded successfully.")


        # -------------------------------------------------
        # STEP 2
        # Train model on Client 1's LOCAL dataset
        # -------------------------------------------------

        print(
            "\n[2] Starting local YOLO training..."
        )

        self.yolo.train(
            data=DATASET_PATH,
            epochs=LOCAL_EPOCHS,
            imgsz=IMAGE_SIZE,

            # Output settings
            project="results",
            name=f"{CLIENT_ID}_training",

            exist_ok=True,

            # Optional settings
            verbose=True
        )

        print(
            "\nLocal training completed."
        )


        # -------------------------------------------------
        # STEP 3
        # Extract updated model parameters
        # -------------------------------------------------

        print(
            "\n[3] Extracting updated model parameters..."
        )

        updated_parameters = get_parameters(
            self.model
        )


        # -------------------------------------------------
        # STEP 4
        # Generate SHA-256 hash
        # -------------------------------------------------

        print(
            "\n[4] Generating SHA-256 hash..."
        )

        model_hash = hash_parameters(
            updated_parameters
        )

        print(
            f"\nCLIENT 1 MODEL HASH:\n{model_hash}"
        )


        # -------------------------------------------------
        # STEP 5
        # Send update back to aggregator
        # -------------------------------------------------

        print(
            "\n[5] Sending updated model to aggregator..."
        )

        metrics = {
            "client_id": CLIENT_ID,
            "model_hash": model_hash
        }

        return (
            updated_parameters,
            self.num_examples,
            metrics
        )


    # =====================================================
    # MODEL EVALUATION
    # =====================================================

    def evaluate(self, parameters, config):

        print("\nCLIENT 1: Evaluating received global model")

        # Load global parameters
        set_parameters(
            self.model,
            parameters
        )

        # For now evaluation is handled by the aggregator
        # using the common VisDrone validation dataset.

        loss = 0.0

        metrics = {}

        return (
            loss,
            self.num_examples,
            metrics
        )


# =========================================================
# START CLIENT
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("STARTING FEDERATED LEARNING CLIENT 1")
    print("=" * 60)

    client = Client1()

    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=client
    )