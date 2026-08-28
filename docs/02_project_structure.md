# 02 — Project Structure

This page describes every file and directory in the repository and what it is responsible for.

---

## Full Directory Tree

```
FL-V1/
│
├── aggregator/                        ← Lives on the SERVER machine
│   ├── __init__.py                    ← Makes aggregator a Python package
│   ├── server.py                      ← Main FL server / aggregator
│   ├── security/
│   │   ├── __init__.py
│   │   └── hashing.py                 ← SHA-256 hash verification
│   ├── logs/                          ← Per-round JSON audit logs (auto-created)
│   └── results/                       ← Reserved for future model checkpoints
│
├── client_project/                    ← Copy this to EACH client machine
│   ├── client.py                      ← Universal FL client (all 3 clients use this)
│   ├── convert_annotations.py         ← VisDrone → YOLO label format converter
│   ├── yolo11n.pt                     ← Base YOLO11n weights (not in Git)
│   ├── security/
│   │   ├── __init__.py
│   │   └── hashing.py                 ← SHA-256 hashing (mirrors aggregator version)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_utils.py             ← get_parameters() / set_parameters()
│   └── data/
│       ├── client_template.yaml       ← Copy and rename for each client
│       └── client<N>/                 ← One folder per client (populated later)
│           ├── images/                ← Training images (~700 JPEGs)
│           ├── annotations/           ← Raw VisDrone annotation .txt files
│           ├── labels/                ← YOLO labels (created by convert_annotations.py)
│           └── client<N>.yaml         ← YOLO dataset config for this client
│
├── client1_project/                   ← Legacy single-client project (kept for reference)
│   ├── client1.py
│   ├── convert_client1.py
│   ├── test_train.py
│   ├── security/
│   │   ├── __init__.py
│   │   └── hashing.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_utils.py
│   └── data/client1/
│       ├── client1.yaml
│       ├── images/
│       ├── annotations/
│       ├── labels/
│       └── labels.cache
│
├── clients/                           ← Dataset shards (created by split_database.py)
│   ├── client1/
│   │   ├── images/                    ← ~⅓ of VisDrone training images
│   │   └── annotations/               ← Matching annotation files
│   ├── client2/
│   │   ├── images/
│   │   └── annotations/
│   └── client3/
│       ├── images/
│       └── annotations/
│
├── VisDrone2019-DET-train/            ← Full raw dataset (not in Git, ~2 GB)
│   ├── images/
│   └── annotations/
│
├── split_database.py                  ← Splits full dataset into 3 equal shards
├── VisDrone.yaml                      ← Ultralytics reference YAML (not used directly)
├── requirements.txt                   ← Python dependencies for the server
├── .gitignore                         ← Excludes weights, datasets, venvs, logs
└── docs/                              ← This documentation
    ├── README.md                      ← Documentation index
    ├── 01_project_overview.md
    ├── 02_project_structure.md        ← You are here
    ├── 03_setup_server.md
    ├── 04_setup_clients.md
    ├── 05_running.md
    ├── 06_configuration.md
    ├── 07_progress_and_logs.md
    └── 08_troubleshooting.md
```

---

## Key Files Explained

### `aggregator/server.py`

The central FL server. Responsibilities:
- Listens on `0.0.0.0:8080` for incoming client connections
- Implements `SecureFedAvg` — a custom Flower strategy that:
  - Tracks connected clients
  - Distributes the global model each round
  - Receives updated weights from all clients
  - Verifies SHA-256 hashes before accepting updates
  - Runs Federated Averaging on accepted updates
  - Saves a JSON log for every round
- Configurable via environment variables (no code edits needed)

### `aggregator/security/hashing.py`

Shared hashing logic used by the **server** to independently recalculate the SHA-256 hash of received parameters. Must be identical to the client-side version.

### `client_project/client.py`

Universal FL client script — one file works for all three client machines. It:
- Accepts `--server`, `--id`, `--data`, `--epochs`, `--imgsz`, `--weights` flags
- Falls back to environment variables if CLI args are not provided
- Loads YOLO model and validates dataset on startup
- Runs the full FL round loop: receive → train → hash → send
- Prints a 5-step progress indicator per round

### `client_project/convert_annotations.py`

Converts VisDrone raw annotation format to YOLO label format.

| Format | Description |
|--------|-------------|
| **VisDrone** | `x, y, width, height, score, class_id, truncation, occlusion` (absolute pixels, top-left corner) |
| **YOLO** | `class_id x_center y_center width height` (normalized 0–1, center coordinates) |

Must be run once per client before starting FL training.

### `client_project/utils/model_utils.py`

Two utility functions for parameter exchange:
- `get_parameters(model)` — extracts `state_dict` values as a list of NumPy arrays
- `set_parameters(model, parameters)` — loads a list of NumPy arrays back into a model's `state_dict`

### `split_database.py`

Run once on the server to divide the full `VisDrone2019-DET-train/` dataset into three equal shards stored in `clients/client1/`, `clients/client2/`, `clients/client3/`. Uses file-system copy, not symlinks.

---

## Data Flow

```
VisDrone2019-DET-train/
        │
        ▼  python split_database.py
clients/
  ├── client1/images/ + annotations/    ← shard 1
  ├── client2/images/ + annotations/    ← shard 2
  └── client3/images/ + annotations/    ← shard 3
        │
        │  (transfer to each client machine)
        ▼
client_project/data/clientN/
  ├── images/
  ├── annotations/
  └── labels/    ← python convert_annotations.py --client clientN
```

---

➡️ Next: [03 — Server Setup](./03_setup_server.md)
