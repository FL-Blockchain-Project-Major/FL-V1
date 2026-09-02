# 04 — Client Setup

This guide explains how to prepare `client1_project/` for training and submission.

---

## 1. Prerequisites

- Python 3.9, 3.10, or 3.11
- `client1_project/` folder on the client machine
- Baseline `yolo11n.pt` weights inside `client1_project/`

---

## 2. Virtual Environment

From inside `client1_project/`:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Data Preparation

Each client must have its own local dataset. Place it inside `client1_project/data/`:

```text
client1_project/
└── data/
    └── client1/
        ├── images/         # Training images (.jpg, .png)
        ├── labels/         # YOLO-format annotation files (.txt)
        └── client1.yaml    # Dataset configuration
```

Example `client1.yaml`:
```yaml
path: /absolute/path/to/client1_project/data/client1
train: images
val: images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

> Use the **absolute path** for `path:` — relative paths can cause Ultralytics resolution errors.

---

## 4. Configure the Client

Open `client1_project/.env` and fill in your values:

```dotenv
FL_CLIENT_ID=client1
FL_SERVER_ADDRESS=192.168.1.5:8080   # Replace with actual server IP
FL_DATASET_YAML=data/client1/client1.yaml
FL_LOCAL_EPOCHS=1
FL_IMAGE_SIZE=64
FL_WEIGHTS=yolo11n.pt
FL_HASH_SECRET=change-me-to-a-strong-secret   # Must match aggregator
```

> `FL_HASH_SECRET` must be the same value set in `aggregator/.env`.
> `FL_CLIENT_ID` must be unique across all clients — duplicate submissions are rejected by the aggregator.

---

➡️ Next: [05 — Running the System](./05_running.md)
