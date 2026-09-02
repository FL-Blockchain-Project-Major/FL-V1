# 04 — Client Setup

This guide explains how to prepare any of `client1_project/`, `client2_project/`, or
`client3_project/` for training and submission.

---

## 1. Prerequisites

- Python 3.9, 3.10, or 3.11
- One client project folder on the client machine
- Baseline `yolo11n.pt` weights inside that project folder

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

Each client must have its own local dataset. Place it inside the matching project:

```text
clientN_project/
└── data/
  └── clientN/
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

Open the matching project `.env` file and fill in your values. The repository includes
matching templates for all three clients:

```dotenv
FL_CLIENT_ID=clientN
FL_SERVER_ADDRESS=192.168.1.5:8080   # Replace with actual server IP
FL_DATASET_YAML=data/clientN/clientN.yaml
FL_LOCAL_EPOCHS=1
FL_IMAGE_SIZE=640
FL_WEIGHTS=yolo11n.pt
FL_HASH_SECRET=change-me-to-a-strong-secret   # Must match aggregator
```

> `FL_HASH_SECRET` must be the same value set in `aggregator/.env`.
> `FL_CLIENT_ID` must be unique across all clients. A repeated ID is rejected by the
> aggregator and does not count toward the remaining clients.

To create all three datasets from the original VisDrone training directory, run from
the repository root:

```bash
python3 split_database.py
```

The splitter copies only images with matching annotations and converts each annotation
to one YOLO label file. It regenerates the generated image and label directories, so
rerunning it does not retain stale or duplicate label files.

Each client also needs the baseline weights file. The included local setup already has
`yolo11n.pt` in all three project folders; when preparing separate client machines,
copy that file into the matching project directory before running the client.

---

➡️ Next: [05 — Running the System](./05_running.md)
