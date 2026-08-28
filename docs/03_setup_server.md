# 03 — Server / Aggregator Setup

This guide walks through setting up the **aggregator machine** — the central server that coordinates all federated learning rounds. Run these steps **only on the server machine**.

---

## Prerequisites

- Linux (Ubuntu 20.04+ recommended) or macOS
- Python 3.9 or higher
- `git` installed
- At least 4 GB free disk space (for dependencies + dataset shards)
- Connected to the same LAN as the client machines

---

## Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd FL-V1
```

If you don't have Git set up, you can copy the project folder to the machine directly.

---

## Step 2 — Create a Virtual Environment

Using a virtual environment isolates dependencies from your system Python.

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Your shell prompt should now show `(.venv)` at the start. You must have the venv active whenever you run any project commands.

To deactivate later:
```bash
deactivate
```

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `flwr` — Flower federated learning framework
- `ultralytics` — YOLO11n model
- `torch` + `torchvision` — PyTorch deep learning backend
- `numpy`, `Pillow`, `tqdm`, `opencv-python` — data utilities

> **Note:** PyTorch with CUDA support (~500 MB) will be downloaded automatically. This may take several minutes depending on your internet speed.

---

## Step 4 — Prepare the Dataset

### 4a. Obtain the VisDrone Training Dataset

The raw training set should be placed at:
```
FL-V1/VisDrone2019-DET-train/
├── images/       ← 6,471 JPEG images
└── annotations/  ← matching .txt files
```

If you don't have it, download it:
```bash
# Manual download from VisDrone GitHub releases
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-train.zip
```

Or using Ultralytics auto-download (requires the full YOLO pipeline):
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').train(data='VisDrone.yaml', epochs=0)"
```

### 4b. Split the Dataset into 3 Client Shards

Run the splitting script once:
```bash
python split_database.py
```

Expected output:
```
Splitting VisDrone dataset into 3 client shards...
Total images found: 6471
Client 1: 2157 images → clients/client1/
Client 2: 2157 images → clients/client2/
Client 3: 2157 images → clients/client3/
Done.
```

This creates:
```
clients/
├── client1/
│   ├── images/       ← ~2157 images
│   └── annotations/  ← matching annotation files
├── client2/
│   ├── images/
│   └── annotations/
└── client3/
    ├── images/
    └── annotations/
```

> You only need to run `split_database.py` **once**. After that the shards are ready to be transferred to client machines.

---

## Step 5 — Find Your Server's LAN IP Address

Run:
```bash
hostname -I | awk '{print $1}'
```

Example output:
```
10.5.70.249
```

**Write this IP down.** You will give it to all 3 client machines so they can connect.

---

## Step 6 — (Optional) Open Firewall Port

If your machine uses a firewall, allow inbound connections on port 8080:

```bash
# UFW (Ubuntu)
sudo ufw allow 8080/tcp
sudo ufw reload

# firewalld (RHEL/Fedora)
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# iptables (manual)
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

If `ufw` is not active, you can skip this step.

---

## Step 7 — Verify the Setup

Run a quick import check to confirm everything is installed correctly:

```bash
python -c "import flwr; import ultralytics; import torch; print('All dependencies OK')"
```

Expected output:
```
All dependencies OK
```

---

## Server is Ready

Your aggregator machine is now fully set up. To **start the server**, see [05 — Running](./05_running.md).

To configure the server (number of rounds, port, minimum clients), see [06 — Configuration](./06_configuration.md).

---

➡️ Next: [04 — Client Setup](./04_setup_clients.md)
