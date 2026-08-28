# 04 — Client Machine Setup

Repeat **every step** on each of the 3 client machines. The only differences between machines are the `--id` flag and the dataset shard they receive.

---

## Prerequisites

- Linux (Ubuntu 20.04+ recommended) or macOS
- Python 3.9 or higher
- Connected to the **same LAN** as the aggregator server
- The **server's LAN IP** (e.g. `10.5.70.249`) — obtained from the server machine

---

## Step 1 — Transfer the Client Project Folder

The `client_project/` folder (from the FL-V1 repo) must exist on each client machine.

### Option A — Copy from the Server via SCP

Run from the **client machine**:
```bash
# Replace <SERVER_USER> and <SERVER_IP> with actual values
scp -r <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/client_project ./client_project
```

### Option B — Git Clone

If the repo is pushed to GitHub/GitLab:
```bash
git clone <your-repo-url>
cd FL-V1/client_project
```

### Option C — USB / Shared Drive

Copy the `client_project/` folder manually via USB stick or shared network drive.

---

## Step 2 — Transfer This Client's Dataset Shard

Each client receives a different slice of the VisDrone dataset. From the server, the shards are in `clients/client1/`, `clients/client2/`, `clients/client3/`.

### From the client machine (pull from server):
```bash
cd client_project

# For Client 1:
scp -r <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/clients/client1/ ./data/client1/

# For Client 2:
scp -r <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/clients/client2/ ./data/client2/

# For Client 3:
scp -r <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/clients/client3/ ./data/client3/
```

### From the server machine (push to client):
```bash
# From server, push Client 1's shard to Client 1's machine:
scp -r clients/client1/ <CLIENT1_USER>@<CLIENT1_IP>:~/client_project/data/client1/
```

After transfer, verify:
```bash
ls client_project/data/client1/
# Expected: images/  annotations/
```

---

## Step 3 — Set Up Python Environment

```bash
cd client_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install flwr ultralytics torch torchvision numpy Pillow tqdm
```

> PyTorch download is ~500 MB. Allow several minutes.

---

## Step 4 — Get the YOLO Base Weights

The base model file `yolo11n.pt` must be inside `client_project/`.

### Option A — Copy from the server:
```bash
# From client machine:
scp <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/client1_project/yolo11n.pt ./yolo11n.pt
```

### Option B — Download automatically via Python:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
# This downloads yolo11n.pt from Ultralytics CDN
# Then move it into client_project/:
mv ~/.config/Ultralytics/yolo11n.pt ./yolo11n.pt
# Or it may be downloaded directly into the current directory
```

Verify it exists:
```bash
ls -lh yolo11n.pt
# Expected: around 5-6 MB
```

---

## Step 5 — Create the Dataset YAML

Each client needs a YAML file telling YOLO where its images and labels are.

```bash
# For Client 1 — create data/client1/client1.yaml
cat > data/client1/client1.yaml << 'EOF'
path: data/client1
train: images
val:   images
nc: 10
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
EOF
```

For **Client 2** — create `data/client2/client2.yaml` (change `path:` to `data/client2`):
```yaml
path: data/client2
train: images
val:   images
nc: 10
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

For **Client 3** — create `data/client3/client3.yaml` (change `path:` to `data/client3`).

Or use the template shortcut:
```bash
# Client 1
cp data/client_template.yaml data/client1/client1.yaml
sed -i 's/CLIENT_ID/client1/g' data/client1/client1.yaml

# Client 2
cp data/client_template.yaml data/client2/client2.yaml
sed -i 's/CLIENT_ID/client2/g' data/client2/client2.yaml

# Client 3
cp data/client_template.yaml data/client3/client3.yaml
sed -i 's/CLIENT_ID/client3/g' data/client3/client3.yaml
```

---

## Step 6 — Convert VisDrone Annotations to YOLO Format

VisDrone uses a different annotation format than YOLO. Run the converter once:

```bash
# Run from inside client_project/ with venv active
source .venv/bin/activate

# For whichever client this machine is:
python convert_annotations.py --client client1   # Client 1 machine
python convert_annotations.py --client client2   # Client 2 machine
python convert_annotations.py --client client3   # Client 3 machine
```

Expected output:
```
Found 2157 annotation files
Converting: 100%|████████████| 2157/2157 [02:15<00:00]

==================================================
Conversion complete for client1
  Converted : 2150
  Skipped   : 7 (no matching image)
  Labels in : /home/.../client_project/data/client1/labels
==================================================
```

After this step, `data/client1/labels/` should contain one `.txt` file per image.

---

## Step 7 — Verify Client Setup

Run a quick sanity check:
```bash
python -c "
import flwr, torch
from ultralytics import YOLO
from pathlib import Path

model = YOLO('yolo11n.pt')
assert Path('data/client1/client1.yaml').exists(), 'YAML missing'
assert Path('data/client1/labels').exists(), 'Labels missing'
print('Client setup verified OK')
"
```

---

## Client is Ready

All three machines should now have:

```
client_project/
├── client.py
├── convert_annotations.py
├── yolo11n.pt              ✔
├── security/
├── utils/
└── data/
    └── clientN/
        ├── images/         ✔ (~2157 files)
        ├── annotations/    ✔ (~2157 files)
        ├── labels/         ✔ (~2150 files, after conversion)
        └── clientN.yaml    ✔
```

To **start the clients and connect them to the server**, see [05 — Running](./05_running.md).

---

## Quick Reference — Setup Checklist

| Step | Task | Done? |
|------|------|-------|
| 1 | Copied `client_project/` to this machine | ☐ |
| 2 | Copied dataset shard (`clients/clientN/`) to `data/clientN/` | ☐ |
| 3 | Created and activated `.venv`, installed dependencies | ☐ |
| 4 | `yolo11n.pt` is in `client_project/` | ☐ |
| 5 | Created `data/clientN/clientN.yaml` | ☐ |
| 6 | Ran `python convert_annotations.py --client clientN` | ☐ |
| 7 | Verified setup passes the sanity check | ☐ |

---

➡️ Next: [05 — Running the System](./05_running.md)
