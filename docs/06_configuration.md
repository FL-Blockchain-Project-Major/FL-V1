# 06 — Configuration Reference

All settings in this project can be controlled without editing any code. You can use either **environment variables** or **CLI flags** (for the client).

---

## Aggregator (`aggregator/server.py`)

Set these environment variables before running `python -m aggregator.server`.

| Environment Variable | Default | Type | Description |
|----------------------|---------|------|-------------|
| `FL_SERVER_ADDRESS` | `0.0.0.0:8080` | `str` | Interface and port the server listens on. Use `0.0.0.0` to accept connections from any IP on the machine. |
| `FL_NUM_ROUNDS` | `3` | `int` | Number of federated learning rounds to run before the server shuts down. |
| `FL_MIN_CLIENTS` | `3` | `int` | Minimum number of clients that must connect before Round 1 begins. |

### Examples

```bash
# Default (3 rounds, 3 clients, port 8080)
python -m aggregator.server

# 5 rounds on a different port, require only 2 clients
FL_SERVER_ADDRESS=0.0.0.0:9090 \
FL_NUM_ROUNDS=5                \
FL_MIN_CLIENTS=2               \
python -m aggregator.server

# Quick single-round test
FL_NUM_ROUNDS=1 FL_MIN_CLIENTS=1 python -m aggregator.server
```

---

## Client (`client_project/client.py`)

The client accepts **both CLI flags and environment variables**. CLI flags take priority over environment variables.

| CLI Flag | Env Variable | Default | Description |
|----------|-------------|---------|-------------|
| `--server` | `FL_SERVER_ADDRESS` | `localhost:8080` | IP and port of the aggregator server |
| `--id` | `FL_CLIENT_ID` | `client1` | Unique name for this client (used in logs and metrics) |
| `--data` | `FL_DATASET_YAML` | `data/client1/client1.yaml` | Path to the YOLO dataset YAML file |
| `--epochs` | `FL_LOCAL_EPOCHS` | `1` | Number of local training epochs per FL round |
| `--imgsz` | `FL_IMAGE_SIZE` | `640` | Input image size for YOLO training |
| `--weights` | `FL_WEIGHTS` | `yolo11n.pt` | Path to the YOLO model weights file |

### Examples

```bash
# Using CLI flags (recommended for clarity)
python client.py \
    --server 10.5.70.249:8080 \
    --id     client2          \
    --data   data/client2/client2.yaml \
    --epochs 3                \
    --imgsz  416

# Using environment variables (good for Docker / CI)
FL_SERVER_ADDRESS=10.5.70.249:8080 \
FL_CLIENT_ID=client3               \
FL_DATASET_YAML=data/client3/client3.yaml \
FL_LOCAL_EPOCHS=2                  \
python client.py

# Mix: env var for server, flags for the rest
FL_SERVER_ADDRESS=10.5.70.249:8080 \
python client.py --id client1 --data data/client1/client1.yaml
```

---

## Dataset YAML (`data/clientN/clientN.yaml`)

Each client's YAML tells YOLO where to find its images and labels.

```yaml
path: data/client1       # Root directory relative to where client.py is run

train: images            # Subdirectory containing training images
val:   images            # Using same set for validation (small dataset)

nc: 10                   # Number of object classes

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

> **Important:** `path:` is relative to the working directory where `python client.py` is run, not relative to the YAML file itself. Always run `client.py` from inside the `client_project/` directory.

---

## Annotation Converter (`convert_annotations.py`)

| CLI Flag | Env Variable | Default | Description |
|----------|-------------|---------|-------------|
| `--client` | `FL_CLIENT_ID` | `client1` | Which client's data to convert (e.g. `client1`, `client2`, `client3`) |

```bash
python convert_annotations.py --client client2
```

---

## Tuning Recommendations

### For faster testing / debugging

```bash
# Server: 1 round, 1 client minimum
FL_NUM_ROUNDS=1 FL_MIN_CLIENTS=1 python -m aggregator.server

# Client: 1 epoch, smaller image size
python client.py --server localhost:8080 --id client1 \
    --data data/client1/client1.yaml --epochs 1 --imgsz 320
```

### For production / full training

```bash
# Server: 10 rounds, all 3 clients
FL_NUM_ROUNDS=10 FL_MIN_CLIENTS=3 python -m aggregator.server

# Client: more local epochs, full image size
python client.py --server 10.5.70.249:8080 --id client1 \
    --data data/client1/client1.yaml --epochs 5 --imgsz 640
```

### Effect of increasing local epochs

| `--epochs` | Pros | Cons |
|-----------|------|------|
| 1 | More frequent aggregation, better convergence control | More communication rounds needed |
| 3–5 | Fewer FL rounds needed overall | Risk of client drift (diverging from global model) |
| 10+ | Very few rounds needed | High client drift, may harm global model quality |

For VisDrone with 3 clients, **1–3 epochs per round** with **5–10 FL rounds** is a good starting point.

---

➡️ Next: [07 — Progress & Logs](./07_progress_and_logs.md)
