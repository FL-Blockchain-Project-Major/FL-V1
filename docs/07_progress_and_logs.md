# 07 — Progress Indicators & Logs

This page explains what every line of terminal output means on both the aggregator and the clients, and how to read the round log files.

---

## Aggregator Terminal Output

### Startup Banner

```
======================================================================
  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n
======================================================================
  Session ID      : 20260829_003900
  Server address  : 0.0.0.0:8080
  FL rounds       : 3
  Required clients: 3
  Log directory   : /home/sayam/Desktop/FL-V1/aggregator/logs

  Tip: override any setting via environment variables:
       FL_SERVER_ADDRESS  FL_NUM_ROUNDS  FL_MIN_CLIENTS

  Waiting for 3 client(s) to connect…
======================================================================
```

| Field | Meaning |
|-------|---------|
| `Session ID` | Unique timestamp-based ID for this run. All logs for this run use this prefix. |
| `Server address` | The interface and port the server is listening on. `0.0.0.0` means all network interfaces. |
| `FL rounds` | How many rounds of federated training will run. |
| `Required clients` | Server will not start Round 1 until this many clients are connected. |
| `Log directory` | Where round JSON logs will be saved. |

---

### Configure Fit (start of each round)

```
======================================================================
  ROUND 1/3  —  CONFIGURE FIT
======================================================================
  Connected clients : 3
  Required clients  : 3
  [████████████████████████████] 3/3 (100%)

  ► Distributing global model to all clients
    Parameters : 2,616,250
    Payload    : 9.9 MB
```

| Line | Meaning |
|------|---------|
| `Connected clients` | How many clients are currently online. |
| Progress bar | Visual indicator of connected vs required clients. |
| `Parameters` | Total number of trainable weights in the YOLO model. |
| `Payload` | Size of the model weights being sent to each client. |

---

### Aggregate Fit (after clients return)

```
======================================================================
  ROUND 1/3  —  AGGREGATE FIT
======================================================================
  Client training time : 142.3s
  Total responses      : 3
  Failures / timeouts  : 0

  ─── Client 1/3 : client1 ───
      Training samples : 2,157
      Parameters       : 2,616,250
      Payload received : 9.9 MB
      Reported hash    : a3f2b891c0d4e6f7…
      Server hash      : a3f2b891c0d4e6f7…
      ✔  HASH VERIFIED  →  UPDATE ACCEPTED

  ─── Client 2/3 : client2 ───
      Training samples : 2,157
      Parameters       : 2,616,250
      Payload received : 9.9 MB
      Reported hash    : d91e3a72b85cf104…
      Server hash      : d91e3a72b85cf104…
      ✔  HASH VERIFIED  →  UPDATE ACCEPTED

  ─── Client 3/3 : client3 ───
      Training samples : 2,157
      Payload received : 9.9 MB
      Reported hash    : ff2c9014e73ab891…
      Server hash      : 9e11bc77a2d34f08…
      ✘  HASH MISMATCH  →  UPDATE REJECTED

  ──────────────────────────────────────────────────
  ROUND 1 SUMMARY
  ──────────────────────────────────────────────────
  [████████████████████████████░░░░░░░░░░░░] 2/3 (67%)
  Accepted : 2  |  Rejected : 1
  Rejected clients : ['client3']

  ► Running FedAvg on 2 accepted update(s)…
  ✔  FedAvg complete (0.84s)
     Aggregated payload : 9.9 MB
  Total round time : 148.1s
  📄  Round log saved → aggregator/logs/session_20260829_003900_round_1.json
```

| Field | Meaning |
|-------|---------|
| `Client training time` | Time elapsed since the server sent the model until the last response arrived. |
| `Total responses` | Number of clients that replied in this round. |
| `Failures / timeouts` | Clients that connected but failed to return an update. |
| `Training samples` | Number of images this client trained on locally. |
| `Payload received` | Size of the weight update sent back from this client. |
| `Reported hash` | SHA-256 hash computed by the client before sending. |
| `Server hash` | SHA-256 hash independently computed by the server on receipt. |
| `✔ HASH VERIFIED` | Hashes match → update is safe and will be included in FedAvg. |
| `✘ HASH MISMATCH` | Hashes differ → update is dropped (corruption or tampering detected). |
| Progress bar (summary) | Accepted updates vs total received. |
| `FedAvg complete` | The averaging step finished successfully. |
| `Total round time` | Wall-clock time for the complete round (distribute → train → aggregate). |

---

## Client Terminal Output

### Startup

```
============================================================
  FEDERATED LEARNING CLIENT  —  CLIENT1
============================================================
  Server   : 10.5.70.249:8080
  Dataset  : data/client1/client1.yaml
  Epochs   : 1 per round
  Img size : 640
  Weights  : yolo11n.pt

  ► Loading YOLO model from yolo11n.pt…
  Training images  : 2,157

  ✔  client1 ready. Connecting to 10.5.70.249:8080…
============================================================
```

---

### Per-Round Progress (5 steps)

```
============================================================
  [client1]  ROUND 1  —  FIT
============================================================

  [1/5] Receiving global model from aggregator…
        ✔  Received  (9.9 MB, 0.12s)

  [2/5] Starting local YOLO training…
        Dataset : data/client1/client1.yaml
        Epochs  : 1
        Images  : 2,157
        ... YOLO output here ...
        ✔  Local training complete (140.2s)

  [3/5] Extracting updated model parameters…
        Tensors  : 297
        Payload  : 9.9 MB

  [4/5] Computing SHA-256 integrity hash…
        Hash     : a3f2b891c0d4e6f7c923b84d12f60a3e…  (0.11s)

  [5/5] Sending update to aggregator…
        Payload  : 9.9 MB
        ✔  Update ready for transmission
  ────────────────────────────────────────────────────────
  Round 1 complete  |  samples=2,157  |  time=140s
============================================================
```

| Step | Meaning |
|------|---------|
| `[1/5] Receiving` | Client received the global model from the server. Shows size and time. |
| `[2/5] Training` | Local YOLO training is running. YOLO's own progress bar appears here. |
| `[3/5] Extracting` | Model weights extracted from PyTorch into NumPy arrays for transmission. |
| `[4/5] Hashing` | SHA-256 computed over all weight arrays. |
| `[5/5] Sending` | Update is ready — Flower handles the actual gRPC transmission. |
| Final line | Quick summary: round number, sample count, training time. |

---

## Round Log Files

After every round, a JSON log is written to `aggregator/logs/`:

**Filename pattern:** `session_<SESSION_ID>_round_<N>.json`

**Example:** `aggregator/logs/session_20260829_003900_round_1.json`

```json
{
    "session_id": "20260829_003900",
    "round": 1,
    "timestamp": "2026-08-29T00:39:00.412387",
    "clients": [
        {
            "client_id": "client1",
            "num_examples": 2157,
            "payload_bytes": 10387500,
            "reported_hash": "a3f2b891c0d4e6f7c923b84d12f60a3e7814...",
            "calculated_hash": "a3f2b891c0d4e6f7c923b84d12f60a3e7814...",
            "status": "accepted"
        },
        {
            "client_id": "client2",
            "num_examples": 2157,
            "payload_bytes": 10387500,
            "reported_hash": "d91e3a72b85cf104...",
            "calculated_hash": "d91e3a72b85cf104...",
            "status": "accepted"
        },
        {
            "client_id": "client3",
            "num_examples": 2157,
            "payload_bytes": 10387500,
            "reported_hash": "ff2c9014e73ab891...",
            "calculated_hash": "9e11bc77a2d34f08...",
            "status": "rejected"
        }
    ],
    "aggregation": "success",
    "round_duration_s": 148.1
}
```

| Field | Meaning |
|-------|---------|
| `session_id` | Unique ID linking all rounds in one run |
| `round` | Which FL round this log is for |
| `timestamp` | When this round's aggregation finished |
| `clients[].num_examples` | Images this client trained on |
| `clients[].payload_bytes` | Weight update size in bytes |
| `clients[].reported_hash` | Hash the client computed and sent |
| `clients[].calculated_hash` | Hash the server computed on receipt |
| `clients[].status` | `"accepted"` or `"rejected"` |
| `aggregation` | `"success"`, `"failed"`, or `"failed — no valid updates"` |
| `round_duration_s` | Total wall-clock time for this round |

### Reading Logs

```bash
# View the latest round log (pretty-printed)
cat aggregator/logs/$(ls -t aggregator/logs/ | head -1) | python -m json.tool

# Check which clients were rejected across all rounds
grep -h '"status": "rejected"' aggregator/logs/*.json
```

---

## YOLO Training Outputs (on client)

YOLO saves training results to `client_project/results/<client_id>_round<N>/`:

```
results/
└── client1_round1/
    ├── weights/
    │   ├── best.pt         ← best checkpoint this round
    │   └── last.pt         ← final checkpoint
    ├── results.csv         ← per-epoch metrics
    ├── confusion_matrix.png
    └── labels.jpg
```

These are the **local** training outputs. The actual federated model lives on the server and is rebuilt each round via FedAvg.

---

➡️ Next: [08 — Troubleshooting](./08_troubleshooting.md)
