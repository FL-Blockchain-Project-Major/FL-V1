# 05 — Running the System

> **Prerequisite:** Complete [03 — Server Setup](./03_setup_server.md) and [04 — Client Setup](./04_setup_clients.md) first.

---

## Overview

```text
1. Start the aggregator on the server machine.
2. Start `client1`, `client2`, and `client3` whenever ready — no need to start simultaneously.
3. Each client trains locally, then sends its model to the aggregator.
4. The aggregator verifies the hash, aggregates, and logs the result.
5. Server and clients shut down automatically when all clients have submitted.
```

**Clients are fully independent.** The server accepts updates from any client at any time.
A client that has already submitted is rejected immediately, logged as
`rejected_duplicate`, and not counted again. The server continues through its
remaining rounds waiting for other unique clients.

---

## Part 1 — Start the Aggregator

From the project root (`FL-V1/`), with the virtual environment active:

```bash
python -m aggregator.server
```

You will see:
```text
════════════════════════════════════════════════════════════════════
  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n
════════════════════════════════════════════════════════════════════
  Session     : 20260902_192700
  Address     : 0.0.0.0:8080
  FL rounds   : 3
  Clients expected : 3  (act independently)
  Logs        : /home/sayam/Desktop/FL-V1/aggregator/logs

  Clients can connect at any time — no need to start simultaneously.
  Duplicate submissions are detected and rejected automatically.

  Waiting on 0.0.0.0:8080…
════════════════════════════════════════════════════════════════════
```

---

## Part 2 — Start the Clients

Navigate into each project folder and run its client script. For example:

### Default (train then connect)
```bash
cd client1_project/
python client1.py
```

### Flags

| Flag | Description |
|------|-------------|
| *(none)* | Train locally, then send model to aggregator (default) |
| `--train` | **Only** run local YOLO training — does NOT connect to aggregator |
| `--connect` | **Only** connect and send the current model weights (skip training) |
| `--train --connect` | Same as the default |

### Examples

```bash
# Train then connect (default)
python client1.py

# Train only (useful for testing locally without an aggregator)
python client1.py --train

# Connect and send a previously-trained model
python client1.py --connect

# Explicit train-then-connect
python client1.py --train --connect

# In separate terminals, repeat with client2_project/client2.py and
# client3_project/client3.py.
```

---

## Part 3 — What Happens During a Session

1. **Client connects** — the aggregator sends the current global model weights.
2. **Local training** — YOLO trains on the client's private dataset.
3. **Hash computed** — an HMAC-SHA256 digest is generated over all weight tensors.
4. **Update sent** — weights + hash are sent to the aggregator.
5. **Aggregator verifies** — hash is recalculated server-side and compared. Mismatches are rejected.
6. **Duplicate check** — if a client with the same `FL_CLIENT_ID` already submitted, the request is rejected immediately and not counted.
7. **FedAvg** — accepted updates are merged into the global model.
8. **Auto-shutdown** — client exits after sending; aggregator exits after all clients have submitted.

---

## Part 4 — Where Are the Results?

| Output | Location |
|--------|----------|
| Session log (JSON) | `aggregator/logs/session_<ID>.json` |
| YOLO training outputs | `client1_project/results/<CLIENT_ID>_training/` |

---

➡️ Next: [06 — Configuration](./06_configuration.md)
