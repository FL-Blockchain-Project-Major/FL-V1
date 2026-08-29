# 05 — Running the System

This guide covers how to start the aggregator and clients, connect them, and monitor a full federated learning session.

> **Prerequisite:** Complete [03 — Server Setup](./03_setup_server.md) and [04 — Client Setup](./04_setup_clients.md) first.

---

## Overview

```
1. Start the aggregator (server)   ← listens, does NOT block on client count
2. Start any client whenever ready ← each client is fully independent
3. Training begins as soon as a client submits an update
4. Other clients can connect/send updates at any time
```

> **Key change:** Clients are now **fully independent**. The server does not wait
> for all clients before starting. Any client can send its update at any time,
> and aggregation happens immediately after each submission.

---

## Part 1 — Configuring the Number of Clients

Open `aggregator/server.py` and change the **single constant** at the top:

```python
# ── Change this to match your number of clients ──
NUM_CLIENTS = 3   # ← set to 1, 2, 3, or more
```

This is informational (shown in the banner). The server will accept updates
from any number of clients independently regardless of this value.

---

## Part 2 — Start the Aggregator

On the **server machine**, with the venv active:

```bash
cd /home/sayam/Desktop/FL-V1
source .venv/bin/activate
python -m aggregator.server
```

You will see:
```
════════════════════════════════════════════════════════════════════════
  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n
════════════════════════════════════════════════════════════════════════
  Session ID      : 20260829_221720
  Server address  : 0.0.0.0:8080
  FL rounds       : 3
  Expected clients: 3  (clients act independently)
  Log directory   : /home/sayam/Desktop/FL-V1/aggregator/logs

  ► Clients can connect and send updates independently.
    Training begins as soon as any client sends an update.

  Waiting for client(s) to connect on 0.0.0.0:8080…
════════════════════════════════════════════════════════════════════════
```

### Custom Settings (Optional)

```bash
FL_SERVER_ADDRESS=0.0.0.0:9000 \
FL_NUM_ROUNDS=5                \
FL_NUM_CLIENTS=5               \
python -m aggregator.server
```

---

## Part 3 — Start Each Client

On **each client machine**, with the venv active. Run from inside `client_project/`.

### Client 1:
```bash
cd ~/client_project
source .venv/bin/activate
python client.py \
    --server 10.5.70.249:8080 \
    --id     client1          \
    --data   data/client1/client1.yaml
```

### Client 2:
```bash
python client.py \
    --server 10.5.70.249:8080 \
    --id     client2          \
    --data   data/client2/client2.yaml
```

### Client 3:
```bash
python client.py \
    --server 10.5.70.249:8080 \
    --id     client3          \
    --data   data/client3/client3.yaml
```

> Replace `10.5.70.249` with your actual server IP.
> To find it: run `hostname -I | awk '{print $1}'` on the server machine.

### Using Environment Variables:
```bash
FL_SERVER_ADDRESS=10.5.70.249:8080 \
FL_CLIENT_ID=client1               \
FL_DATASET_YAML=data/client1/client1.yaml \
python client.py
```

---

## Part 4 — What Happens

Clients are **fully independent**. You do NOT need to start them all at once.

```
T=0s    Aggregator starts, waiting for connections

T=10s   Client 1 connects → immediately starts Round 1
        [client1] Training on 2157 images × 1 epoch…

T=45s   Client 1 finishes → sends update → aggregation happens
        ✔  client1: ACCEPTED | hash verified | FedAvg complete

T=90s   Client 2 connects (can be any time after client 1)
        [client2] Training on 2157 images × 1 epoch…

T=135s  Client 2 finishes → sends update → aggregation happens
        ✔  client2: ACCEPTED | hash verified | FedAvg complete

        (Client 3 can arrive whenever it's ready — no deadline)
```

> Each client triggers its own independent aggregation cycle.
> No client needs to wait for another to be running.

---

## Part 5 — Stopping the System

### Stop the aggregator:
Press `Ctrl + C` in the aggregator terminal.

If the port is stuck (from a previous run):
```bash
fuser -k 8080/tcp
```

### Stop a client:
Press `Ctrl + C` in the client's terminal.

---

## Part 6 — Restarting After a Failed Run

**On the server:**
```bash
fuser -k 8080/tcp
sleep 1
python -m aggregator.server
```

**On the clients:**
Simply run the `python client.py ...` command again.

---

## Where Are the Results?

| Output | Location |
|--------|----------|
| Per-round JSON logs | `aggregator/logs/session_<ID>_round_<N>.json` |
| YOLO training metrics | `client_project/results/<client_id>_round<N>/` |
| YOLO training weights | `client_project/results/<client_id>_round<N>/weights/` |

For details on reading logs and progress output, see [07 — Progress & Logs](./07_progress_and_logs.md).

---

➡️ Next: [06 — Configuration](./06_configuration.md)
