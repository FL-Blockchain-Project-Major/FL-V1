# 05 — Running the System

This guide covers how to start the aggregator and all three clients, connect them, and monitor a full federated learning session.

> **Prerequisite:** Complete [03 — Server Setup](./03_setup_server.md) and [04 — Client Setup](./04_setup_clients.md) first.

---

## Overview

The correct startup order is:

```
1. Start the aggregator (server)   ← blocks, waiting for clients
2. Start Client 1
3. Start Client 2
4. Start Client 3
          ↓
   FL training begins automatically once all 3 clients connect
```

Clients can be started in any order, and may connect at different times. The aggregator waits until all required clients are connected before starting Round 1.

---

## Part 1 — Start the Aggregator

On the **server machine**, with the venv active:

```bash
cd /home/sayam/Desktop/FL-V1
source .venv/bin/activate
python -m aggregator.server
```

You will see:
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

The server is now **listening on port 8080** for incoming client connections.

### Custom Settings (Optional)

```bash
# Change number of rounds, port, or minimum clients:
FL_SERVER_ADDRESS=0.0.0.0:9000 \
FL_NUM_ROUNDS=5                \
FL_MIN_CLIENTS=2               \
python -m aggregator.server
```

See [06 — Configuration](./06_configuration.md) for all options.

---

## Part 2 — Start Each Client

On **each client machine**, with the venv active. Run from inside `client_project/`.

### Client 1 Machine:
```bash
cd ~/client_project
source .venv/bin/activate

python client.py \
    --server 10.5.70.249:8080 \
    --id     client1          \
    --data   data/client1/client1.yaml
```

### Client 2 Machine:
```bash
python client.py \
    --server 10.5.70.249:8080 \
    --id     client2          \
    --data   data/client2/client2.yaml
```

### Client 3 Machine:
```bash
python client.py \
    --server 10.5.70.249:8080 \
    --id     client3          \
    --data   data/client3/client3.yaml
```

> Replace `10.5.70.249` with your actual server IP.  
> To find it: run `hostname -I | awk '{print $1}'` on the server machine.

### Using Environment Variables Instead of Flags:
```bash
FL_SERVER_ADDRESS=10.5.70.249:8080 \
FL_CLIENT_ID=client1               \
FL_DATASET_YAML=data/client1/client1.yaml \
python client.py
```

---

## Part 3 — What Happens Next

Once **all 3 clients** are connected, the aggregator automatically starts Round 1. No manual action is needed.

### Full Timeline

```
T=0s     Aggregator starts, waits for clients
T=10s    Client 1 connects
T=15s    Client 2 connects
T=20s    Client 3 connects  ← All clients connected: Round 1 begins

────────────────────── ROUND 1 ──────────────────────
T=20s    Server sends global model to all clients (~10 MB each)
T=21s    All clients receive model, begin local YOLO training
          [Client 1] Training on 2157 images × 1 epoch…
          [Client 2] Training on 2157 images × 1 epoch…
          [Client 3] Training on 2157 images × 1 epoch…
T=160s   Clients finish training, compute SHA-256 hash, send weights back
T=161s   Server receives 3 updates, verifies hashes
T=162s   FedAvg aggregation (~1s)
T=163s   Round 1 complete. Round log saved.

────────────────────── ROUND 2 ──────────────────────
T=163s   Server sends updated global model to all clients
          … (same cycle) …

────────────────────── ROUND 3 ──────────────────────
          … (same cycle) …

T=~490s  All 3 rounds complete. Aggregator shuts down.
          Clients disconnect automatically.
```

---

## Part 4 — Stopping the System

### Stop the aggregator:
Press `Ctrl + C` in the terminal where it is running.

If the port is stuck (from a previous run):
```bash
fuser -k 8080/tcp
```

### Stop a client:
Press `Ctrl + C` in the client's terminal.

---

## Part 5 — Restarting After a Failed Run

If a run was interrupted:

**On the server:**
```bash
# Kill any leftover process on the port
fuser -k 8080/tcp
sleep 1
# Restart
python -m aggregator.server
```

**On the clients:**
Simply run the `python client.py ...` command again — the client always starts from the current global model for each round.

---

## Part 6 — Running with Fewer Than 3 Clients (Testing)

You can test with 1 or 2 clients by lowering `FL_MIN_CLIENTS`:

```bash
# Server: require only 1 client
FL_MIN_CLIENTS=1 python -m aggregator.server

# Client 1 only:
python client.py --server 10.5.70.249:8080 --id client1 --data data/client1/client1.yaml
```

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
