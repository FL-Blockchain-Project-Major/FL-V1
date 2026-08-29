# 05 — Running the System

This guide covers how to start the aggregator and clients, connect them, and monitor a full federated learning session.

> **Prerequisite:** Complete [03 — Server Setup](./03_setup_server.md) and [04 — Client Setup](./04_setup_clients.md) first.

---

## Overview

```text
1. Start the aggregator (server).
2. Start any client whenever ready.
3. Training begins as soon as a client connects.
4. The client automatically shuts down when finished.
5. The server automatically shuts down when all expected clients have submitted.
```

**Key behavior:** Clients are **fully independent**. The server does not wait for all clients to connect before starting. Any client can send its update at any time.

---

## Part 1 — Start the Aggregator

On the **server machine**, ensure your virtual environment is active, then run:

**Linux / macOS:**
```bash
python3 -m aggregator.server
```

**Windows:**
```powershell
python -m aggregator.server
```

You will see a pristine UI:
```text
════════════════════════════════════════════════════════════════════
  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n
════════════════════════════════════════════════════════════════════
  Session ID      : 20260830_002925
  Server address  : 0.0.0.0:8080
  FL rounds       : 3
  Expected clients: 3  (clients act independently)
  Log directory   : /home/sayam/Desktop/FL-V1/aggregator/logs

  ► Clients can connect and send updates independently.
    Training begins as soon as any client sends an update.

  Waiting for client(s) to connect on 0.0.0.0:8080…
════════════════════════════════════════════════════════════════════
```

---

## Part 2 — Start the Clients

On **each client machine** (or separate terminal), activate the virtual environment and run the client script. Provide the Server IP, a unique ID, and the path to the YAML data config.

*(Replace `192.168.1.5:8080` with your actual server IP address)*

**Client 1:**
```bash
python client.py --server 192.168.1.5:8080 --id client1 --data data/client1/client1.yaml
```

**Client 2:**
```bash
python client.py --server 192.168.1.5:8080 --id client2 --data data/client2/client2.yaml
```

You will see:
```text
════════════════════════════════════════════════════════════
  FL CLIENT  —  CLIENT1
────────────────────────────────────────────────────────────
  Server   : 192.168.1.5:8080
  Dataset  : data/client1/client1.yaml
  Images   : 100
  Epochs   : 1 / round   |  Img size : 640
════════════════════════════════════════════════════════════
  ✔  Ready — connecting independently (no waiting for other clients)
════════════════════════════════════════════════════════════

  Connecting to aggregator at 192.168.1.5:8080…
```

---

## Part 3 — What Happens Next

Clients do not need to be started at the same time. The flow looks like this:

1. **Client connects:** The server immediately sends the global weights.
2. **Local Training:** The client trains silently (TQDM progress bars are suppressed for a clean terminal). The server displays `Waiting for client(s) to train`.
3. **Data Transmission:** The client extracts weights, computes a SHA-256 hash, and sends the payload back to the server.
4. **Auto-Shutdown (Client):** The client logs `✔ Task complete. Client will auto-disconnect` and terminates automatically.
5. **Aggregation:** The server accepts the hash, performs FedAvg, logs the metrics, and increments its completed client counter.
6. **Auto-Shutdown (Server):** Once all expected clients have submitted, the server logs `🎉 ALL 3 CLIENT(S) HAVE SUCCESSFULLY COMPLETED TRAINING` and terminates automatically.

---

## Where Are the Results?

| Output | Location |
|--------|----------|
| Central Aggregation Log | `aggregator/logs/session_<ID>.json` |
| YOLO training metrics | `client_project/results/<client_id>_round<N>/` |
| Local client weights | `client_project/results/<client_id>_round<N>/weights/` |

---

➡️ Next: [06 — Configuration](./06_configuration.md)
