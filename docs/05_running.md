# 05 — Running the System

This guide covers how to start the aggregator and clients, connect them, and monitor a full federated learning session.

> **Prerequisite:** Complete [03 — Server Setup](./03_setup_server.md) and [04 — Client Setup](./04_setup_clients.md) first.

---

## Overview

```text
1. Start the aggregator (server).
2. Start any client whenever ready.
3. The client trains locally (completely offline).
4. After training, the client connects to the server once to upload its .pt file and exits automatically.
5. The server automatically shuts down when all expected clients have submitted.
```

**Key behavior:** Clients are **fully independent**. The client trains locally first, then uploads. The server only waits for the uploads and does not coordinate training rounds.

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
  Upload port     : 8090  (HTTP — all interfaces)
  Expected clients: 3
  Models saved to : /home/sayam/Desktop/FL-V1/aggregator/received_models
  Log directory   : /home/sayam/Desktop/FL-V1/aggregator/logs

  ► Clients should connect using ONE of these IPs:
      --server <THIS_MACHINE_IP>:8090

  ► Workflow:  Client trains → connects → uploads .pt once → exits
  ► Server auto-shuts down after all 3 client(s) upload.

  Waiting for client uploads…
════════════════════════════════════════════════════════════════════
```

---

## Part 2 — Start the Clients

On **each client machine** (or separate terminal), activate the virtual environment and run the client script. Provide the Server IP, a unique ID, and the path to the YAML data config.

*(Replace `192.168.1.5:8090` with your actual server IP address)*

**Client 1:**
```bash
python client.py --server 192.168.1.5:8090 --id client1 --data data/client1/client1.yaml
```

**Client 2:**
```bash
python client.py --server 192.168.1.5:8090 --id client2 --data data/client2/client2.yaml
```

You will see:
```text
════════════════════════════════════════════════════════════
  FL CLIENT  —  CLIENT1
────────────────────────────────────────────────────────────
  Server   : 192.168.1.5:8090
  Dataset  : data/client1/client1.yaml
  Images   : 100
  Epochs   : 1 / round   |  Img size : 640
════════════════════════════════════════════════════════════
  ✔  Ready — connecting independently (no waiting for other clients)
════════════════════════════════════════════════════════════

  Connecting to aggregator at 192.168.1.5:8090…
```

---

## Part 3 — What Happens Next

Clients do not need to be started at the same time. The flow looks like this:

1. **Local Training:** The client starts training silently locally, completely disconnected from the server.
2. **Data Transmission:** After training finishes, the client connects to the server, uploads the trained `.pt` model file, and provides metadata (e.g., training time, dataset size, and a SHA-256 hash).
3. **Auto-Shutdown (Client):** The client logs `🎉 Training complete. Process exiting.` and terminates automatically.
4. **Validation:** The server accepts the upload, verifies the SHA-256 hash, and saves the file in `aggregator/received_models/`.
5. **Auto-Shutdown (Server):** Once all expected clients have submitted their models, the server logs `🎉 ALL 3 CLIENT(S) UPLOADED SUCCESSFULLY.` and terminates automatically.

---

## Where Are the Results?

| Output | Location |
|--------|----------|
| Central Aggregation Log | `aggregator/logs/session_<ID>.json` |
| Uploaded client models | `aggregator/received_models/` |
| Local client training results | `client_project/results/<client_id>_local/` |

---

➡️ Next: [06 — Configuration](./06_configuration.md)
