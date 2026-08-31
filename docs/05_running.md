# 05 — Running the System

This guide covers the actual workflow used by this repository: local training or direct upload, then a single HTTP model submission.

> Prerequisite: ensure the Python dependencies are installed and the dataset YAML path is valid for your environment.

---

## Overview

```text
1. Start the aggregator server.
2. Run a client in train mode or connect mode.
3. The client trains locally if required.
4. The client uploads the `.pt` model to the aggregator via HTTP.
5. The server validates the hash and stores the model.
6. The server exits automatically once the expected number of valid uploads is reached.
```

**Important:** In this repo, the server is an upload-only aggregator. It does not coordinate rounds or merge models in real time.

---

## Part 1 — Start the Aggregator

From the project root:

```bash
python -m aggregator.server
```

Example output:
```text
  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n
  Upload port     : 8090  (HTTP — all interfaces)
  Expected clients: 3
  Models saved to : C:\...\aggregator\received_models

  ► Clients should connect using ONE of these IPs:
      --server 192.168.1.5:8090

  Waiting for client uploads…
```

---

## Part 2 — Run a Client

Use the recommended client script in the project root or inside `client_project/`.

### Train and upload in one step
```bash
python client_project/client.py --train --server 192.168.1.5:8090 --id client1 --data client_project/data/client1/client1.yaml
```

### Upload an already trained model
```bash
python client_project/client.py --connect --server 192.168.1.5:8090 --id client1 --data client_project/data/client1/client1.yaml --model client_project/results/client1_training/weights/best.pt
```

### Compatibility example
```bash
python client1_project/client1.py --connect --server 127.0.0.1:8090 --id client1 --data data/client1/client1.yaml --model results/client1_training/weights/best.pt
```

---

## Part 3 — What happens next

1. **Train mode:** the client reads the dataset and runs YOLO training locally.
2. **Connect mode:** the client finds the existing `.pt` file and uploads it directly.
3. **Upload:** the client sends the model file and metadata to the aggregator endpoint `/upload`.
4. **Validation:** the server computes the file hash and accepts or rejects it.
5. **Auto-shutdown:** once all expected clients have uploaded successfully, the server stops automatically.

---

## Where are the results?

| Output | Location |
|---|---|
| Aggregator session log | `aggregator/logs/session_<ID>.json` |
| Uploaded model files | `aggregator/received_models/` |
| Local client training results | `client_project/results/` or `client1_project/results/` |

---

➡️ Next: [06 — Configuration](./06_configuration.md)
