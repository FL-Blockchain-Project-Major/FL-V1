# 01 — Project Overview

## What Is This Project?

This project implements a **privacy-preserving Federated Learning (FL)** pipeline that trains a **YOLOv11-nano object detection model** on drone imagery from the **VisDrone 2019** dataset — spread across **3 physically separate machines** — without any machine ever sharing its raw images.

The key idea: instead of centralising data, we bring the model to the data. Each client trains locally, then sends only the model weight updates to a central aggregator.

---

## Why Federated Learning?

In traditional machine learning, all training data is collected on a single machine. This raises:

- **Privacy concerns** — raw data must leave each device
- **Bandwidth costs** — transmitting large datasets is expensive
- **Legal restrictions** — data sovereignty laws may prevent sharing

Federated Learning solves this by keeping data local. Only compressed model parameters (~10 MB) are ever transmitted, not images (which could be hundreds of GB).

---

## How It Works — Step by Step

```
Round N
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  AGGREGATOR
  ┌───────────────────────────────────────────────────────────┐
  │  1. Sends current global model to all 3 clients           │
  └───────────────────────────────────────────────────────────┘
                         ▼ (model weights, ~10 MB)
  CLIENT 1             CLIENT 2             CLIENT 3
  ┌────────────┐       ┌────────────┐       ┌────────────┐
  │ 2. Receives│       │ 2. Receives│       │ 2. Receives│
  │    model   │       │    model   │       │    model   │
  │            │       │            │       │            │
  │ 3. Trains  │       │ 3. Trains  │       │ 3. Trains  │
  │  on local  │       │  on local  │       │  on local  │
  │  ~700 imgs │       │  ~700 imgs │       │  ~700 imgs │
  │            │       │            │       │            │
  │ 4. Computes│       │ 4. Computes│       │ 4. Computes│
  │  SHA-256   │       │  SHA-256   │       │  SHA-256   │
  │  hash      │       │  hash      │       │  hash      │
  └────────────┘       └────────────┘       └────────────┘
                         ▲ (updated weights + hash)
  AGGREGATOR
  ┌───────────────────────────────────────────────────────────┐
  │  5. Verifies SHA-256 hash of each client's update         │
  │  6. Runs FedAvg → merges 3 updates into new global model  │
  │  7. Saves round log to aggregator/logs/                   │
  └───────────────────────────────────────────────────────────┘
                         → Repeat for next round
```

---

## Security Layer

Each client computes a **SHA-256 hash** of its weight tensors (covering shape, dtype, and raw bytes of every tensor) before sending. The aggregator independently recalculates the same hash on receipt. If they match, the update is accepted. If not, the update is silently dropped — protecting against:

- **Corrupted transmissions** (network errors)
- **Model poisoning attacks** (malicious weight injection)

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| FL Framework | [Flower (flwr)](https://flower.ai/) | Orchestrates rounds, gRPC transport |
| Object Detection | [Ultralytics YOLO11n](https://docs.ultralytics.com/) | Model architecture |
| Dataset | [VisDrone 2019 DET](https://github.com/VisDrone/VisDrone-Dataset) | Drone aerial imagery |
| Deep Learning | [PyTorch](https://pytorch.org/) | Tensor ops, model state dicts |
| Security | Python `hashlib` (SHA-256) | Weight integrity verification |
| Logging | JSON | Per-round audit trail |

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     AGGREGATOR (Server)                       │
│                      0.0.0.0:8080                             │
│                                                               │
│  SecureFedAvg strategy                                        │
│   • Hash verification                                         │
│   • Federated Averaging                                       │
│   • Round logging                                             │
└─────────────┬─────────────────────────────────────────────────┘
              │  gRPC over LAN (port 8080)
    ┌─────────┼──────────┐
    │         │          │
┌───▼────┐ ┌──▼────┐ ┌───▼───┐
│Client 1│ │Client2│ │Client3│
│~700 img│ │~700img│ │~700img│
│ YOLO   │ │ YOLO  │ │ YOLO  │
│ train  │ │ train │ │ train │
└────────┘ └───────┘ └───────┘
```

- **Protocol:** Flower gRPC (insecure mode — add TLS for production)
- **Default port:** `8080`
- **Training rounds:** 3 (configurable)
- **Model size:** ~10 MB (YOLO11n)

---

## Dataset — VisDrone 2019 DET

VisDrone is a large-scale drone-captured visual benchmark with **10 object classes**:

| ID | Class | ID | Class |
|----|---------|----|-------|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

The full training set (~6,400 images) is split equally across the 3 client machines by `split_database.py`.

---

## Federated Averaging (FedAvg)

FedAvg computes the **weighted average** of all client model updates, where the weight of each client's contribution is proportional to its number of training samples:

```
Global Model = Σ (nᵢ / N) × Local_Model_i
              for each accepted client i
              where N = total samples across all accepted clients
```

This ensures clients with more data have a proportionally stronger influence on the global model.

---

➡️ Next: [02 — Project Structure](./02_project_structure.md)
