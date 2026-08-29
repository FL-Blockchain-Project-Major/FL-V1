# 01 — Project Overview

## What is Federated Learning?

Federated Learning (FL) is a machine learning approach where a **central aggregator** (the server) manages a global model, and multiple **edge devices** (the clients) train that model on their own local data. 

Instead of sending raw images to the central server (which can be a massive privacy risk and consume huge bandwidth), clients only send the **updated model weights** back to the server. The server then averages these weights and creates an improved global model.

## How This Project Works

This project trains a YOLOv11 Nano model using the VisDrone dataset across distributed clients. 

### Core Mechanics
1. **Aggregator (Server):** 
   - Starts up and waits for clients to connect.
   - Accepts incoming connections at any time (asynchronous).
   - Tracks the number of successfully accepted updates.
   - Merges client updates via the **Federated Averaging (FedAvg)** strategy.
   - Automatically shuts down once it receives updates from all expected clients.
   - Creates a single `session_<ID>.json` log containing all metrics.

2. **Client Nodes:**
   - Run independently on different machines (or different terminal windows).
   - Download the global model from the server.
   - Train the model on their private portion of the dataset.
   - Compute a SHA-256 hash for security verification.
   - Send the updated model weights and hash back to the server.
   - **Automatically disconnect and shut down** upon successful transmission.

### Cross-Platform Support
This architecture is purely Python-based and uses HTTP/gRPC. It supports any operating system:
- Windows 10 / 11
- macOS (Intel & Apple Silicon)
- Linux (Ubuntu, Debian, Fedora, etc.)

---

➡️ Next: [02 — Project Structure](./02_project_structure.md)
