# 01 — Project Overview

## What this project does

This project is a lightweight federated-learning-style setup for training a YOLO object-detection model on distributed data without moving raw image files to a central location.

Each client keeps its own local dataset, trains a YOLO model locally, and then uploads the trained `.pt` file to a central HTTP aggregator. The aggregator verifies the SHA-256 hash, saves the accepted model, and waits until the expected number of valid clients have submitted their files.

## How the workflow works

### 1. Aggregator (server)
- Starts as an HTTP upload server on a configured port (default: `8090`).
- Exposes an upload endpoint at `/upload`.
- Receives model files from clients with metadata such as `client_id`, `epochs`, `num_examples`, and `model_hash`.
- Verifies the hash for integrity.
- Saves the accepted file under `aggregator/received_models/`.
- Automatically shuts down after all expected uploads have succeeded.

### 2. Client nodes
- Run independently on different machines or terminal windows.
- Option A: `--train` trains locally and automatically uploads the result.
- Option B: `--connect` uploads a model that has already been trained and saved on disk.
- Computes a SHA-256 hash for the model before uploading.
- Exits automatically after a successful upload.

### 3. Why this design
This setup is intentionally simple and reliable for local experimentation and small deployment tests. It does not require a full Flower-based training loop or a dynamic global-model merge pipeline; it focuses on a robust upload-and-verify workflow that is easy to debug.

### Cross-platform support
The stack is Python-based and works across:
- Windows 10 / 11
- macOS
- Linux

---

➡️ Next: [02 — Project Structure](./02_project_structure.md)
