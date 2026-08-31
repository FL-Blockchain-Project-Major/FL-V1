# Federated Learning with YOLO

A lightweight, cross-platform client/server setup for training YOLO models on distributed datasets without sharing raw images.

This repository uses a simple HTTP upload workflow: each client trains locally, optionally connects to the aggregator, and submits the trained `.pt` file once. The aggregator verifies the hash and stores the accepted model.

## Features

- **Cross-platform:** Works on Windows, macOS, and Linux.
- **Explicit client modes:** `--train` for local training and `--connect` for uploading an already-trained model.
- **Independent clients:** Clients can run on any machine and connect to the server whenever they are ready.
- **Strong integrity checks:** SHA-256 hashing rejects tampered or corrupted model uploads.
- **Clean logging:** Each upload is recorded in a session JSON log and stored under `aggregator/received_models/`.
- **Simple server lifecycle:** The aggregator waits for the configured number of valid uploads, then exits automatically.

## Documentation Index

1. [Project Overview](./01_project_overview.md)
2. [Project Structure](./02_project_structure.md)
3. [Server Setup](./03_setup_server.md)
4. [Client Setup](./04_setup_clients.md)
5. [Running the System](./05_running.md)
6. [Configuration Guide](./06_configuration.md)
7. [Progress & Logs](./07_progress_and_logs.md)
8. [Troubleshooting](./08_troubleshooting.md)
