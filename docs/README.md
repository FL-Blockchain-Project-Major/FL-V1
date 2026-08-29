# Federated Learning with YOLO & Flower

A streamlined, cross-platform Federated Learning (FL) system using [Flower](https://flower.ai/) and [YOLOv11](https://github.com/ultralytics/ultralytics) for object detection.

This project enables multiple independent clients to collaboratively train a global YOLO model without ever sharing their raw image data.

## Features

- **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
- **Independent Clients:** Clients connect at any time, perform local training, send updates to the server, and automatically shut down. 
- **Auto-Shutdown Server:** The server automatically tracks how many updates it has received. Once the expected number of clients submit their data, it aggregates the final global model and shuts down.
- **Centralized Logging:** The server outputs a single, clean JSON log file summarizing the performance, hashes, and timing of all clients in the session.
- **Integrity Verification:** Uses SHA-256 hashing to ensure models sent over the network haven't been tampered with.
- **Clean Terminal UI:** Muted unnecessary YOLO progress bars, providing a pristine command-line interface.

## Documentation Index

Read the documentation in the following order to understand and deploy the project:

1. [Project Overview](./01_project_overview.md)
2. [Project Structure](./02_project_structure.md)
3. [Server Setup](./03_setup_server.md)
4. [Client Setup](./04_setup_clients.md)
5. [Running the System](./05_running.md)
6. [Configuration Guide](./06_configuration.md)
7. [Progress & Logs](./07_progress_and_logs.md)
8. [Troubleshooting](./08_troubleshooting.md)
