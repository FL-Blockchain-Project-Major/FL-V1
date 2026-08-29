# 02 — Project Structure

The repository is divided into two primary logical components: the **Server (Aggregator)** and the **Client**.

## Directory Layout

```text
FL-V1/
├── aggregator/              # Server-side logic
│   ├── server.py            # Main aggregator script
│   ├── security/            # Integrity and hashing tools
│   ├── logs/                # Centralized JSON session logs (auto-generated)
│   └── results/             # Saved global models
│
├── client_project/          # Client-side logic (Deploy this to edge nodes)
│   ├── client.py            # Main client script
│   ├── yolo11n.pt           # Initial baseline YOLO weights
│   ├── data/                # Local datasets (images and labels)
│   ├── utils/               # PyTorch/Flower conversion utilities
│   ├── security/            # Integrity and hashing tools
│   └── results/             # Local YOLO training outputs
│
├── client1_project/         # Legacy fallback / Demo client 1
├── client2_project/         # Legacy fallback / Demo client 2
├── client3_project/         # Legacy fallback / Demo client 3
│
├── docs/                    # Documentation (you are here)
├── requirements.txt         # Python dependencies
└── VisDrone.yaml            # Original dataset config
```

### Key Files Explained

- **`aggregator/server.py`**: The central nervous system. It listens for connections, merges weights, and logs progress.
- **`client_project/client.py`**: The script that runs on the client. It downloads the weights, trains locally, and uploads the results.
- **`yolo11n.pt`**: The starting weights. You must provide a baseline model so that all clients start from the same architecture.

## Deployment Strategy

In a real-world scenario, you do **not** copy the entire `FL-V1` folder to the edge devices. 

- The **Server Machine** only needs the `aggregator/` directory and `requirements.txt`.
- The **Client Machines** only need the `client_project/` directory and `requirements.txt`.

For local testing, you can run everything from the main `FL-V1` directory using different terminal windows.

---

➡️ Next: [03 — Server Setup](./03_setup_server.md)
