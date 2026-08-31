# 02 — Project Structure

The repository is divided into a server-side aggregator and one or more client-side training scripts.

## Directory layout

```text
FL-V1/
├── aggregator/
│   ├── server.py              # HTTP upload aggregator
│   ├── received_models/       # Accepted model files from clients
│   └── logs/                  # Session logs for accepted uploads
│
├── client_project/
│   ├── client.py              # Recommended client script (train or connect)
│   ├── yolo11n.pt             # Baseline YOLO weights
│   ├── data/                  # Dataset YAML and image folders
│   ├── security/              # Hashing helpers
│   ├── utils/                 # Model helpers
│   └── results/               # Local training outputs
│
├── client1_project/
│   ├── client1.py             # Client 1 example / compatibility script
│   ├── data/
│   ├── results/
│   └── security/
│
├── client2_project/
├── client3_project/
├── docs/
├── requirements.txt
├── split_database.py
├── verify_client_once.py
├── VisDrone.yaml
└── yolo11n.pt
```

### Key files explained

- **`aggregator/server.py`**: Runs the upload server, saves accepted model files, validates hashes, and exits after the expected number of valid uploads.
- **`client_project/client.py`**: Main client entry point. Use `--train` to train and upload locally or `--connect` to upload an existing trained model.
- **`client1_project/client1.py`**: Example client script for the first dataset and a compatibility/legacy path for this repo.
- **`requirements.txt`**: Python dependencies for the project.

## Deployment strategy

For a real setup, keep the server on one machine and run the client from a separate machine or terminal session.

- The **server machine** needs the `aggregator/` directory and Python dependencies.
- The **client machine** needs the client project folder and the YOLO dataset config.
- Local testing is easiest by opening multiple terminals in the same workspace.

---

➡️ Next: [03 — Server Setup](./03_setup_server.md)
