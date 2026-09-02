# 06 — Configuration

Configuration is split between `.env` files (for secrets and defaults) and environment variable overrides (for one-off changes at launch).

---

## Aggregator — `aggregator/.env`

| Key | Default | Description |
|-----|---------|-------------|
| `FL_NUM_CLIENTS` | `3` | Total unique clients expected. Server auto-shuts down after all submit. |
| `FL_SERVER_ADDRESS` | `0.0.0.0:8080` | Host and port the aggregator binds to. |
| `FL_NUM_ROUNDS` | `3` | Number of Flower federated rounds. |
| `FL_HASH_SECRET` | *(see note)* | Shared secret for HMAC-SHA256 hashing. **Must match clients.** |

> **Security:** Change `FL_HASH_SECRET` to a strong random string before deployment. The same value must be set in every client's `.env`. Never commit this value to version control.

### Overriding at launch
```bash
FL_NUM_CLIENTS=2 FL_SERVER_ADDRESS=0.0.0.0:9000 python -m aggregator.server
```

---

## Client — `client1_project/.env`

| Key | Default | Description |
|-----|---------|-------------|
| `FL_CLIENT_ID` | `client1` | Unique identifier for this client (used in logs and duplicate detection). |
| `FL_SERVER_ADDRESS` | `localhost:8080` | Aggregator address. Replace with the server IP for remote machines. |
| `FL_DATASET_YAML` | `data/client1/client1.yaml` | Path to the YOLO dataset config, relative to `client1_project/`. |
| `FL_LOCAL_EPOCHS` | `1` | YOLO training epochs per round. |
| `FL_IMAGE_SIZE` | `640` | Image size for YOLO training. |
| `FL_WEIGHTS` | `yolo11n.pt` | Baseline YOLO weights file inside `client1_project/`. |
| `FL_HASH_SECRET` | *(see note)* | Shared secret for HMAC-SHA256 hashing. **Must match aggregator.** |

---

## Client — Command-Line Flags

| Flag | Description |
|------|-------------|
| *(none)* | Train locally then connect to aggregator (default behavior) |
| `--train` | Run local YOLO training only — does not connect |
| `--connect` | Connect and send model to aggregator — skips training |
| `--train --connect` | Explicit train-then-connect (same as default) |

### Example: remote aggregator
```bash
# In client1_project/.env:
FL_SERVER_ADDRESS=192.168.1.5:8080
FL_HASH_SECRET=my-strong-random-secret

python client1.py
```

---

➡️ Next: [07 — Progress and Logs](./07_progress_and_logs.md)
