# 06 — Configuration

You can customize the behavior of the Server and the Clients easily without modifying the Python source code. You can use **Environment Variables** or **Command-Line Arguments**.

## Server Configuration

Open `aggregator/server.py` and modify the `NUM_CLIENTS` constant at the top of the file to change the total expected number of clients before the server auto-shuts down.

```python
NUM_CLIENTS = 3 
```

You can also override settings via environment variables when launching the server:

**Linux / macOS:**
```bash
FL_NUM_CLIENTS=5 FL_SERVER_ADDRESS=0.0.0.0:9000 python3 -m aggregator.server
```

**Windows (PowerShell):**
```powershell
$env:FL_NUM_CLIENTS="5"; $env:FL_SERVER_ADDRESS="0.0.0.0:9000"; python -m aggregator.server
```

## Client Configuration

Clients use command-line arguments for configuration.

### Available Arguments

| Argument | Default | Description |
|---|---|---|
| `--server` | `localhost:8090` | IP and port of the aggregator |
| `--id` | `client1` | Unique ID for the client (used for logging) |
| `--data` | `data/client1/client1.yaml` | Path to the YAML dataset file |
| `--epochs` | `1` | Number of YOLO training epochs |
| `--imgsz` | `640` | Image size for YOLO |
| `--weights` | `yolo11n.pt` | Path to the initial YOLO weights |

### Example Client Command

```bash
python client.py \
    --server 192.168.1.5:8090 \
    --id my_custom_client \
    --data data/custom/dataset.yaml \
    --epochs 5 \
    --imgsz 320
```

---

➡️ Next: [07 — Progress and Logs](./07_progress_and_logs.md)
