# 06 — Configuration

The client and server can be configured through environment variables or command-line arguments.

## Server configuration

The server uses these environment variables:

| Variable | Default | Description |
|---|---|---|
| `FL_NUM_CLIENTS` | `3` | Total valid uploads required before shutdown |
| `FL_UPLOAD_PORT` | `8090` | HTTP port used by the aggregator |

Example:

```bash
FL_NUM_CLIENTS=2 FL_UPLOAD_PORT=8090 python -m aggregator.server
```

## Client configuration

The main client script supports the following arguments:

| Argument | Default | Description |
|---|---|---|
| `--train` | off | Train locally before uploading |
| `--connect` | off | Upload an existing trained model without retraining |
| `--model` | none | Path to a local `.pt` file used with `--connect` |
| `--server` | `localhost:8090` | Aggregator host and port |
| `--id` | `client1` | Client name used in logs |
| `--data` | `data/client1/client1.yaml` | Dataset YAML config |
| `--epochs` | `1` | Local YOLO epochs |
| `--imgsz` | `640` | YOLO image size |
| `--weights` | `yolo11n.pt` | Initial weights file |

### Examples

```bash
python client_project/client.py --train --server 192.168.1.5:8090 --id client1 --data client_project/data/client1/client1.yaml
```

```bash
python client_project/client.py --connect --server 192.168.1.5:8090 --id client1 --data client_project/data/client1/client1.yaml --model client_project/results/client1_training/weights/best.pt
```

---

➡️ Next: [07 — Progress and Logs](./07_progress_and_logs.md)
