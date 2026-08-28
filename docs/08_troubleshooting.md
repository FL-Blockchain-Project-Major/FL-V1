# 08 — Troubleshooting

Common errors and how to fix them.

---

## Server / Aggregator Errors

---

### `ModuleNotFoundError: No module named 'aggregator'`

**Cause:** Running `python aggregator/server.py` directly instead of as a module.

**Fix:** Always use the `-m` flag from the project root:
```bash
cd /home/sayam/Desktop/FL-V1
python -m aggregator.server
```

---

### `ModuleNotFoundError: No module named 'security'`

**Cause:** Running the aggregator with `python aggregator/server.py` (wrong way), which breaks relative imports.

**Fix:** Same as above — always use:
```bash
python -m aggregator.server
```

---

### `Port in server address 0.0.0.0:8080 is already in use`

**Cause:** A previous server process is still holding port 8080 (e.g. after a crash or Ctrl+C).

**Fix:**
```bash
# Kill whatever is holding port 8080
fuser -k 8080/tcp
sleep 1

# Now restart
python -m aggregator.server
```

To check what is using the port:
```bash
fuser 8080/tcp        # shows PID
lsof -i :8080         # shows full process info
```

---

### `FL_MIN_CLIENTS is set to X, but only Y clients are available`

**Cause:** Not enough clients connected before the timeout (if one is set).

**Fix:** Either start more clients, or lower `FL_MIN_CLIENTS`:
```bash
FL_MIN_CLIENTS=2 python -m aggregator.server
```

---

### `✘ ERROR: No valid updates received. Global model unchanged.`

**Cause:** All client updates failed hash verification in a round.

**Fix:**
1. Check client-side output for the hash values reported.
2. Check network stability — packet corruption can cause hash mismatches.
3. Ensure `security/hashing.py` on the client and server are **identical** (same hash function, same byte ordering).

---

## Client Errors

---

### `ModuleNotFoundError: No module named 'security'`

**Cause:** Running `python client.py` from the wrong directory.

**Fix:** Always run from inside `client_project/`:
```bash
cd client_project
python client.py ...
```

---

### `Dataset YAML not found: data/client1/client1.yaml`

**Cause:** The YAML config file was not created or is in the wrong place.

**Fix:**
```bash
# Check if it exists
ls data/client1/client1.yaml

# If missing, create it:
cat > data/client1/client1.yaml << 'EOF'
path: data/client1
train: images
val:   images
nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
EOF
```

---

### `yolo11n.pt not found`

**Cause:** The base model weights file is missing from `client_project/`.

**Fix:**
```bash
# Option A: download automatically
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Option B: copy from server
scp <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/client1_project/yolo11n.pt ./
```

---

### `No annotation files found in data/client1/annotations/`

**Cause:** Dataset shard was not correctly transferred, or was placed in the wrong directory.

**Fix:**
```bash
# Check what you have
ls data/client1/
# Should see: images/  annotations/

# If annotations/ is missing, re-transfer from server:
scp -r <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/clients/client1/ ./data/client1/
```

---

### `Connection refused` / `Failed to connect to server`

**Cause:** The aggregator is not running, or the IP/port is wrong.

**Fix:**
1. Confirm the server is running: you should see the waiting banner on the server machine.
2. Verify the IP is correct:
   ```bash
   # On server machine:
   hostname -I | awk '{print $1}'
   ```
3. Test connectivity from the client machine:
   ```bash
   ping <SERVER_IP>
   nc -zv <SERVER_IP> 8080     # test port reachability
   ```
4. If `nc` times out, the firewall is blocking port 8080:
   ```bash
   # On server machine:
   sudo ufw allow 8080/tcp
   ```

---

### `grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE`

**Cause:** Server went down while clients were training (network drop, server crash).

**Fix:** Restart the server first, then restart all clients. They will reconnect for the next round.

---

### `CUDA out of memory`

**Cause:** The GPU on the client machine does not have enough VRAM for the batch size at the given image size.

**Fix:**
```bash
# Reduce image size
python client.py ... --imgsz 416
# or
python client.py ... --imgsz 320
```

Or force CPU training (slower but always works):
```bash
FL_DEVICE=cpu python client.py ...
```

---

### Labels are empty / no detections during training

**Cause:** The annotation converter was not run, or ran with errors.

**Fix:**
```bash
# Run the converter
python convert_annotations.py --client client1

# Verify labels were created
ls data/client1/labels/ | head -5
wc -l data/client1/labels/*.txt | tail -1    # total lines across all labels
```

---

## Network / Connectivity

---

### Clients are on different subnets / can't ping each other

**Symptom:** `ping <SERVER_IP>` from client fails.

**Fix:** All machines must be on the **same LAN**. Connect them to the same router or switch. VPN-based solutions (WireGuard, ZeroTier) can bridge machines across the internet if needed.

---

### Training is very slow

**Possible causes and fixes:**

| Cause | Fix |
|-------|-----|
| Running on CPU with large images | Use `--imgsz 320` or `--imgsz 416` |
| Too many epochs per round | Reduce `--epochs` to `1` |
| Low RAM causing swapping | Reduce batch size (default in YOLO is auto) |
| Slow disk I/O for dataset | Move dataset to SSD if possible |

---

### Hash mismatch happening consistently on one client

**Cause:** The `hashing.py` on that client differs from the server's version.

**Fix:** Copy the canonical version from the server:
```bash
scp <SERVER_USER>@<SERVER_IP>:/home/sayam/Desktop/FL-V1/aggregator/security/hashing.py \
    ./security/hashing.py
```

Both files must be byte-for-byte identical.

---

## Dependency / Environment

---

### `ImportError: cannot import name 'X' from 'flwr'`

**Cause:** Wrong version of Flower installed.

**Fix:**
```bash
pip install --upgrade flwr
```

Check version:
```bash
python -c "import flwr; print(flwr.__version__)"
```

---

### `pip install` fails with compilation errors

**Cause:** Missing system build tools.

**Fix (Ubuntu/Debian):**
```bash
sudo apt-get install -y build-essential python3-dev
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### Virtual environment not activated

**Symptom:** `python: command not found` or wrong package versions.

**Fix:**
```bash
# Activate venv before running anything
source .venv/bin/activate   # Linux / macOS

# Confirm:
which python
# Should show: /path/to/FL-V1/.venv/bin/python
```

---

## Still Stuck?

1. Check the round log files in `aggregator/logs/` — they contain per-client status and hash values.
2. Run with verbose YOLO output by temporarily removing `verbose=False` from `client.py`.
3. Isolate the issue: test with `FL_MIN_CLIENTS=1 FL_NUM_ROUNDS=1` using a single client on the same machine as the server.

---

← Back to [README](./README.md)
