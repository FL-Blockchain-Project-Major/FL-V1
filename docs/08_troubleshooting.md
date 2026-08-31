# 08 — Troubleshooting

If you encounter issues, these are the most common fixes.

### 1. "Could not find trained weights" on the client
```text
✘  Could not find trained weights in results\client1_local\weights/
```
**Cause:** The client is looking in the wrong training output directory.
**Fix:** Use `--connect` with an explicit `--model` path, or make sure the YOLO run was saved in a location the client can find. The repo commonly produces files such as:
- `client_project/results/client1_training/weights/best.pt`
- `client1_project/results/client1_training/weights/best.pt`

### 2. Client cannot reach the aggregator
```text
✘  Could not connect to http://192.168.1.5:8090/upload
```
**Cause:** The server is not running or the IP/port is wrong.
**Fix:**
- Start the aggregator with `python -m aggregator.server`.
- Confirm the correct IP and port.
- Check firewall settings on Windows.

### 3. Server rejects the upload
```text
Hash check   : ✘  REJECTED (hash mismatch)
```
**Cause:** The uploaded model file is corrupted or changed after hashing.
**Fix:** Train again or resend the same file. The server is intentionally strict about file integrity.

### 4. Port already in use
**Cause:** Another instance of the server is still running.
**Fix:**
- **Windows:** `netstat -ano | findstr :8090` and then `taskkill /PID <PID> /F`
- **Linux/macOS:** `fuser -k 8090/tcp`

### 5. Local training runs but no file is created
**Cause:** The project is using a different YOLO run folder than the default fallback path.
**Fix:** Use the connect flow with a known file path:
```bash
python client_project/client.py --connect --server 127.0.0.1:8090 --id client1 --data client_project/data/client1/client1.yaml --model client_project/results/client1_training/weights/best.pt
```

### 6. Wrong working directory
**Cause:** The client expects the dataset and project paths to resolve relative to the project root or the script directory.
**Fix:** Run the commands from the repository root or use full absolute paths for `--data` and `--model`.
