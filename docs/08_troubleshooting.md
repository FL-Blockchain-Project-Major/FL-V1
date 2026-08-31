# 08 — Troubleshooting

If you encounter issues, here are the most common solutions.

### 1. "Dataset YAML not found" Error on Client
```text
✘ Dataset YAML not found: data/client1/client1.yaml
```
**Cause:** The client cannot find the YOLO configuration file.
**Fix:** Check that your `--data` parameter is pointing to the correct relative or absolute path. Ensure you are running the client script from *inside* the `client_project` directory.

### 2. Client Hangs on "Connecting to aggregator"
```text
Connecting to aggregator at 192.168.1.5:8090…
```
**Cause:** The client cannot reach the server IP.
**Fix:**
- Verify the IP address is correct.
- Ensure the Server script is actively running and waiting for clients.
- (Windows) Ensure your firewall isn't blocking Python network connections on port 8090.
- Try pinging the server from the client machine: `ping 192.168.1.5`.

### 3. Server Says "HASH MISMATCH → REJECTED"
```text
✘ client1 | hash=a4d3f34f… → REJECTED
```
**Cause:** The model weights were corrupted during network transfer.
**Fix:** This is an automatic security feature. The server will reject the bad weights. Simply run the client script again.

### 4. Port Already in Use (Address already in use)
**Cause:** An old instance of the server is still running in the background.
**Fix:**
- **Linux/macOS:** `fuser -k 8090/tcp`
- **Windows:** 
  1. Open PowerShell as Admin.
  2. `netstat -ano | findstr :8090` to find the PID.
  3. `taskkill /PID <PID> /F`

### 5. Out of Memory (OOM) Errors
**Cause:** The GPU or RAM is full during training.
**Fix:**
- Reduce the batch size (pass `--batch 4` or similar to the YOLO arguments if modified in code).
- Reduce `--imgsz` to 320 instead of 640.
- Ensure clients are running independently instead of concurrently on the same machine.
