# 03 — Server Setup

This guide shows you how to set up the central Aggregator Server. The steps are provided for Windows, Linux, and macOS.

## 1. Prerequisites

- Python 3.9, 3.10, or 3.11 installed on your system.
- Network access (Port 8080 open if clients are on different machines).

## 2. Create a Virtual Environment

Open a terminal (or Command Prompt / PowerShell on Windows) and navigate to the project root (`FL-V1`).

**Linux / macOS:**
```bash
cd /path/to/FL-V1
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
cd C:\path\to\FL-V1
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

With the virtual environment active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes `flwr`, `ultralytics`, and `torch`. 

## 4. Verify Server Configuration

Open `aggregator/server.py` in a text editor and ensure the `NUM_CLIENTS` constant is set to the number of clients you intend to test with.

```python
# ── Change this single constant to match your number of clients ──
NUM_CLIENTS = 3
```

By default, the server binds to `0.0.0.0:8080`, meaning it will accept connections from any IP address on port 8080.

## 5. Get Your IP Address

If your clients are on different computers, they need to know the Server's IP address.

**Linux / macOS:**
```bash
hostname -I
# OR
ipconfig getifaddr en0
```

**Windows:**
```powershell
ipconfig
```
Look for the `IPv4 Address` under your active Wi-Fi or Ethernet adapter (e.g., `192.168.1.5`).

---

➡️ Next: [04 — Client Setup](./04_setup_clients.md)
