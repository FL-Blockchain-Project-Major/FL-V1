# 03 — Server Setup

This guide shows how to set up the central Aggregator on the server machine.

---

## 1. Prerequisites

- Python 3.9, 3.10, or 3.11
- Network access — clients need to reach the server on the configured port (default `8080`)

---

## 2. Create a Virtual Environment

From the project root (`FL-V1/`):

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure the Aggregator

Open `aggregator/.env` and set the values for your environment:

```dotenv
FL_NUM_CLIENTS=3            # How many unique clients will submit
FL_SERVER_ADDRESS=0.0.0.0:8080
FL_NUM_ROUNDS=3
FL_HASH_SECRET=change-me-to-a-strong-secret   # Must match all clients
```

> **Important:** `FL_HASH_SECRET` must be identical on the server and every client. It is used for HMAC-SHA256 integrity verification.

---

## 5. Find Your Server IP

Clients on other machines need your server's IP address.

**Linux / macOS:**
```bash
hostname -I
```

**Windows:**
```powershell
ipconfig
# Look for IPv4 Address under your active adapter
```

Set `FL_SERVER_ADDRESS` in each client's `.env` to `<your-ip>:8080`.

---

➡️ Next: [04 — Client Setup](./04_setup_clients.md)
