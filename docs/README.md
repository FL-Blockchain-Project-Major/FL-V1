# FL-V1 Documentation Index

Welcome to the **Federated Learning — VisDrone / YOLO11n** project documentation.

---

## 📚 Documentation Files

| # | File | Contents |
|---|------|----------|
| 1 | [01_project_overview.md](./01_project_overview.md) | What this project is, architecture, tech stack, how FL works |
| 2 | [02_project_structure.md](./02_project_structure.md) | Directory layout, what every file does |
| 3 | [03_setup_server.md](./03_setup_server.md) | Setting up the aggregator / server machine |
| 4 | [04_setup_clients.md](./04_setup_clients.md) | Setting up each of the 3 client machines |
| 5 | [05_running.md](./05_running.md) | How to start the server and clients, connect them |
| 6 | [06_configuration.md](./06_configuration.md) | All environment variables, CLI flags, YAML options |
| 7 | [07_progress_and_logs.md](./07_progress_and_logs.md) | Understanding terminal output, progress bars, round logs |
| 8 | [08_troubleshooting.md](./08_troubleshooting.md) | Common errors and how to fix them |

---

## Quick-Start (TL;DR)

**Server machine:**
```bash
git clone <repo-url> && cd FL-V1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python split_database.py          # split dataset → 3 client shards
python -m aggregator.server       # start and wait for clients
```

**Each client machine:**
```bash
# copy client_project/ + data/clientN/ from server
cd client_project
python -m venv .venv && source .venv/bin/activate
pip install flwr ultralytics torch torchvision numpy Pillow tqdm
python convert_annotations.py --client clientN
python client.py --server <SERVER_IP>:8080 --id clientN --data data/clientN/clientN.yaml
```

---

> For the full step-by-step walkthrough start with [01_project_overview.md](./01_project_overview.md).
