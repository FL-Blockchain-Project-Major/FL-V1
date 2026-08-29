# 04 — Client Setup

This guide explains how to prepare edge devices (clients) for training. These steps apply whether you are running multiple clients on different physical computers or simulating them in multiple terminal windows on the same computer.

## 1. Prerequisites

- Python 3.9, 3.10, or 3.11 installed.
- The `client_project` folder copied to the client machine.
- A baseline `yolo11n.pt` model file placed inside the `client_project` directory.

## 2. Virtual Environment

Open a terminal/command prompt and navigate to the `client_project` folder.

**Linux / macOS:**
```bash
cd /path/to/client_project
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
cd C:\path\to\client_project
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data Preparation

Each client must have its own subset of the data. The data should be organized inside the `client_project/data/` folder.

A typical client dataset structure looks like this:
```text
client_project/
└── data/
    └── client1/
        ├── images/         # Local images (.jpg, .png)
        ├── labels/         # YOLO-format text files (.txt)
        └── client1.yaml    # Dataset configuration file
```

The `client1.yaml` file tells YOLO where to find the data and the class names. An example:
```yaml
path: /absolute/path/to/client_project/data/client1
train: images
val: images

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
```

*Note: Ensure the `path` variable inside the `.yaml` file is an absolute path to avoid Ultralytics resolution errors.*

---

➡️ Next: [05 — Running the System](./05_running.md)
