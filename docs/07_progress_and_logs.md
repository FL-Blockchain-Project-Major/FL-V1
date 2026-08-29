# 07 — Progress and Logs

The system relies on a clean, centralized logging structure rather than messy terminal outputs.

## Terminal UI

We have intentionally disabled standard verbose libraries (like `flwr` telemetry, `grpc` warnings, and YOLO `tqdm` progress bars) to ensure your terminal interface is perfectly readable.

During execution, you will see a simple animated spinner:
`⠋ Waiting for client(s) to train … (45s)`

Once an operation completes, you get a clean summary:

```text
  [1/5] Global model received  (10.1 MB, 0.02s)
  [2/5] Local training  (100 images × 1 epoch(s))…
  [2/5] ✔  Training complete  (139.7s)
  [3/5] Params extracted  (499 tensors  10.1 MB)
  [4/5] Hash : 8129634f8cfaaaac9f5786845e63ae21…  (0.02s)
  [5/5] Sending update to aggregator  (10.1 MB)
────────────────────────────────────────────────────────────
  [client1] Round 0 complete  |  images=100  time=140s
  ✔  Task complete. Client will auto-disconnect.
```

## JSON Session Logs

The primary output of the Aggregator Server is a single JSON file generated in the `aggregator/logs/` directory.

The file is named `session_<TIMESTAMP>.json`. 

Instead of multiple files per round, this central log appends all accepted client metrics into an array. It looks like this:

```json
{
    "session_id": "20260830_002925",
    "rounds": [
        {
            "round": 1,
            "timestamp": "2026-08-30T00:32:00.123456",
            "clients": [
                {
                    "client_id": "client1",
                    "num_examples": 100,
                    "payload_bytes": 10559300,
                    "reported_hash": "8129634f8cfaaaac9f5786845e63ae21...",
                    "calculated_hash": "8129634f8cfaaaac9f5786845e63ae21...",
                    "status": "accepted"
                }
            ],
            "aggregation": "success",
            "round_duration_s": 140.5
        },
        {
            "round": 2,
            "timestamp": "2026-08-30T00:37:56.654321",
            "clients": [
                {
                    "client_id": "client2",
                    "num_examples": 100,
                    "payload_bytes": 10559300,
                    "reported_hash": "a4d3f34f8cfaaaac9f5786845e63b398...",
                    "calculated_hash": "a4d3f34f8cfaaaac9f5786845e63b398...",
                    "status": "accepted"
                }
            ],
            "aggregation": "success",
            "round_duration_s": 356.3
        }
    ]
}
```

This makes it incredibly easy to parse the results later with a simple Python data analysis script.

---

➡️ Next: [08 — Troubleshooting](./08_troubleshooting.md)
