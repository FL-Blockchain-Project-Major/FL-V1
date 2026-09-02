"""
Federated Learning Aggregator — server.py
==========================================
Run from the project root:

    python -m aggregator.server

All configuration is loaded from aggregator/.env (see that file for keys).
You can also override any value via environment variables at launch:

    FL_NUM_CLIENTS=2 FL_SERVER_ADDRESS=0.0.0.0:9000 python -m aggregator.server
"""

# Suppress framework noise before any imports
import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")

import hmac as _hmac
import json
import sys
import threading
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.server import start_server, ServerConfig

from .security.hashing import hash_parameters


# =========================================================
# CONFIGURATION  (overridable via environment variables)
# =========================================================

# Total unique clients expected. Server shuts down after all have submitted.
NUM_CLIENTS    = int(os.environ.get("FL_NUM_CLIENTS",    "3"))
SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080")
NUM_ROUNDS     = int(os.environ.get("FL_NUM_ROUNDS",     "3"))

LOG_DIR = Path("aggregator/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# HELPERS
# =========================================================

def _sep(char="═", width=68):
    return char * width


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _bytes_of(arrays) -> int:
    return sum(a.nbytes for a in arrays)


def hmac_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing-based hash attacks."""
    return _hmac.compare_digest(a.lower(), b.lower())


# =========================================================
# SPINNER
# =========================================================

class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str):
        self._label  = label
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._elapsed = 0.0

    def _spin(self):
        start = time.time()
        for frame in cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self._elapsed = time.time() - start
            sys.__stdout__.write(f"\r  {frame}  {self._label} … ({self._elapsed:.0f}s)")
            sys.__stdout__.flush()
            time.sleep(0.1)
        sys.__stdout__.write(f"\r  ✔  {self._label} ({self._elapsed:.1f}s)          \n")
        sys.__stdout__.flush()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# =========================================================
# STRATEGY
# =========================================================

class SecureFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg with:
      - Duplicate client rejection (same client_id refused per session)
      - HMAC-SHA256 hash verification of every update
      - Clean terminal output and JSON session logs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._round_start: float    = 0.0
        self._fit_spinner: Optional[Spinner] = None
        # Track which clients have already submitted successfully
        self._accepted_ids: set = set()

    # ── Prepare round ───────────────────────────────────────────────────
    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n{_sep()}")
        print(f"  ROUND {server_round}/{NUM_ROUNDS}  —  waiting for client(s)")
        print(_sep("─"))
        print(f"  Submitted so far : {len(self._accepted_ids)}/{NUM_CLIENTS}  "
              f"|  Remaining : {NUM_CLIENTS - len(self._accepted_ids)}")

        self._round_start = time.time()
        self._fit_spinner = Spinner("Waiting for client(s) to train")
        self._fit_spinner.start()

        return super().configure_fit(server_round, parameters, client_manager)

    # ── Aggregate updates ───────────────────────────────────────────────
    def aggregate_fit(self, server_round, results, failures):
        elapsed = time.time() - self._round_start

        if self._fit_spinner:
            self._fit_spinner.stop()
            self._fit_spinner = None

        print(f"\n{_sep()}")
        print(f"  ROUND {server_round}/{NUM_ROUNDS}  —  aggregating")
        print(_sep("─"))
        print(f"  Responses : {len(results)}  |  Failures : {len(failures)}  "
              f"|  Elapsed : {elapsed:.1f}s\n")

        accepted_results = []
        round_log = {
            "session_id": SESSION_ID,
            "round":      server_round,
            "timestamp":  datetime.now().isoformat(),
            "clients":    [],
        }

        for idx, (client_proxy, fit_res) in enumerate(results, 1):
            client_id     = fit_res.metrics.get("client_id", f"unknown_{idx}")
            reported_hash = fit_res.metrics.get("model_hash")
            num_examples  = fit_res.num_examples
            arrays        = parameters_to_ndarrays(fit_res.parameters)
            payload       = _human_bytes(_bytes_of(arrays))

            # ── Duplicate check ─────────────────────────────────────────
            if client_id in self._accepted_ids:
                print(f"  ⚠  {client_id:14s} | DUPLICATE — already submitted. Rejected, not counted.")
                round_log["clients"].append({
                    "client_id": client_id,
                    "status":    "rejected_duplicate",
                })
                continue

            # ── Hash verification ───────────────────────────────────────
            calculated_hash = hash_parameters(arrays)
            hash_ok = reported_hash is not None and hmac_compare(reported_hash, calculated_hash)
            status  = "accepted" if hash_ok else "rejected_hash_mismatch"
            sym     = "✔" if hash_ok else "✘"
            lbl     = "ACCEPTED" if hash_ok else "REJECTED (hash mismatch)"

            print(f"  {sym} {client_id:14s} | samples={num_examples:,}  "
                  f"payload={payload}  hash={calculated_hash[:16]}…  → {lbl}")

            if hash_ok:
                accepted_results.append((client_proxy, fit_res))
                self._accepted_ids.add(client_id)

            round_log["clients"].append({
                "client_id":       client_id,
                "num_examples":    num_examples,
                "payload_bytes":   _bytes_of(arrays),
                "reported_hash":   reported_hash,
                "calculated_hash": calculated_hash,
                "status":          status,
            })

        accepted = len(accepted_results)
        print(f"\n  Accepted this round : {accepted}  "
              f"|  Total session : {len(self._accepted_ids)}/{NUM_CLIENTS}")

        if accepted == 0:
            print("  ✘  No valid updates — global model unchanged.")
            round_log["aggregation"] = "skipped — no valid updates"
            self._save_log(round_log)
            return None, {}

        # ── FedAvg ─────────────────────────────────────────────────────
        print(f"\n  ► FedAvg on {accepted} update(s)…")
        with Spinner("FedAvg aggregation"):
            aggregated, aggregated_metrics = super().aggregate_fit(
                server_round, accepted_results, failures
            )

        total_time = time.time() - self._round_start
        round_log["aggregation"]     = "success" if aggregated else "failed"
        round_log["round_duration_s"] = round(total_time, 2)

        print(f"  Total round time : {total_time:.1f}s")
        print(_sep() + "\n")

        self._save_log(round_log)

        # Check if all clients are done
        if len(self._accepted_ids) >= NUM_CLIENTS:
            print(_sep())
            print(f"  🎉  ALL {NUM_CLIENTS} CLIENT(S) SUBMITTED — aggregation complete.")
            print(f"  Shutting down.")
            print(_sep() + "\n")
            os._exit(0)

        # Brief pause so clients can cleanly disconnect before the next round
        time.sleep(2)
        return aggregated, aggregated_metrics

    # ── Evaluation (minimal — just log) ────────────────────────────────
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        for _, eval_res in results:
            cid  = eval_res.metrics.get("client_id", "unknown")
            print(f"  eval  {cid}: loss={eval_res.loss:.4f}  samples={eval_res.num_examples:,}")
        return super().aggregate_evaluate(server_round, results, failures)

    # ── Persist round log ───────────────────────────────────────────────
    def _save_log(self, round_log: dict):
        log_file = LOG_DIR / f"session_{SESSION_ID}.json"
        if log_file.exists():
            try:
                data = json.loads(log_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"session_id": SESSION_ID, "rounds": []}
        else:
            data = {"session_id": SESSION_ID, "rounds": []}

        data["rounds"].append(round_log)
        log_file.write_text(json.dumps(data, indent=4), encoding="utf-8")
        print(f"  📄  Log → {log_file.name}")


# =========================================================
# STRATEGY INSTANCE
# =========================================================

strategy = SecureFedAvg(
    fraction_fit=1.0,
    min_fit_clients=1,        # Start as soon as any client connects
    min_available_clients=1,  # Do not wait for all clients simultaneously
    fraction_evaluate=0.0,
    min_evaluate_clients=0,
)


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    print("\n" + _sep())
    print("  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n")
    print(_sep())
    print(f"  Session     : {SESSION_ID}")
    print(f"  Address     : {SERVER_ADDRESS}")
    print(f"  FL rounds   : {NUM_ROUNDS}")
    print(f"  Clients expected : {NUM_CLIENTS}  (act independently)")
    print(f"  Logs        : {LOG_DIR.resolve()}")
    print(f"\n  Clients can connect at any time — no need to start simultaneously.")
    print(f"  Duplicate submissions are detected and rejected automatically.")
    print(f"\n  Waiting on {SERVER_ADDRESS}…")
    print(_sep() + "\n")

    start_server(
        server_address=SERVER_ADDRESS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
