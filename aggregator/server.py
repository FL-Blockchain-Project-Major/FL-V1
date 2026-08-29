"""
Federated Learning Aggregator — server.py
==========================================
Run from the project root:
    python -m aggregator.server

Override defaults with environment variables:
    FL_SERVER_ADDRESS=0.0.0.0:8080 python -m aggregator.server
    FL_NUM_ROUNDS=5 FL_MIN_CLIENTS=2 python -m aggregator.server
"""

# ── Suppress all unnecessary warnings and logs BEFORE any imports ──────────
import logging
import os
import warnings

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

import json
import sys
import threading
import time
from datetime import datetime
from itertools import cycle
from typing import Optional
from pathlib import Path

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import start_server, ServerConfig

from .security.hashing import hash_parameters


# =========================================================
# CONFIGURATION  (override via environment variables)
# =========================================================

# ── Change this single constant to match your number of clients ──
NUM_CLIENTS = int(os.environ.get("FL_NUM_CLIENTS", "3"))

SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080")
NUM_ROUNDS     = int(os.environ.get("FL_NUM_ROUNDS",  "3"))

# Clients required to START a round — set to 1 so any client can
# send updates independently without waiting for others.
MIN_FIT_CLIENTS       = 1
MIN_AVAILABLE_CLIENTS = 1

LOG_DIR     = Path("aggregator/logs")
RESULTS_DIR = Path("aggregator/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# HELPERS
# =========================================================

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _bytes_of(arrays) -> int:
    return sum(a.nbytes for a in arrays)


def _progress_bar(current: int, total: int, width: int = 36) -> str:
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total else 0
    return f"[{bar}] {current}/{total} ({pct:.0f}%)"


def _sep(char: str = "═", width: int = 68) -> str:
    return char * width


# =========================================================
# LIVE SPINNER
# =========================================================

class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Working"):
        self._label   = label
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._elapsed = 0.0

    def _spin(self):
        start = time.time()
        for frame in cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self._elapsed = time.time() - start
            print(f"\r  {frame}  {self._label} … ({self._elapsed:.0f}s)",
                  end="", flush=True)
            time.sleep(0.1)
        print(f"\r  ✔  {self._label} done  ({self._elapsed:.1f}s)          ")

    def start(self):
        self._thread.start()
        return self

    def stop(self, success: bool = True, final_msg: str = ""):
        self._stop.set()
        self._thread.join()
        if not success:
            print(f"\r  ✘  {self._label} failed.                             ")
        elif final_msg:
            print(f"\r  ✔  {final_msg}          ")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# =========================================================
# CUSTOM FEDAVG STRATEGY
# =========================================================

class SecureFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg with:
      - SHA-256 hash verification of every client update
      - Clean per-round progress output
      - JSON round logs saved to aggregator/logs/
      - Clients operate INDEPENDENTLY — any client can submit
        an update at any time without waiting for others.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._round_start: float = 0.0
        self._fit_spinner: Optional[Spinner] = None
        self.total_accepted_clients = 0

    # ── Called before server sends global model to clients ─────────────
    def configure_fit(self, server_round, parameters, client_manager):
        connected = client_manager.num_available()

        print(f"\n{_sep()}")
        print(f"  ROUND {server_round}/{NUM_ROUNDS}  —  CONFIGURE FIT")
        print(f"{_sep('─')}")
        print(f"  Connected clients : {connected}  |  Expected total : {NUM_CLIENTS}")

        arrays = parameters_to_ndarrays(parameters)
        print(f"  Global model      : {sum(a.size for a in arrays):,} params  "
              f"({_human_bytes(_bytes_of(arrays))})")

        self._round_start = time.time()

        self._fit_spinner = Spinner(
            f"Round {server_round}/{NUM_ROUNDS} — waiting for client(s) to train"
        )
        self._fit_spinner.start()

        return super().configure_fit(server_round, parameters, client_manager)

    # ── Called after clients return their updates ───────────────────────
    def aggregate_fit(self, server_round, results, failures):
        elapsed = time.time() - self._round_start

        if self._fit_spinner is not None:
            self._fit_spinner.stop(
                final_msg=f"Round {server_round} — {len(results)} client(s) responded in {elapsed:.1f}s"
            )
            self._fit_spinner = None

        print(f"\n{_sep()}")
        print(f"  ROUND {server_round}/{NUM_ROUNDS}  —  AGGREGATE FIT")
        print(f"{_sep('─')}")
        print(f"  Responses : {len(results)}  |  Failures : {len(failures)}  "
              f"|  Training time : {elapsed:.1f}s")
        print()

        accepted_results = []
        rejected_clients = []

        round_log = {
            "session_id": SESSION_ID,
            "round":      server_round,
            "timestamp":  datetime.now().isoformat(),
            "clients":    [],
        }

        # ── Verify each client update ───────────────────────────────────
        for idx, (client_proxy, fit_res) in enumerate(results, 1):
            client_id     = fit_res.metrics.get("client_id", f"client_{idx}")
            reported_hash = fit_res.metrics.get("model_hash", None)
            num_examples  = fit_res.num_examples

            arrays   = parameters_to_ndarrays(fit_res.parameters)
            payload  = _human_bytes(_bytes_of(arrays))
            n_params = sum(a.size for a in arrays)

            calculated_hash = hash_parameters(arrays)
            hash_valid = (reported_hash is not None) and (reported_hash == calculated_hash)
            status_sym = "✔" if hash_valid else "✘"
            status_lbl = "ACCEPTED" if hash_valid else "REJECTED"

            print(f"  {status_sym} {client_id:12s} | "
                  f"samples={num_examples:,}  params={n_params:,}  "
                  f"payload={payload}  hash={calculated_hash[:16]}…  → {status_lbl}")

            if hash_valid:
                accepted_results.append((client_proxy, fit_res))
                client_status = "accepted"
            else:
                rejected_clients.append(client_id)
                client_status = "rejected"

            round_log["clients"].append({
                "client_id":       client_id,
                "num_examples":    num_examples,
                "payload_bytes":   _bytes_of(arrays),
                "reported_hash":   reported_hash,
                "calculated_hash": calculated_hash,
                "status":          client_status,
            })

        # ── Round summary ───────────────────────────────────────────────
        accepted = len(accepted_results)
        rejected = len(rejected_clients)

        print()
        print(f"  {_progress_bar(accepted, len(results))}")
        print(f"  Accepted : {accepted}  |  Rejected : {rejected}")
        if rejected_clients:
            print(f"  Rejected : {rejected_clients}")

        if accepted == 0:
            print(f"\n  ✘  No valid updates received — global model unchanged.")
            round_log["aggregation"] = "failed — no valid updates"
            self._save_round_log(round_log)
            return None, {}

        # ── FedAvg aggregation ──────────────────────────────────────────
        print(f"\n  ► FedAvg on {accepted} update(s)…")
        agg_start = time.time()

        with Spinner("FedAvg aggregation"):
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, accepted_results, failures
            )

        agg_elapsed = time.time() - agg_start

        if aggregated_parameters is not None:
            agg_arrays = parameters_to_ndarrays(aggregated_parameters)
            print(f"  ✔  Aggregation complete ({agg_elapsed:.2f}s)  "
                  f"| payload={_human_bytes(_bytes_of(agg_arrays))}")
            round_log["aggregation"] = "success"
        else:
            print(f"  ✘  FedAvg failed.")
            round_log["aggregation"] = "failed"

        total_round_time = time.time() - self._round_start
        round_log["round_duration_s"] = round(total_round_time, 2)
        print(f"  Total round time : {total_round_time:.1f}s")
        print(f"{_sep()}\n")

        self._save_round_log(round_log)
        
        # Give clients 3 seconds to process the response and auto-disconnect 
        # so they don't accidentally get sampled for the next round.
        time.sleep(3)
        
        self.total_accepted_clients += accepted
        if self.total_accepted_clients >= NUM_CLIENTS:
            print(f"\n{_sep()}")
            print(f"  🎉 ALL {NUM_CLIENTS} CLIENT(S) HAVE SUCCESSFULLY COMPLETED TRAINING.")
            print(f"  Aggregator shutting down.")
            print(f"{_sep()}\n")
            os._exit(0)
        
        return (aggregated_parameters, aggregated_metrics)

    # ── Evaluation callback ─────────────────────────────────────────────
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        print(f"\n  ROUND {server_round}  —  EVALUATE")
        for _, eval_res in results:
            cid  = eval_res.metrics.get("client_id", "unknown")
            loss = eval_res.loss
            n    = eval_res.num_examples
            print(f"    {cid}: loss={loss:.4f}  samples={n:,}")

        return super().aggregate_evaluate(server_round, results, failures)

    def _save_round_log(self, round_log: dict):
        filename = LOG_DIR / f"session_{SESSION_ID}_round_{round_log['round']}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(round_log, f, indent=4)
        print(f"  📄  Log → {filename}")


# =========================================================
# STRATEGY — clients are fully independent
# =========================================================

strategy = SecureFedAvg(
    fraction_fit=1.0,
    min_fit_clients=MIN_FIT_CLIENTS,             # 1 — a round starts with any available client
    min_available_clients=MIN_AVAILABLE_CLIENTS, # 1 — don't wait for all clients
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
    print(f"  Session ID      : {SESSION_ID}")
    print(f"  Server address  : {SERVER_ADDRESS}")
    print(f"  FL rounds       : {NUM_ROUNDS}")
    print(f"  Expected clients: {NUM_CLIENTS}  (clients act independently)")
    print(f"  Log directory   : {LOG_DIR.resolve()}")
    print(f"\n  ► Clients can connect and send updates independently.")
    print(f"    Training begins as soon as any client sends an update.")
    print(f"\n  Waiting for client(s) to connect on {SERVER_ADDRESS}…")
    print(_sep() + "\n")

    start_server(
        server_address=SERVER_ADDRESS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
