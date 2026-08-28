"""
Federated Learning Aggregator — server.py
==========================================
Run from the project root:
    python -m aggregator.server

Override defaults with environment variables:
    FL_SERVER_ADDRESS=0.0.0.0:8080 python -m aggregator.server
    FL_NUM_ROUNDS=5 FL_MIN_CLIENTS=2 python -m aggregator.server
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from .security.hashing import hash_parameters

# =========================================================
# CONFIGURATION  (override via environment variables)
# =========================================================

SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080")
NUM_ROUNDS     = int(os.environ.get("FL_NUM_ROUNDS",    "3"))
MIN_CLIENTS    = int(os.environ.get("FL_MIN_CLIENTS",   "3"))

LOG_DIR        = Path("aggregator/logs")
RESULTS_DIR    = Path("aggregator/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# HELPERS
# =========================================================

def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Return a simple ASCII progress bar string."""
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total else 0
    return f"[{bar}] {current}/{total} ({pct:.0f}%)"


def _bytes_of(arrays) -> int:
    """Total bytes of a list of NumPy arrays."""
    return sum(a.nbytes for a in arrays)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# =========================================================
# CUSTOM FEDAVG STRATEGY WITH PROGRESS + HASH VERIFICATION
# =========================================================


class SecureFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg extended with:
      - SHA-256 hash verification of every client update
      - Detailed per-round progress and transfer stats
      - JSON round logs saved to aggregator/logs/
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._round_start: float = 0.0

    # ------------------------------------------------------------------
    # Flower callback: called just before the server sends the global
    # model out to clients at the start of each round.
    # ------------------------------------------------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n{'='*70}")
        print(f"  ROUND {server_round}/{NUM_ROUNDS}  —  CONFIGURE FIT")
        print(f"{'='*70}")

        # Count connected clients
        connected = client_manager.num_available()
        print(f"  Connected clients : {connected}")
        print(f"  Required clients  : {MIN_CLIENTS}")
        print(_progress_bar(connected, MIN_CLIENTS, 30))

        # Report size of global model being distributed
        arrays = parameters_to_ndarrays(parameters)
        size   = _bytes_of(arrays)
        params = sum(a.size for a in arrays)
        print(f"\n  ► Distributing global model to all clients")
        print(f"    Parameters : {params:,}")
        print(f"    Payload    : {_human_bytes(size)}")

        self._round_start = time.time()

        return super().configure_fit(server_round, parameters, client_manager)

    # ------------------------------------------------------------------
    # Flower callback: called after all clients return their updates.
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):

        elapsed_client = time.time() - self._round_start

        print(f"\n{'='*70}")
        print(f"  ROUND {server_round}/{NUM_ROUNDS}  —  AGGREGATE FIT")
        print(f"{'='*70}")
        print(f"  Client training time : {elapsed_client:.1f}s")
        print(f"  Total responses      : {len(results)}")
        print(f"  Failures / timeouts  : {len(failures)}")
        print()

        accepted_results = []
        rejected_clients = []

        round_log = {
            "session_id":  SESSION_ID,
            "round":       server_round,
            "timestamp":   datetime.now().isoformat(),
            "clients":     [],
        }

        # -----------------------------------------------------------------
        # Verify each client update
        # -----------------------------------------------------------------
        for idx, (client_proxy, fit_res) in enumerate(results, 1):

            client_id     = fit_res.metrics.get("client_id", f"client_{idx}")
            reported_hash = fit_res.metrics.get("model_hash", None)
            num_examples  = fit_res.num_examples

            arrays   = parameters_to_ndarrays(fit_res.parameters)
            payload  = _human_bytes(_bytes_of(arrays))
            n_params = sum(a.size for a in arrays)

            print(f"  ─── Client {idx}/{len(results)} : {client_id} ───")
            print(f"      Training samples : {num_examples:,}")
            print(f"      Parameters       : {n_params:,}")
            print(f"      Payload received : {payload}")

            # Recalculate hash server-side
            calculated_hash = hash_parameters(arrays)

            print(f"      Reported hash    : {reported_hash[:16]}…" if reported_hash else "      Reported hash    : (none)")
            print(f"      Server hash      : {calculated_hash[:16]}…")

            hash_valid = (reported_hash is not None) and (reported_hash == calculated_hash)

            if hash_valid:
                print(f"      ✔  HASH VERIFIED  →  UPDATE ACCEPTED")
                accepted_results.append((client_proxy, fit_res))
                client_status = "accepted"
            else:
                print(f"      ✘  HASH MISMATCH  →  UPDATE REJECTED")
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

            print()

        # -----------------------------------------------------------------
        # Round summary
        # -----------------------------------------------------------------
        total     = len(results)
        accepted  = len(accepted_results)
        rejected  = len(rejected_clients)

        print(f"  {'─'*50}")
        print(f"  ROUND {server_round} SUMMARY")
        print(f"  {'─'*50}")
        print(f"  {_progress_bar(accepted, total, 30)}")
        print(f"  Accepted : {accepted}  |  Rejected : {rejected}")

        if rejected_clients:
            print(f"  Rejected clients : {rejected_clients}")

        # -----------------------------------------------------------------
        # Safety check: abort round if nothing is usable
        # -----------------------------------------------------------------
        if accepted == 0:
            print("\n  ✘  ERROR: No valid updates received. Global model unchanged.")
            round_log["aggregation"] = "failed — no valid updates"
            self._save_round_log(round_log)
            return None, {}

        # -----------------------------------------------------------------
        # Federated Averaging
        # -----------------------------------------------------------------
        agg_start = time.time()
        print(f"\n  ► Running FedAvg on {accepted} accepted update(s)…")

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, accepted_results, failures
        )

        agg_elapsed = time.time() - agg_start

        if aggregated_parameters is not None:
            agg_arrays = parameters_to_ndarrays(aggregated_parameters)
            print(f"  ✔  FedAvg complete  ({agg_elapsed:.2f}s)")
            print(f"     Aggregated payload : {_human_bytes(_bytes_of(agg_arrays))}")
            round_log["aggregation"] = "success"
        else:
            print(f"  ✘  FedAvg failed.")
            round_log["aggregation"] = "failed"

        total_round_time = time.time() - self._round_start
        round_log["round_duration_s"] = round(total_round_time, 2)
        print(f"  Total round time : {total_round_time:.1f}s")
        print(f"{'='*70}\n")

        self._save_round_log(round_log)
        return (aggregated_parameters, aggregated_metrics)

    # ------------------------------------------------------------------
    # Optional: log evaluation results if clients send them
    # ------------------------------------------------------------------
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        print(f"\n  ROUND {server_round}  —  EVALUATE RESULTS")
        for client_proxy, eval_res in results:
            cid  = eval_res.metrics.get("client_id", "unknown")
            loss = eval_res.loss
            n    = eval_res.num_examples
            print(f"    {cid}: loss={loss:.4f}  samples={n:,}")

        return super().aggregate_evaluate(server_round, results, failures)

    # ------------------------------------------------------------------
    def _save_round_log(self, round_log: dict):
        filename = LOG_DIR / f"session_{SESSION_ID}_round_{round_log['round']}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(round_log, f, indent=4)
        print(f"  📄  Round log saved → {filename}")


# =========================================================
# STRATEGY CONFIGURATION
# =========================================================

strategy = SecureFedAvg(
    fraction_fit=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    fraction_evaluate=0.0,
    min_evaluate_clients=0,
)


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  FEDERATED LEARNING AGGREGATOR  —  VisDrone / YOLO11n")
    print("=" * 70)
    print(f"  Session ID      : {SESSION_ID}")
    print(f"  Server address  : {SERVER_ADDRESS}")
    print(f"  FL rounds       : {NUM_ROUNDS}")
    print(f"  Required clients: {MIN_CLIENTS}")
    print(f"  Log directory   : {LOG_DIR.resolve()}")
    print(f"\n  Tip: override any setting via environment variables:")
    print(f"       FL_SERVER_ADDRESS  FL_NUM_ROUNDS  FL_MIN_CLIENTS")
    print(f"\n  Waiting for {MIN_CLIENTS} client(s) to connect…")
    print("=" * 70 + "\n")

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
