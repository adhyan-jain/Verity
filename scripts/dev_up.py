#!/usr/bin/env python
"""
One-command local bring-up for Verity: trains the fraud model if needed,
starts all four backend services (agent :8000, fraud :8001, ledger :8002,
typology :8003), then starts the dashboard dev server (:3000) in the
foreground. Ctrl+C stops everything.

Usage:
    python scripts/dev_up.py            # backend services + dashboard
    python scripts/dev_up.py --no-dashboard   # backend services only
"""

import argparse
import os
import subprocess
import sys

from _services import (
    ROOT,
    ensure_fraud_model_trained,
    load_env_file,
    start_all_backend_services,
    stop_all,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Skip starting the Vite dev server"
    )
    args = parser.parse_args()

    load_env_file()
    ensure_fraud_model_trained()

    procs = start_all_backend_services()
    dashboard_proc = None

    try:
        if not args.no_dashboard:
            print("[start] dashboard -> http://localhost:3000")
            dashboard_env = dict(os.environ)
            dashboard_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=os.path.join(ROOT, "dashboard"),
                env=dashboard_env,
                shell=(os.name == "nt"),
            )

        print("\nVerity is up:")
        print("  agent     http://localhost:8000")
        print("  fraud     http://localhost:8001")
        print("  ledger    http://localhost:8002")
        print("  typology  http://localhost:8003")
        if dashboard_proc is not None:
            print("  dashboard http://localhost:3000")
        print("\nPress Ctrl+C to stop everything.\n")

        if dashboard_proc is not None:
            dashboard_proc.wait()
        else:
            for proc in procs.values():
                proc.wait()
        return 0
    except KeyboardInterrupt:
        print("\n[stop] shutting down...")
        return 0
    finally:
        if dashboard_proc is not None and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
        stop_all(procs)


if __name__ == "__main__":
    sys.exit(main())
