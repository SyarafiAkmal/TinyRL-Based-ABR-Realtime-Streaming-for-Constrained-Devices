import json
import subprocess
import time
import signal
from datetime import datetime

SCENARIOS_PATH  = "./src/shared/train_schemes.json"
SHARED_PATH     = "./src/shared/status.json"
CONTAINER_NAME  = "edge_node_container"
CLOUD_CONTAINER = "cloud_node_container"
BASELINE_RP4    = 0.25

SCENARIO_DURATIONS = {
    "optimal_conditions":       180,
    "high_network_activity":    120,
    "thermal_throttling_start": 120,
    "low_battery_powersave":    120,
    "critical_system_stress":   300,  # 5 min
}

# ---------------------------------------------------------------------------
# Graceful Ctrl+C
# ---------------------------------------------------------------------------

_running = True

def _handle_sigint(sig, frame):
    global _running
    print("\n[orchestrator] Ctrl+C received, finishing current scenario then stopping...")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def f_thermal(T_celsius: float) -> float:
    severity = max(0.0, min(1.0, (T_celsius - 20) / (80 - 20)))
    return max(0.4, 1.0 - severity)


def compute_c_limit(sc: dict) -> float:
    return BASELINE_RP4 * sc["battery_level"] * f_thermal(sc["temp_celsius"])


def apply_hardware_limit(sc: dict):
    c_limit = compute_c_limit(sc)
    cpus    = max(0.01, c_limit)

    if c_limit < 0.01:
        print(f"  ⚠ c_limit={c_limit:.5f} below Docker floor 0.01, clamped")

    print(f"  Applying CPU quota: {cpus:.5f} (C_limit={c_limit:.5f})")
    try:
        subprocess.run(
            ["docker", "update", f"--cpus={cpus:.5f}", CONTAINER_NAME],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to update container CPU quota: {e.stderr}")


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def apply_network_conditions(sc: dict):
    rate  = sc.get("rate",  "10mbit")
    delay = sc.get("delay", "0ms")
    loss  = sc.get("loss",  "0%").rstrip("%")

    subprocess.run(
        ["docker", "exec", CLOUD_CONTAINER,
         "tc", "qdisc", "del", "dev", "eth0", "root"],
        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    )

    cmd = [
        "docker", "exec", CLOUD_CONTAINER,
        "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
        "rate",  rate,
        "delay", delay,
        "loss",  f"{loss}%",
    ]

    print(f"  Applying tc netem: rate={rate}  delay={delay}  loss={loss}%")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to apply network conditions: {e.stderr}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

with open(SCENARIOS_PATH, "r") as f:
    scenarios: dict = json.load(f)

total      = len(scenarios)
pass_count = 0

while _running:
    pass_count += 1
    print(f"\n{'='*50}")
    print(f">>> Pass {pass_count} started  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*50}")

    for idx, (name, sc) in enumerate(scenarios.items(), start=1):
        if not _running:
            break

        duration = SCENARIO_DURATIONS.get(name, 60)
        print(f"\n[{idx}/{total}] Scenario: {name}  (pass {pass_count})")
        print(f"  temp={sc['temp_celsius']}°C  battery={sc['battery_level']}  "
              f"rate={sc['rate']}  delay={sc['delay']}  loss={sc['loss']}")

        # Write scenario info to shared status so edge-node can log it
        with open(SHARED_PATH, "w") as f:
            json.dump({**sc, "scenario": name, "pass": pass_count}, f, indent=2)
        print(f"  status.json updated")

        apply_hardware_limit(sc)
        apply_network_conditions(sc)

        print(f"  Running for {duration}s...")
        for _ in range(duration):
            if not _running:
                break
            time.sleep(1)

        print(f"  Done.")

    print(f"\n>>> Pass {pass_count} complete.  [{datetime.now().strftime('%H:%M:%S')}]")

print("\n>>> Orchestrator stopped cleanly.")