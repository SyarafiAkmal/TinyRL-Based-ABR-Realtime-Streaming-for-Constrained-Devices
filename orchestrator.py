import json
import subprocess
import time
import os
from datetime import datetime

SCENARIOS_PATH = "./src/shared/train_schemes.json"
SHARED_PATH    = "./src/shared/status.json"
CONTAINER_NAME = "edge_node_container"
CLOUD_CONTAINER = "cloud_node_container"
BASELINE_S3    = 0.05  # kapasitas total ESP32-S3 di Docker

SCENARIO_DURATION = 20

def f_thermal(T_celsius: float) -> float:
    """Normalizing suhu antara 20°C (optimal) dan 100°C (throttling parah)."""
    severity = max(0.0, min(1.0, (T_celsius - 20) / (100 - 20)))
    return 1.0 - severity

def compute_c_limit(sc: dict) -> float:
    """C_limit = baseline_ESP32-S3 × f_battery × f_thermal."""
    
    f_b = sc['battery_level']
    f_t = f_thermal(sc['temp_celsius'])
    C_limit = BASELINE_S3 * f_b * f_t

    return C_limit

def apply_hardware_limit(sc: dict):
    """
    Apply cpu quota on edge_node_container.
    
    C_limit = baseline_ESP32-S3 × f_battery × f_thermal
    """
    c_limit = compute_c_limit(sc)
    cpus = max(0.01, c_limit)
    if c_limit < 0.01:
        print(f"  ⚠ c_limit={c_limit:.5f} below Docker floor 0.01, clamped")

    print(f"  Applying CPU quota: {cpus:.5f} (C_limit={c_limit:.5f})")
    try:
        subprocess.run(
            ["docker", "update", f"--cpus={cpus:.5f}", CONTAINER_NAME],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to update container CPU quota: {e}")

    

def apply_network_conditions():
    """Apply tc netem qdisc on cloud_node's eth0 egress."""
    pass


with open(SCENARIOS_PATH, 'r') as f:
    scenarios = json.load(f)

for i, sc in enumerate(scenarios):
    # Apply hardware limits based on current state
    print(f"[Scenario {sc}]")
    with open(SHARED_PATH, 'w') as f:
        json.dump(scenarios[sc], f)
    print(f"[Scenario {sc}] Applying hardware limits...")
    apply_hardware_limit(scenarios[sc])

    # Apply network conditions based on current state
    # print(f"[Scenario {sc}] Applying network conditions...")
    # apply_network_conditions()

    time.sleep(SCENARIO_DURATION)
    print(f"[Scenario {sc}] Done.")

print("\n>>> All scenarios done.")