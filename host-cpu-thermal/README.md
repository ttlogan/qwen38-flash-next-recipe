# Host CPU Thermal — governor + performance-core cap

Host-level thermal relief for the GB10 SoC during vLLM inference. Unlike the
container-level `mods/busy-loop-fix` (patches the spin-wait in `shm_broadcast.py`),
this runs on the **host** node and persists across reboots as a systemd unit.

The GB10's CPU ACPI zones (0/5, the SoC package) run hot under TP=2 decode
because the 10 Cortex-X925 performance cores boost to 3.9GHz and get pinned by
the IPC/NCCL work. Capping the performance cores to 2.8GHz (same as the
efficiency cores) drops the SoC ~9C and the performance-core cluster ~14C with
~no single-stream token cost (memory-bandwidth-bound inference).

## Files

| Path | Purpose |
|---|---|
| `cpu-governor.service` | systemd unit: set `schedutil` governor, then cap X925 cores to 2.8GHz |
| `gx10-cpu-cap.sh` | caps CPUs 5-9,15-19 to 2.8GHz (idempotent; A725 cores untouched) |

## Install (both nodes)

```bash
sudo install -m 0755 gx10-cpu-cap.sh /usr/local/sbin/gx10-cpu-cap.sh
sudo cp cpu-governor.service /etc/systemd/system/cpu-governor.service
sudo systemctl daemon-reload
sudo systemctl enable --now cpu-governor.service
```

## Apply immediately (no reboot)

```bash
sudo systemctl restart cpu-governor.service
```

## Verify

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c   # 20 schedutil
for c in 5 6 7 8 9 15 16 17 18 19; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq; done | sort -u   # 2800000
```

## Revert

```bash
sudo systemctl disable --now cpu-governor.service
sudo cpupower frequency-set -g performance
```

## Notes

- Governor driver is `cppc_cpufreq`; both `schedutil` and `performance` are in
  `scaling_available_governors`.
- `scaling_max_freq` is a sysfs write that resets on reboot, so it **must** be
  re-applied by the boot unit (the `ExecStart` does this).
- `nvidia-smi -pl` is a NO-OP on the GB10 ("not supported in current scope") —
  do not use a power-limit service.
- Do not cap below 2.8GHz; throughput starts to drop noticeably around 2.15GHz.
