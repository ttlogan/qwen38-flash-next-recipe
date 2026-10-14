#!/bin/sh
# Cap the 10 Cortex-X925 performance cores (the GB10 heat source) to 2.8GHz.
# On the Ascent GX10 / DGX Spark, CPUs 5-9 and 15-19 are the 3.9GHz X925 cores.
# Idempotent on any boot: cores that can't reach 3.9GHz are already <= 2.8GHz,
# and writing scaling_max_freq is safe to repeat. The A725 efficiency cores
# (CPUs 0-4, 10-14) are left at their native frequency.
set -e
for c in 5 6 7 8 9 15 16 17 18 19; do
  [ -w /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq ] || continue
  echo 2800000 > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq 2>/dev/null || true
done
