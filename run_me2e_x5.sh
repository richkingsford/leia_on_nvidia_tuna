#!/usr/bin/env bash
# 5x modified-E2E trial, with the standalone honest reset (reset_once.py) between trials.
# Each me2e skips its own initial reset (robot starts each trial from a fresh reset pose).
# Safety: abort the whole loop if a reset shows the wrong-way "distance teleport" signature
# or if the OAK throws a crash dump during a reset.
set -u
cd /home/richkingsford/cub-queso

MASTER=/tmp/me2e_x5_master.log
: > "$MASTER"

for i in 1 2 3 4 5; do
  echo "================ TRIAL $i / 5 ================" | tee -a "$MASTER"
  echo "[$(date '+%H:%M:%S')] TRIAL $i: me2e (skip-initial-reset)" | tee -a "$MASTER"
  python3 run_me2e_evidence_once.py --skip-initial-reset > "/tmp/me2e_x5_c${i}.log" 2>&1
  me2e_rc=$?
  echo "[$(date '+%H:%M:%S')] TRIAL $i: me2e exit=$me2e_rc | $(python3 site_latest.py)" | tee -a "$MASTER"
  if grep -q "has crashed" "/tmp/me2e_x5_c${i}.log"; then
    echo "[$(date '+%H:%M:%S')] TRIAL $i: NOTE camera crash dump during me2e (recovers on restart)" | tee -a "$MASTER"
  fi

  sleep 3  # let the OAK USB re-enumerate between processes

  echo "[$(date '+%H:%M:%S')] TRIAL $i: reset (reset_once.py)" | tee -a "$MASTER"
  python3 reset_once.py > "/tmp/me2e_x5_reset_c${i}.log" 2>&1
  reset_rc=$?
  reset_line=$(grep -E "^\[RESET\] (success|DONE)" "/tmp/me2e_x5_reset_c${i}.log" | tr '\n' ' ')
  guard=$(python3 reset_guard.py "/tmp/me2e_x5_reset_c${i}.log")
  echo "[$(date '+%H:%M:%S')] TRIAL $i: reset exit=$reset_rc | $reset_line" | tee -a "$MASTER"
  echo "[$(date '+%H:%M:%S')] TRIAL $i: reset guard: $guard" | tee -a "$MASTER"

  # Hard safety stop on a wrong-way reset.
  if [[ "$guard" == SUSPECT* ]]; then
    echo "[$(date '+%H:%M:%S')] ABORT: wrong-way reset detected after trial $i; stopping loop for safety." | tee -a "$MASTER"
    python3 -c "import a_follow_the_brick as f; from helper_robot_control import Robot; r=Robot(); f._stop_robot(r); r.close()" 2>/dev/null
    exit 10
  fi

  sleep 3
done

echo "================ ALL 5 TRIALS COMPLETE ================" | tee -a "$MASTER"
