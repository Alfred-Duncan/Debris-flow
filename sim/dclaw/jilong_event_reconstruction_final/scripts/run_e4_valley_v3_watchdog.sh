#!/usr/bin/env bash
set -uo pipefail
CASE="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$CASE/runs/C3_entrainment_E4_valley_v3"
OUT="$RUN/_output"
LOG="$RUN/run.log"
WATCH="$RUN/watchdog.log"
mkdir -p "$OUT"
cd "$CASE"
python setrun.py > "$RUN/setrun.log" 2>&1 || { echo "$(date -Is) SETRUN_FAILURE" >> "$WATCH"; exit 2; }
cp *.data jilong_copernicus_64m.tt3 "$OUT/"
rm -f "$OUT"/fort.q???? "$OUT"/fort.t????
START=$(date +%s)
( cd "$OUT" && "$CASE/xdclaw" ) > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$RUN/pid"
echo "$(date -Is) START pid=$PID log=$LOG" >> "$WATCH"
LAST=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
  NOW=$(date +%s); EL=$((NOW-START)); PS=$(ps -p "$PID" -o stat=,pcpu=,pmem= 2>/dev/null | xargs || true)
  LMT=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  OMT=$(find "$OUT" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); OMT=${OMT:-0}
  FR=$(find "$OUT" -name 'fort.q????' | wc -l)
  FATAL=$(tail -80 "$LOG" 2>/dev/null | grep -Eic 'error|fatal|segmentation|segfault|traceback|nan|inf|floating point|forrtl|abort|killed|cannot open|no such file|stopping' || true)
  echo "$(date -Is) HEARTBEAT elapsed_s=$EL pid=$PID ps=[$PS] frames=$FR newest_output=$OMT fatal_matches=$FATAL" >> "$WATCH"
  if [ "$FATAL" -gt 0 ]; then echo "$(date -Is) RUNTIME_FAILURE fatal-pattern" >> "$WATCH"; fi
  if [ "$LMT" -le "$LAST" ] && [ "$OMT" -le "$LAST" ] && [ $((NOW-LAST)) -ge 180 ] && echo "$PS" | grep -Eq '^[A-Za-z].* 0\.0'; then
     echo "$(date -Is) POSSIBLE_STALL terminate pid=$PID" >> "$WATCH"; kill -TERM "$PID" 2>/dev/null || true; sleep 10; kill -KILL "$PID" 2>/dev/null || true; break
  fi
  LAST=$NOW
done
wait "$PID"; RC=$?
echo "$(date -Is) EXIT rc=$RC frames=$(find "$OUT" -name 'fort.q????' | wc -l)" >> "$WATCH"
exit "$RC"
