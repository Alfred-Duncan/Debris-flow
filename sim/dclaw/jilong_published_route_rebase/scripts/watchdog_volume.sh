#!/usr/bin/env bash
set -uo pipefail
CASE=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?case_name}; VOL=${2:?volume_m3}; TFINAL=900
RUN="$CASE/runs/$NAME"; OUT="$RUN/_output"
test ! -e "$RUN" || { echo "Run already exists: $RUN" >&2; exit 2; }
mkdir -p "$OUT"; cd "$CASE"
python - "$NAME" "$VOL" <<'PY'
import json,sys
name,v=sys.argv[1],float(sys.argv[2])
json.dump({'case_id':name,'V_m3':v,'T_s':90.0,'tfinal_s':900.0,'output_interval_s':30.0,'entrainment':1,'sourcefix':True,'momentum_factor':1.0},open('active_run.json','w'),indent=2)
PY
python setrun.py > "$RUN/setrun.log" 2>&1 || exit 3
test "$(tr -d '[:space:]' < source_momentum_factor.data)" = "1" || { echo "K_u_NOT_1" >&2; exit 4; }
cp active_run.json conservative_source.data source_momentum_factor.data claw.data dclaw.data geoclaw.data "$RUN/"
cp *.data "$CASE/terrain/published_route_domain_64m.tt3" "$OUT/"
sed -i 's#terrain/published_route_domain_64m.tt3#published_route_domain_64m.tt3#' "$OUT/topo.data"
START=$(date +%s)
(cd "$OUT" && exec "$CASE/xdclaw") > "$RUN/run.log" 2>&1 & PID=$!
echo "$PID" > "$RUN/pid"; echo "$(date -Is) START case=$NAME pid=$PID V_m3=$VOL K_u=1" > "$RUN/watchdog.log"
LAST_LOG=0; LAST_OUT=0
while kill -0 "$PID" 2>/dev/null; do
 sleep 60; NOW=$(date +%s); PS=$(ps -p "$PID" -o pid=,stat=,pcpu=,pmem= | xargs || true)
 LM=$(stat -c %Y "$RUN/run.log" 2>/dev/null || echo 0)
 OM=$(find "$OUT" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); OM=${OM:-0}
 FR=$(find "$OUT" -name 'fort.q????' | wc -l)
 FATAL=$(tail -100 "$RUN/run.log" 2>/dev/null | grep -Eic 'error|fatal|segmentation|segfault|traceback|nan|floating point|forrtl|abort|killed|cannot open|no such file' || true)
 echo "$(date -Is) HEARTBEAT case=$NAME elapsed=$((NOW-START)) pid=$PID ps=[$PS] frames=$FR latest_frame_timestamp=$OM run_log_timestamp=$LM fatal_patterns=$FATAL" >> "$RUN/watchdog.log"
 if [ "$LM" -le "$LAST_LOG" ] && [ "$OM" -le "$LAST_OUT" ] && [ $((NOW-START)) -ge 180 ] && echo "$PS" | grep -q ' 0.0'; then echo "$(date -Is) POSSIBLE_STALL" >> "$RUN/watchdog.log"; kill "$PID" || true; fi
 LAST_LOG=$LM; LAST_OUT=$OM
done
wait "$PID"; RC=$?
echo "$(date -Is) EXIT rc=$RC frames=$(find "$OUT" -name 'fort.q????' | wc -l)" >> "$RUN/watchdog.log"
exit "$RC"
