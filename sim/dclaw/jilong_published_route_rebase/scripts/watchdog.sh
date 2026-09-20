#!/usr/bin/env bash
set -uo pipefail
CASE=$(cd "$(dirname "$0")/.."&&pwd); NAME=$1; EN=$2; RUN="$CASE/runs/$NAME"; OUT="$RUN/_output"; mkdir -p "$OUT"; cd "$CASE"
python - <<PY
import json
p='active_run.json';json.dump({'case_id':'$NAME','V_m3':2e6,'T_s':90.,'tfinal_s':900.,'output_interval_s':30.,'entrainment':int('$EN')},open(p,'w'),indent=2)
PY
python setrun.py > "$RUN/setrun.log" 2>&1 || exit 2
cp *.data "$CASE/terrain/published_route_domain_64m.tt3" "$OUT/"; sed -i 's#terrain/published_route_domain_64m.tt3#published_route_domain_64m.tt3#' "$OUT/topo.data"; rm -f "$OUT"/fort.q???? "$OUT"/fort.t????
START=$(date +%s); (cd "$OUT"&&exec "$CASE/xdclaw") > "$RUN/run.log" 2>&1 & PID=$!;echo "$PID">"$RUN/pid";echo "$(date -Is) START pid=$PID">"$RUN/watchdog.log";LAST=$START
while kill -0 "$PID" 2>/dev/null;do sleep 60;NOW=$(date +%s);PS=$(ps -p "$PID" -o stat=,pcpu=,pmem=|xargs||true);LM=$(stat -c %Y "$RUN/run.log" 2>/dev/null||echo 0);OM=$(find "$OUT" -type f -printf '%T@\n'|sort -nr|head -1|cut -d. -f1);OM=${OM:-0};FR=$(find "$OUT" -name 'fort.q????'|wc -l);F=$(tail -100 "$RUN/run.log"|grep -Eic 'error|fatal|segmentation|segfault|traceback|nan|floating point|forrtl|abort|killed|cannot open|no such file'||true);echo "$(date -Is) HEARTBEAT elapsed=$((NOW-START)) ps=[$PS] frames=$FR newest=$OM log=$LM fatal=$F">>"$RUN/watchdog.log";if [ "$LM" -le "$LAST" ]&&[ "$OM" -le "$LAST" ]&&[ $((NOW-LAST)) -ge 180 ]&&echo "$PS"|grep -q ' 0.0';then echo "$(date -Is) POSSIBLE_STALL">>"$RUN/watchdog.log";kill "$PID"||true;fi;LAST=$NOW;done
wait "$PID";echo "$(date -Is) EXIT rc=$? frames=$(find "$OUT" -name 'fort.q????'|wc -l)">>"$RUN/watchdog.log"
