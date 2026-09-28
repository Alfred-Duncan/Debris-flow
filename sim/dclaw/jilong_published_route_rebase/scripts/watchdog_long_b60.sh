#!/usr/bin/env bash
set -uo pipefail
CASE=$(cd "$(dirname "$0")/.." && pwd); NAME=RECON_B_rate060_LONG3600; RUN="$CASE/runs/$NAME"; OUT="$RUN/_output"
test ! -e "$RUN" || { echo "Run already exists" >&2; exit 2; }; mkdir -p "$OUT"; cd "$CASE"
python3 - <<'EOF'
import json
x=json.load(open('runs/RECON_B_rate060/active_run_snapshot.json'))
x.update({'case_id':'RECON_B_rate060_LONG3600','tfinal_s':3600.0,'output_interval_s':30.0,'continuation_method':'EXACT_B60_RERUN_FROM_ZERO'})
open('active_run.json','w').write(json.dumps(x,indent=2)+'\n')
EOF
python3 setrun.py > "$RUN/setrun.log" 2>&1 || exit 3
python3 - "$RUN" <<'EOF'
from pathlib import Path
import hashlib,json
c=Path.cwd();r=Path(__import__('sys').argv[1]); r.joinpath('active_run_snapshot.json').write_bytes(c.joinpath('active_run.json').read_bytes())
def h(p):return hashlib.sha256((c/p).read_bytes()).hexdigest()
r.joinpath('input_hashes.json').write_text(json.dumps({p:h(p) for p in ['terrain/published_route_domain_64m.tt3','entrainment/erodible_thickness_fullroute_8m.tt3','model_geometry_sourcefix.json','source/source_support_published_s0_floorfix.json','src2.f90','xdclaw']},indent=2)+'\n')
EOF
cp active_run.json conservative_source.data source_momentum_factor.data claw.data dclaw.data geoclaw.data "$RUN/"
cp *.data "$CASE/terrain/published_route_domain_64m.tt3" "$OUT/"; sed -i 's#terrain/published_route_domain_64m.tt3#published_route_domain_64m.tt3#' "$OUT/topo.data"
START=$(date +%s); (cd "$OUT" && exec "$CASE/xdclaw") > "$RUN/run.log" 2>&1 & PID=$!; echo "$PID" > "$RUN/pid"; echo "$(date -Is) START case=$NAME pid=$PID tfinal=3600" > "$RUN/watchdog.log"
LASTL=0; LASTO=0
while kill -0 "$PID" 2>/dev/null; do
 sleep 60; NOW=$(date +%s); PS=$(ps -p "$PID" -o pid=,stat=,pcpu=,pmem= | xargs || true); LM=$(stat -c %Y "$RUN/run.log" 2>/dev/null || echo 0); OM=$(find "$OUT" -type f -printf '%T@\n' | sort -nr | head -1 | cut -d. -f1); OM=${OM:-0}; FR=$(find "$OUT" -name 'fort.q????'|wc -l); ST=$(tail -1 "$RUN/run.log"|sed -n 's/.*final t = *\([0-9.Ee+-]*\).*/\1/p')
 F=$(tail -100 "$RUN/run.log"|grep -Eic 'error|fatal|segmentation|segfault|traceback|nan|floating point|forrtl|abort|killed|cannot open|no such file'||true); echo "$(date -Is) HEARTBEAT elapsed=$((NOW-START)) solver_t=${ST:-NA} pid=$PID ps=[$PS] frames=$FR latest_output_mtime=$OM run_log_mtime=$LM fatal_patterns=$F" >> "$RUN/watchdog.log"
 if [ "$LM" -le "$LASTL" ] && [ "$OM" -le "$LASTO" ] && [ $((NOW-START)) -ge 180 ] && echo "$PS"|grep -Eq ' 0\.0( |$)';then echo "$(date -Is) POSSIBLE_STALL" >> "$RUN/watchdog.log";kill "$PID"||true;fi; LASTL=$LM;LASTO=$OM
done
wait "$PID";RC=$?;echo "$(date -Is) EXIT rc=$RC frames=$(find "$OUT" -name 'fort.q????'|wc -l)" >> "$RUN/watchdog.log";exit "$RC"
