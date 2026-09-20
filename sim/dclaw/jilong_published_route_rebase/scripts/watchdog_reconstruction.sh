#!/usr/bin/env bash
# Run exactly one predeclared reconstruction case with a 60 s watchdog.
set -uo pipefail
CASE=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?case}; HE=${2:?h_e}; RATE=${3:?rate}; PHI=${4:?phi}; MANNING=${5:?manning}; MASK=${6:?mask}
RUN="$CASE/runs/$NAME"; OUT="$RUN/_output"
test ! -e "$RUN" || { echo "Run already exists: $RUN" >&2; exit 2; }
mkdir -p "$OUT"; cd "$CASE"
python3 - "$NAME" "$HE" "$RATE" "$PHI" "$MANNING" "$MASK" <<'EOF'
import json,sys
name,he,rate,phi,n,mask=sys.argv[1],*map(float,sys.argv[2:6]),sys.argv[6]
a={'case_id':name,'V_m3':1.0e7,'T_s':90.0,'tfinal_s':900.0,'output_interval_s':30.0,'entrainment':1,'sourcefix':True,'momentum_factor':1.0,'h_e_m':he,'entrainment_rate':rate,'phi_deg':phi,'manning':n,'erodible_thickness_file':mask,'physical_change':'predeclared reconstruction-ladder parameter only'}
open('active_run.json','w').write(json.dumps(a,indent=2)+'\n')
EOF
python3 setrun.py > "$RUN/setrun.log" 2>&1 || exit 3
test "$(tr -d '[:space:]' < source_momentum_factor.data)" = "1" || { echo K_u_NOT_1 >&2; exit 4; }
python3 - "$RUN" "$MASK" <<'EOF'
import hashlib,json,math,sys
from pathlib import Path
r=Path(sys.argv[1]); c=Path.cwd(); mask=Path(sys.argv[2])
r.joinpath('active_run_snapshot.json').write_bytes(c.joinpath('active_run.json').read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(c.joinpath('active_run.json').read_text()); rows=[]
for t,expected in [(30,2.5e6),(60,7.5e6),(90,1e7)]:
 got=.5*a['V_m3']*(1-math.cos(math.pi*t/a['T_s'])); err=abs(got-expected)/expected
 rows.append({'time_s':t,'expected_conservative_source_mass_m3':expected,'analytic_half_cosine_integral_m3':got,'relative_error':err,'pass':err<=.01})
gate={'case':a['case_id'],'V_m3':a['V_m3'],'T_s':a['T_s'],'mass_gate_pass':all(x['pass'] for x in rows),'rows':rows,'sha256':{'erodible_tt3':sha(c/mask),'terrain':sha(c/'terrain/published_route_domain_64m.tt3'),'executable':sha(c/'xdclaw')}}
r.joinpath('source_mass_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
r.joinpath('input_hashes.json').write_text(json.dumps(gate['sha256'],indent=2)+'\n')
if not gate['mass_gate_pass']: raise SystemExit('SOURCE_MASS_GATE_FAIL')
EOF
cp active_run.json conservative_source.data source_momentum_factor.data claw.data dclaw.data geoclaw.data "$RUN/"
cp *.data "$CASE/terrain/published_route_domain_64m.tt3" "$OUT/"
sed -i 's#terrain/published_route_domain_64m.tt3#published_route_domain_64m.tt3#' "$OUT/topo.data"
START=$(date +%s)
(cd "$OUT" && exec "$CASE/xdclaw") > "$RUN/run.log" 2>&1 & PID=$!
echo "$PID" > "$RUN/pid"; echo "$(date -Is) START case=$NAME pid=$PID h_e=$HE rate=$RATE phi=$PHI n=$MANNING" > "$RUN/watchdog.log"
LAST_LOG=0; LAST_OUT=0
while kill -0 "$PID" 2>/dev/null; do
 sleep 60; NOW=$(date +%s); PS=$(ps -p "$PID" -o pid=,stat=,pcpu=,pmem= | xargs || true)
 LM=$(stat -c %Y "$RUN/run.log" 2>/dev/null || echo 0); OM=$(find "$OUT" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); OM=${OM:-0}; FR=$(find "$OUT" -name 'fort.q????' | wc -l)
 FATAL=$(tail -100 "$RUN/run.log" 2>/dev/null | grep -Eic 'error|fatal|segmentation|segfault|traceback|nan|floating point|forrtl|abort|killed|cannot open|no such file' || true)
 echo "$(date -Is) HEARTBEAT case=$NAME elapsed=$((NOW-START)) pid=$PID ps=[$PS] frames=$FR latest_frame_mtime=$OM run_log_mtime=$LM fatal_patterns=$FATAL" >> "$RUN/watchdog.log"
 if [ "$LM" -le "$LAST_LOG" ] && [ "$OM" -le "$LAST_OUT" ] && [ $((NOW-START)) -ge 180 ] && echo "$PS" | grep -Eq ' 0\.0( |$)'; then echo "$(date -Is) POSSIBLE_STALL" >> "$RUN/watchdog.log"; kill "$PID" || true; fi
 LAST_LOG=$LM; LAST_OUT=$OM
done
wait "$PID"; RC=$?
echo "$(date -Is) EXIT rc=$RC frames=$(find "$OUT" -name 'fort.q????' | wc -l)" >> "$RUN/watchdog.log"
exit "$RC"
