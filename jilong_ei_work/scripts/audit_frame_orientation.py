from __future__ import annotations
import argparse, json, sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--workspace",required=True)
a=p.parse_args()
root=Path(a.workspace)
sys.path.insert(0,str(root))
from src.physics_frame_adapter import orientation_audit
frame=sorted((root/"outputs"/"physics_smoke60"/"frames").glob("frame_*.npz"))[0]
report=orientation_audit(root/"data"/"downloads"/"corridor60s.npz",frame)
(root/"reports"/"FRAME_INDEX_ORIENTATION_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))

