import hashlib,json,math,struct,subprocess
from pathlib import Path
C=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256((C/p).read_bytes()).hexdigest()
files=['src2.f90','setrun.py','model_geometry_sourcefix.json','terrain/published_route_domain_64m.tt3','entrainment/erodible_thickness_e4_published.tt3','source/source_support_published_s0_floorfix.json','xdclaw']
build=json.load(open(C/'reports/MOMENTUM_SWEEP_BUILD.json'))
d={'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=C).strip(),'sha256':{p:sha(p) for p in files},'executable_reused':sha('xdclaw')==build['binary_sha256'],'previous_build_record':'reports/MOMENTUM_SWEEP_BUILD.json','frozen':{'T_s':90,'K_u':1,'W_ref_m':192,'source_cells':12,'source_area_m2':49152}}
(C/'reports/VOLUME_SWEEP_BASELINE_PROVENANCE.json').write_text(json.dumps(d,indent=2)+'\n')
# tt3 ascii header six values; data begins line 7
p=C/'entrainment/erodible_thickness_e4_published.tt3'
lines=p.read_text().splitlines(); head={}
for line in lines[:6]:
 a=line.split(); head[a[-1].lower()]=float(a[0])
ncols=int(head['ncols']); nrows=int(head['nrows']); cell=float(head['cellsize']); vals=[]
for line in lines[6:]: vals.extend(map(float,line.split()))
import numpy as np
x=np.array(vals); positive=x[x>0]
mat={'file':str(p.relative_to(C)),'sha256':sha(p),'ncols':ncols,'nrows':nrows,'cellsize_m':cell,'number_erodible_cells':int(positive.size),'erodible_physical_area_m2':float(positive.size*cell*cell),'h_e_m':4.0,'initial_available_erodible_material_volume_m3':float(positive.size*cell*cell*4.0),'mask_positive_thickness_min_m':float(positive.min()) if positive.size else None,'mask_positive_thickness_max_m':float(positive.max()) if positive.size else None}
(C/'reports/VOLUME_SWEEP_AVAILABLE_MATERIAL.json').write_text(json.dumps(mat,indent=2)+'\n')
rows=[]
for name,V in [('V2',2e6),('V4',4e6),('V6',6e6),('V10',1e7)]:
 T,W,g=90.,192.,9.81; q=.5*math.pi*V/T; h=((q/W)**2/g)**(1/3); u=(g*h)**.5
 rows.append({'case':name,'V_m3':V,'T_s':T,'K_u':1.0,'Q_peak_m3s':q,'h_ref_peak_m':h,'u_peak_ms':u,'V_t30_expected_m3':.25*V,'V_t60_expected_m3':.75*V,'V_t90_expected_m3':V})
import csv
with open(C/'results/VOLUME_SOURCE_SCENARIOS.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps({'executable_reused':d['executable_reused'],'available_material':mat,'source_rows':rows},indent=2))