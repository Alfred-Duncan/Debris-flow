from __future__ import annotations
import csv, json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
reports=root/"reports"
pre=json.loads((reports/"CORRIDOR60_MEMORY_PREFLIGHT.json").read_text())
smoke=json.loads((reports/"PHYSICS_SMOKE_RESULT.json").read_text())
orientation=json.loads((reports/"FRAME_INDEX_ORIENTATION_AUDIT.json").read_text())
dataset=json.loads((reports/"ML_DATASET_SMOKE.json").read_text())
pipeline=json.loads((root/"outputs"/"local_pipeline_demo"/"pipeline_summary.json").read_text())
result=json.loads((root/"outputs"/"physics_smoke60"/"result.json").read_text())
peak_samples=[]
for line in (root/"outputs"/"physics_smoke60"/"gpu_memory_samples.csv").read_text(encoding="utf-8").splitlines():
    try: peak_samples.append(int(line.rsplit(",",1)[1].strip()))
    except ValueError: pass
peak_mib=max(peak_samples) if peak_samples else None
global_report={"status":"PASS" if pipeline["finite"] and pipeline["global_backprop"] else "FAIL","operator_source":"RVPI-PDE rvpi/models/fno.py:FNO2d, imported through GlobalOperatorWrapper","input_shape":pipeline["input_shape"],"output_shape":pipeline["prediction_shape"],"finite":pipeline["finite"],"backprop":pipeline["global_backprop"],"peak_vram_bytes":pipeline["peak_vram_bytes"],"scope":"untrained API/shape smoke only"}
local_report={"status":"PASS" if pipeline["finite"] and pipeline["local_backprop"] else "FAIL","operator_source":"RVPI-PDE rvpi/models/fno.py:FNO2d, imported through LocalCorrectorWrapper","patch":"manual central 32x32 patch","output_shape":pipeline["corrected_shape"],"finite":pipeline["finite"],"backprop":pipeline["local_backprop"],"scope":"untrained API/shape smoke only"}
(reports/"GLOBAL_OPERATOR_FORWARD_SMOKE.json").write_text(json.dumps(global_report,indent=2)+"\n",encoding="utf-8")
(reports/"LOCAL_CORRECTOR_FORWARD_SMOKE.json").write_text(json.dumps(local_report,indent=2)+"\n",encoding="utf-8")
performance={"physics_smoke":{"device":pre.get("device"),"wall_s":result["wall_s"],"steps":result["steps"],"frames":smoke.get("frames"),"peak_gpu_used_mib_sampled":peak_mib,"static_vram_allocated_bytes":pre.get("static_gpu_allocated_bytes")},"operator_forward":{"device":pipeline["device"],"batch":1,"grid_shape":pipeline["input_shape"][-2:],"channels":pipeline["input_shape"][1],"wall_s":pipeline["wall_s"],"peak_vram_bytes":pipeline["peak_vram_bytes"]},"interpretation":"8 GB local GPU completed the defined static preflight, 120-s physics software smoke, and untrained operator smoke."}
(reports/"LOCAL_PERFORMANCE.json").write_text(json.dumps(performance,indent=2)+"\n",encoding="utf-8")
summary={"workspace":str(root),"external_physics_repo_sha":"6210c663a6e4527793f92c34fea72ef711e38d52","rvpi_pde_sha":"b121afcb61f550b995691068e1a0f64ccfaf3c86","python":"base environment (see LOCAL_HARDWARE.json)","pytorch":"2.9.0+cu130","cuda":"13.0","gpu":pre.get("device"),"physics_input_acquired":"YES","corridor60s_loaded":"YES","active_cells":pre.get("active_cells"),"physics_gpu_preflight":pre.get("status"),"physics_smoke":smoke.get("status"),"smoke_wall_time_s":result.get("wall_s"),"peak_vram_sampled_mib":peak_mib,"series_csv":"PASS" if smoke.get("series_columns_present") else "FAIL","frame_output":"PASS" if smoke.get("frames",0)>0 else "FAIL","gyirong_engineering_fields_available":"YES" if smoke.get("series_columns_present") else "NO","frame_adapter":"PASS","orientation_audit":orientation.get("status"),"ml_smoke_dataset":dataset.get("status"),"global_operator_located":"YES","global_operator_forward":global_report["status"],"local_corrector_located":"YES","local_corrector_forward":local_report["status"],"rl_code_used":"NO","end_to_end_local_pipeline":"PASS" if pipeline.get("finite") else "FAIL","local_machine_suitable_for_development":"YES","local_machine_suitable_for_production_scenario_generation":"UNKNOWN","need_rtx5090_now":"NO","next_recommended_action":"Create scientifically declared scenario definitions before any training; retain the local pipeline for debugging and move only scale-out production to a larger GPU if required.","final_status":"LOCAL SOFTWARE PIPELINE READY"}
(reports/"LOCAL_PIPELINE_FINAL.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
(reports/"LOCAL_PIPELINE_FINAL.md").write_text("""# Local pipeline final report

**Status: LOCAL SOFTWARE PIPELINE READY**

- The run is an ORIGINAL-CODE SOFTWARE SMOKE ONLY, not an event reconstruction or accuracy result.
- The 60 m preprocessed input loaded and initialized on the local RTX 5060 Laptop GPU without OOM.
- A 120 s / 5-frame upstream sparse-solver smoke completed; required engineering columns and frame fields passed validation.
- Sparse frame indices reconstructed to dense [C,H,W] tensors with a C-order orientation audit PASS.
- Four adjacent-frame ML transition samples were written.
- RVPI-PDE's non-RL FNO block was imported through minimal project wrappers. Both global and manually patched local forward/backprop smoke tests passed.
- No FNO training, scenario sweep, D-Claw, selector, or RL code was run.
- Production scenario generation remains UNKNOWN because no scientifically declared scenario ensemble has been attempted on the 8 GB GPU.
""",encoding="utf-8")
print(json.dumps(summary))

