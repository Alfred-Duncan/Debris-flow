"""Frozen deterministic EngineeringROI-v1 scoring and allocation.

Production scoring is vectorized over all eligible patch cores.  The retained
reference implementation is synthetic-test-only and is never called by an
evaluator execution path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.global_operator_v2.oracle_refinement import PatchLayout, select_random
from .support_guard import support_mask

COMPONENTS = ("dynamic", "predicted_front", "support_risk", "engineering_section")
STRATEGIES = ("random_all", "random_support", "dynamic_only", "front_only", "support_risk_only",
              "section_only", "engineering_roi", "engineering_roi_no_diversity")


@dataclass(frozen=True)
class ROIStaticMetadata:
    layout: PatchLayout
    active: torch.Tensor
    route_tensor: torch.Tensor
    section_mask: torch.Tensor
    patch_index_map: torch.Tensor
    active_counts: torch.Tensor
    section_counts: torch.Tensor
    patch_ids: np.ndarray
    row_ids: np.ndarray
    col_ids: np.ndarray


def load_engineering_roi_config(path: str | Path) -> tuple[dict, str]:
    raw=Path(path).read_bytes();config=json.loads(raw.decode("utf-8"))
    expected={"version":"EngineeringROI-v1","wet_threshold_m":.05,"debris_front_h_threshold_m":.10,"debris_front_c_threshold":.05,"front_half_width_m":1000.,"shallow_margin_upper_h_m":.15,"budgets":[.05,.10,.20],"normalization":"positive_percentile_rank","candidate_rule":"support_overlap_and_positive_score","diversity_rule":"nonadjacent_first_then_fill"}
    if any(config.get(key)!=value for key,value in expected.items()) or config.get("components")!=list(COMPONENTS):raise RuntimeError("ENGINEERING_ROI_CONFIG_INVALID")
    weights=config.get("weights",{})
    if set(weights)!=set(COMPONENTS) or any(weights[name]!=.25 for name in COMPONENTS) or sum(weights.values())!=1.0 or config.get("truth_in_roi_scoring") is not False:raise RuntimeError("ENGINEERING_ROI_CONFIG_INVALID")
    return config,hashlib.sha256(raw).hexdigest()


def build_section_mask(transects: Mapping, shape: tuple[int,int]) -> torch.Tensor:
    height,width=map(int,shape);mask=torch.zeros((height,width),dtype=torch.bool);points=0
    if not isinstance(transects,Mapping) or not transects:raise RuntimeError("ENGINEERING_SECTION_METADATA_INVALID")
    for name,transect in transects.items():
        if not isinstance(transect,Mapping):raise RuntimeError(f"ENGINEERING_SECTION_METADATA_INVALID:{name}")
        rows,cols=transect.get("rows"),transect.get("cols")
        if rows is None or cols is None or len(rows)!=len(cols):raise RuntimeError(f"ENGINEERING_SECTION_METADATA_INVALID:{name}")
        for row,col in zip(rows,cols):
            row,col=int(row),int(col)
            if not(0<=row<height and 0<=col<width):raise RuntimeError(f"ENGINEERING_SECTION_METADATA_INVALID:{name}")
            mask[row,col]=True;points+=1
    if points==0 or not bool(mask.any()):raise RuntimeError("ENGINEERING_SECTION_METADATA_INVALID")
    return mask


def build_roi_static_metadata(layout: PatchLayout, active: torch.Tensor, route_chainage_m, section_mask: torch.Tensor, device=None) -> ROIStaticMetadata:
    """Pre-create route, section, and patch tensors once per evaluator session."""
    target=torch.device(device) if device is not None else active.device;mask=(active if active.ndim==4 else active[:,None]).to(target).bool()
    if mask.shape[0]!=1 or mask.shape[1]!=1:raise ValueError("ROI static active must have shape [1,1,H,W]")
    height,width=mask.shape[-2:];route=np.asarray(route_chainage_m,dtype=np.float64)
    if route.shape!=(height,width):raise ValueError("route_chainage_m shape mismatch")
    section=torch.as_tensor(section_mask,device=target,dtype=torch.bool)
    if tuple(section.shape)!=(height,width):raise ValueError("section_mask shape mismatch")
    index=torch.full((height,width),-1,device=target,dtype=torch.long);patch_ids=[];rows=[];cols=[]
    for number,patch in enumerate(layout.eligible):
        index[patch.r0:patch.r1,patch.c0:patch.c1]=number;patch_ids.append(patch.patch_id);rows.append(patch.row_id);cols.append(patch.col_id)
    valid=index>=0;count=len(patch_ids);active_counts=torch.zeros(count,device=target,dtype=torch.float64).scatter_add(0,index[valid],mask[0,0][valid].to(torch.float64));section_counts=torch.zeros(count,device=target,dtype=torch.float64).scatter_add(0,index[valid],section[valid].to(torch.float64))
    return ROIStaticMetadata(layout,mask,torch.as_tensor(route,device=target),section,index,active_counts,section_counts,np.asarray(patch_ids,dtype=np.int64),np.asarray(rows,dtype=np.int64),np.asarray(cols,dtype=np.int64))


def positive_percentile_rank(values) -> np.ndarray:
    raw=np.asarray(values,dtype=np.float64);out=np.zeros_like(raw);positive=np.isfinite(raw)&(raw>0);pool=raw[positive]
    if pool.size:out[positive]=np.asarray([(pool<=value).sum()/pool.size for value in pool],dtype=np.float64)
    return out


def patch_reduce_sum(field: torch.Tensor, metadata: ROIStaticMetadata) -> torch.Tensor:
    value=field[0,0] if field.ndim==4 else field
    if tuple(value.shape)!=tuple(metadata.patch_index_map.shape):raise ValueError("patch reduction shape mismatch")
    valid=metadata.patch_index_map>=0;out=torch.zeros(len(metadata.patch_ids),device=value.device,dtype=value.dtype)
    return out.scatter_add(0,metadata.patch_index_map[valid],value[valid])


def patch_reduce_mean(field: torch.Tensor, mask: torch.Tensor, metadata: ROIStaticMetadata) -> tuple[torch.Tensor,torch.Tensor]:
    value=field[0,0] if field.ndim==4 else field;where=mask[0,0] if mask.ndim==4 else mask
    count=patch_reduce_sum(where.to(value.dtype),metadata);total=patch_reduce_sum(value*where.to(value.dtype),metadata)
    return total/count.clamp_min(1),count


def _validate_states(current,provisional,encoded_current,encoded_provisional,metadata):
    values=(current,provisional,encoded_current,encoded_provisional)
    if any(value.ndim!=4 or value.shape[0]!=1 for value in values) or current.shape!=provisional.shape or encoded_current.shape!=encoded_provisional.shape or current.shape[-2:]!=encoded_current.shape[-2:] or encoded_current.shape[1]!=6:raise ValueError("EngineeringROI state shapes invalid")
    if tuple(current.shape[-2:])!=tuple(metadata.patch_index_map.shape):raise ValueError("EngineeringROI static metadata shape mismatch")


def compute_roi_components(current_physical: torch.Tensor, provisional_physical: torch.Tensor, encoded_current: torch.Tensor, encoded_provisional: torch.Tensor, global_delta_scales, metadata: ROIStaticMetadata, config: Mapping) -> dict:
    """Vectorized four-component scoring with no teacher/target/future inputs."""
    _validate_states(current_physical,provisional_physical,encoded_current,encoded_provisional,metadata);wet=float(config["wet_threshold_m"]);shallow=float(config["shallow_margin_upper_h_m"])
    support=support_mask(current_physical,provisional_physical,metadata.active,wet);scales=torch.as_tensor(global_delta_scales,device=encoded_current.device,dtype=encoded_current.dtype)[None,:,None,None]
    if scales.shape[1]!=6:raise ValueError("global_delta_scales must contain six values")
    dynamic=(((encoded_provisional-encoded_current)/scales).square().mean(1,keepdim=True)).sqrt();h,c=provisional_physical[:,:1],provisional_physical[:,3:4];route=metadata.route_tensor
    front_cells=(h[:,0]>float(config["debris_front_h_threshold_m"]))&(c[:,0]>float(config["debris_front_c_threshold"]))&torch.isfinite(route)[None];front_available=bool(front_cells.any().item());front_chainage=float(route[front_cells[0]].max().item()) if front_available else float("nan")
    front_zone=((torch.abs(route[None,None]-front_chainage)<=float(config["front_half_width_m"]))&metadata.active&support) if front_available else torch.zeros_like(support);risk=metadata.active&(((current_physical[:,:1]<wet)&(h>=wet))|((h>=wet)&(h<shallow)))
    support_overlap=patch_reduce_sum(support.to(torch.float64),metadata).to(torch.int64);dynamic_raw,_=patch_reduce_mean(dynamic,support,metadata);front_raw=patch_reduce_sum(front_zone.to(torch.float64),metadata)/metadata.active_counts.clamp_min(1);risk_raw=patch_reduce_sum(risk.to(torch.float64),metadata)/metadata.active_counts.clamp_min(1);section_overlap=patch_reduce_sum((metadata.section_mask[None,None]&support).to(torch.float64),metadata);section_raw=section_overlap/metadata.section_counts.clamp_min(1);section_raw=torch.where(metadata.section_counts>0,section_raw,torch.zeros_like(section_raw))
    raw={"dynamic":dynamic_raw.detach().cpu().numpy().astype(np.float64),"predicted_front":front_raw.detach().cpu().numpy(),"support_risk":risk_raw.detach().cpu().numpy(),"engineering_section":section_raw.detach().cpu().numpy()};ranks={name:positive_percentile_rank(raw[name]) for name in COMPONENTS};base=sum(float(config["weights"][name])*ranks[name] for name in COMPONENTS);overlap=support_overlap.detach().cpu().numpy();candidates=(overlap>0)&(base>0)
    return {"patches":tuple(metadata.layout.eligible),"patch_ids":metadata.patch_ids,"support_overlap":overlap,"raw":raw,"ranks":ranks,"base_score":base,"candidates":candidates,"predicted_front_chainage_m":front_chainage,"front_available":front_available}


def _reference_compute_roi_components(current_physical,provisional_physical,encoded_current,encoded_provisional,global_delta_scales,metadata,config) -> dict:
    """Slow loop reference retained strictly for synthetic equivalence tests."""
    _validate_states(current_physical,provisional_physical,encoded_current,encoded_provisional,metadata);wet=float(config["wet_threshold_m"]);shallow=float(config["shallow_margin_upper_h_m"]);support=support_mask(current_physical,provisional_physical,metadata.active,wet);scales=torch.as_tensor(global_delta_scales,device=encoded_current.device,dtype=encoded_current.dtype)[None,:,None,None];dynamic=(((encoded_provisional-encoded_current)/scales).square().mean(1,keepdim=True)).sqrt();h,c=provisional_physical[:,:1],provisional_physical[:,3:4];route=metadata.route_tensor;front_cells=(h[:,0]>float(config["debris_front_h_threshold_m"]))&(c[:,0]>float(config["debris_front_c_threshold"]))&torch.isfinite(route)[None];available=bool(front_cells.any().item());chainage=float(route[front_cells[0]].max().item()) if available else float("nan");front=((torch.abs(route[None,None]-chainage)<=float(config["front_half_width_m"]))&metadata.active&support) if available else torch.zeros_like(support);risk=metadata.active&(((current_physical[:,:1]<wet)&(h>=wet))|((h>=wet)&(h<shallow)));raw={name:[] for name in COMPONENTS};overlap=[]
    for patch in metadata.layout.eligible:
        core=(slice(patch.r0,patch.r1),slice(patch.c0,patch.c1));s=support[0,0][core];a=metadata.active[0,0][core];sec=metadata.section_mask[core];count=int(a.sum().item());n=int(s.sum().item());overlap.append(n);raw['dynamic'].append(float(dynamic[0,0][core][s].mean().item()) if n else 0.);raw['predicted_front'].append(float(front[0,0][core].sum().item()/max(count,1)));raw['support_risk'].append(float(risk[0,0][core].sum().item()/max(count,1)));den=int(sec.sum().item());raw['engineering_section'].append(float((sec&s).sum().item()/den) if den else 0.)
    raw={name:np.asarray(values,dtype=np.float64) for name,values in raw.items()};ranks={name:positive_percentile_rank(raw[name]) for name in COMPONENTS};base=sum(float(config['weights'][name])*ranks[name] for name in COMPONENTS);overlap=np.asarray(overlap,dtype=np.int64);return {"patches":tuple(metadata.layout.eligible),"patch_ids":metadata.patch_ids,"support_overlap":overlap,"raw":raw,"ranks":ranks,"base_score":base,"candidates":(overlap>0)&(base>0),"predicted_front_chainage_m":chainage,"front_available":available}


def _ordered(indices,scores,patches):return sorted(map(int,indices),key=lambda index:(-float(scores[index]),int(patches[index].patch_id)))
def _diverse(indices,scores,patches,maximum):
    ranked=_ordered(indices,scores,patches);chosen=[]
    for index in ranked:
        patch=patches[index]
        if all(max(abs(patch.row_id-patches[other].row_id),abs(patch.col_id-patches[other].col_id))>=2 for other in chosen):
            chosen.append(index)
            if len(chosen)==maximum:return chosen,len(chosen),0
    first=len(chosen)
    for index in ranked:
        if index not in chosen:
            chosen.append(index)
            if len(chosen)==maximum:break
    return chosen,first,len(chosen)-first


def select_engineering_roi(current_physical,provisional_physical,encoded_current,encoded_provisional,global_delta_scales,metadata,config,budget_fraction,strategy="engineering_roi",seed=None):
    if strategy not in STRATEGIES:raise ValueError("ENGINEERING_ROI_STRATEGY_INVALID")
    if float(budget_fraction) not in tuple(map(float,config['budgets'])):raise ValueError("ENGINEERING_ROI_BUDGET_INVALID")
    info=compute_roi_components(current_physical,provisional_physical,encoded_current,encoded_provisional,global_delta_scales,metadata,config);patches,maximum=info['patches'],metadata.layout.count_for_budget(budget_fraction)
    if strategy=='random_all':selected=tuple(select_random(metadata.layout,maximum,int(seed)));selected_indices=[next(i for i,p in enumerate(patches) if p.patch_id==value.patch_id) for value in selected];first,second=len(selected),0;candidate_count=len(metadata.layout.eligible)
    elif strategy=='random_support':
        candidates=np.flatnonzero(info['support_overlap']>0);candidates=np.asarray(sorted(candidates,key=lambda index:patches[int(index)].patch_id),dtype=int);rng=np.random.default_rng(int(seed));picks=rng.choice(candidates,size=min(maximum,len(candidates)),replace=False) if len(candidates) else np.empty(0,dtype=int);selected_indices=sorted(map(int,picks),key=lambda index:patches[index].patch_id);selected=tuple(patches[index] for index in selected_indices);first,second=len(selected),0;candidate_count=len(candidates)
    else:
        key={'dynamic_only':'dynamic','front_only':'predicted_front','support_risk_only':'support_risk','section_only':'engineering_section'}.get(strategy);scores=info['ranks'][key] if key else info['base_score'];candidates=np.flatnonzero((info['support_overlap']>0)&(scores>0));candidate_count=len(candidates)
        if strategy=='engineering_roi_no_diversity':selected_indices=_ordered(candidates,scores,patches)[:maximum];first,second=len(selected_indices),0
        else:selected_indices,first,second=_diverse(candidates,scores,patches,maximum)
        selected=tuple(patches[index] for index in selected_indices)
    average=lambda name:float(np.mean(info['ranks'][name][selected_indices])) if selected_indices else 0.
    return selected,{"budget_fraction":float(budget_fraction),"budget_max_count":maximum,"candidate_count":candidate_count,"selected_count":len(selected),"actual_active_fraction":metadata.layout.selected_active_fraction(selected),"predicted_front_chainage_m":info['predicted_front_chainage_m'],"front_available":info['front_available'],"mean_selected_dynamic_rank":average('dynamic'),"mean_selected_front_rank":average('predicted_front'),"mean_selected_support_risk_rank":average('support_risk'),"mean_selected_section_rank":average('engineering_section'),"mean_selected_base_score":float(np.mean(info['base_score'][selected_indices])) if selected_indices else 0.,"selected_section_patch_count":int(sum(info['ranks']['engineering_section'][index]>0 for index in selected_indices)),"first_pass_count":first,"second_pass_count":second}
