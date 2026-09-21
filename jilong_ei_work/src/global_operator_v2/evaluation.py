"""Shared post-freeze streaming evaluation and immutable-artifact guards."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import torch
from .dataset import build_features
from .losses import project_physical

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:Path(tmp).write_text(json.dumps(payload,indent=2,sort_keys=True));os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

@torch.no_grad()
def streaming_rollout(model,store,row,transform,normalizer,device,steps=144,on_state=None):
    """The single autoregressive core for VAL, TEST, holdout and H0 evaluation."""
    index=int(store.rows.index[store.rows.scenario_id.eq(row.scenario_id)][0]);previous,current,_,params,time,_=store.sample(index,0)
    tensor=lambda x:torch.from_numpy(x).unsqueeze(0).to(device);previous,current,params=tensor(previous),tensor(current),tensor(params);static=torch.from_numpy(store.static).unsqueeze(0).to(device)
    if on_state:on_state(0,current)
    for step in range(steps):
        prediction=project_physical(transform.decode(model(build_features(previous,current,static,params,torch.tensor([time+step/144.],device=device),transform,normalizer),transform.encode(current))))
        previous,current=current,prediction
        if on_state:on_state((step+1)*10,current)

def freeze_best_candidate(source,destination,manifest,metadata):
    """Copy exactly the selected checkpoint and record hashes for reproducibility."""
    source,destination=Path(source),Path(destination)
    if not source.exists():raise RuntimeError('BEST_CANDIDATE_MISSING')
    destination.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=destination.parent,suffix='.pt.tmp');os.close(fd)
    try:
        with source.open('rb') as inp,Path(tmp).open('wb') as out:out.write(inp.read())
        os.replace(tmp,destination)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    source_hash,destination_hash=sha256(source),sha256(destination)
    if source_hash!=destination_hash:raise RuntimeError('FREEZE_HASH_MISMATCH')
    atomic_json(manifest,{**metadata,'source_checkpoint':str(source),'sha256':source_hash})
    return source_hash

def final_holdout_lock(path,checkpoint_path,config_hash,metrics_hash,force=False):
    path=Path(path)
    if path.exists() and not force:raise RuntimeError('FINAL_HOLDOUT_ALREADY_EVALUATED')
    payload={'checkpoint_sha256':sha256(checkpoint_path),'config_hash':config_hash,'metrics_hash':metrics_hash,'primary_evaluation':not force}
    if force:payload['status']='NON_PRIMARY_RECOMPUTE'
    atomic_json(path,payload);return payload
