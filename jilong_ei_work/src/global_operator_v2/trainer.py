"""Closed-loop trainer primitives used by the formal launcher and bounded smoke tests."""
from __future__ import annotations
import os,random,tempfile,time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .dataset import build_features
from .losses import project_physical, step_loss, amplitude_guard, rollout_objective

CURRICULUM=(("A",2000,1),("B",2000,2),("C",4000,4),("D",6000,6))

def choose_amp(device: torch.device):
    # RVPI's spectral layer uses complex einsum after rfft2.  Its current CUDA
    # implementation rejects both ComplexBFloat16 and ComplexHalf, so enabling
    # autocast would crash or silently change the numerical operator.  Keep the
    # feature gate explicit; a future real-valued spectral kernel may return
    # BF16 first and FP16+GradScaler second here.
    return False,None

def save_checkpoint(path, model, optimizer, scheduler, step, transform, config, normalizer=None, stage=None, best_metric=None, stage_step=None, stage_updates_total=None, architecture=None, best_checkpoint_path=None, stage_best_score=None):
    """Write a complete resumable checkpoint atomically (including on Windows)."""
    path=__import__('pathlib').Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    guarded=bool(getattr(model,'momentum_state_guard',False))
    semantics={"bounded_residual":bool(getattr(model,'bounded_residual',False)),"delta_bounds":getattr(model,'delta_bounds',torch.empty(0)).detach().cpu().tolist(),"momentum_state_guard":guarded,"momentum_state_bounds":getattr(model,'momentum_state_bounds',torch.empty(0)).detach().cpu().tolist() if guarded else None,"momentum_envelope_version":getattr(model,'momentum_envelope_version',None)}
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,
             "checkpoint_version":"2.5","step":step,"stage_name":stage,"stage_step":stage_step,"stage_updates_total":stage_updates_total,"global_step":step,"best_val_score":best_metric,"best_metric":best_metric,"best_checkpoint_path":best_checkpoint_path,"stage_best_score":stage_best_score,"transform":transform.to_dict(),"feature_normalizer":normalizer.to_dict() if normalizer else None,"delta_normalization":getattr(model,'delta_normalization',None),"model_semantics":semantics,"config":config,"config_hash":__import__('hashlib').sha256(__import__('json').dumps(config,sort_keys=True).encode()).hexdigest(),"architecture":architecture,"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.pt.tmp');os.close(fd)
    try: torch.save(payload,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def restore_checkpoint(path, model, optimizer=None, scheduler=None):
    ck=torch.load(path,map_location="cpu",weights_only=False);semantics=ck.get('model_semantics')
    if semantics:
        expected=bool(semantics.get('bounded_residual'))
        if expected and not hasattr(model,'delta_bounds'):raise RuntimeError('DELTA_MODEL_SEMANTICS_MISMATCH')
        if hasattr(model,'delta_bounds'):
            bounds=torch.as_tensor(semantics.get('delta_bounds',[]),dtype=model.delta_bounds.dtype)
            if bool(getattr(model,'bounded_residual',False))!=expected or bounds.shape!=model.delta_bounds.shape or not torch.equal(bounds,model.delta_bounds.detach().cpu()):raise RuntimeError('DELTA_MODEL_SEMANTICS_MISMATCH')
    expected_guard=bool((semantics or {}).get('momentum_state_guard',False));requested_guard=bool(getattr(model,'momentum_state_guard',False))
    if expected_guard:
        bounds=torch.as_tensor((semantics or {}).get('momentum_state_bounds',[]),dtype=model.momentum_state_bounds.dtype)
        if not requested_guard or bounds.shape!=model.momentum_state_bounds.shape or not torch.equal(bounds,model.momentum_state_bounds.detach().cpu()):raise RuntimeError('MOMENTUM_GUARD_SEMANTICS_MISMATCH')
    elif requested_guard:
        ck['checkpoint_migration']='ADD_TRAIN_ONLY_MOMENTUM_STATE_GUARD'
    state_dict=dict(ck['model'])
    if not expected_guard and requested_guard:state_dict.pop('momentum_state_bounds',None)
    missing,unexpected=model.load_state_dict(state_dict,strict=False)
    if unexpected or set(missing)-{'momentum_state_bounds'}:raise RuntimeError(f'CHECKPOINT_MODEL_STATE_MISMATCH missing={missing} unexpected={unexpected}')
    if optimizer and ck.get("optimizer"):optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):scheduler.load_state_dict(ck["scheduler"])
    torch.set_rng_state(ck["torch_rng"]);np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
    if torch.cuda.is_available() and ck.get("cuda_rng"):torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck

def train_stage(model,optimizer,scheduler,state,updates,batch_for_step,loss_for_batch,checkpoint_every=250,on_update=None):
    """Reusable stateful training loop; sampling itself is supplied statelessly.

    The launcher supplies a real frame batch and closed-loop ``rollout_loss``;
    keeping those I/O concerns outside this primitive makes resume behaviour
    testable without data or CUDA.
    """
    model.train()
    for _ in range(state.stage_step,updates):
        if torch.cuda.is_available():torch.cuda.synchronize()
        started=time.perf_counter()
        batch=batch_for_step(state.global_step)
        optimizer.zero_grad(set_to_none=True);loss,details=loss_for_batch(batch)
        if not torch.isfinite(loss):raise RuntimeError(f'NONFINITE_TRAINING_LOSS stage={state.current_stage} global_step={state.global_step}')
        loss.backward()
        if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):raise RuntimeError(f'NONFINITE_GRADIENT stage={state.current_stage} global_step={state.global_step}')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step()
        if torch.cuda.is_available():torch.cuda.synchronize()
        details=dict(details);details['total_loss']=loss.detach();details['step_seconds']=time.perf_counter()-started
        state.stage_step+=1;state.global_step+=1
        if on_update:on_update(state,details,checkpoint_every)
    return state

def is_improvement(candidate,best):
    """Strict ordering prevents a later worse validation from replacing best."""
    return best is None or candidate<best

def rollout_loss(model, transform, normalizer, previous, current, targets, static, params, times, active, cell_area, amp_dtype=None, gradient_checkpointing=True, burn_in_steps=0, burn_targets=None, teacher_current=None, delta_normalization=None, weighting=None):
    """Closed loop with no-grad burn-in followed by gradient-tracked BPTT.

    Burn-in exposes the model to its own state distribution without retaining a
    graph.  Teacher fields remain the only source of delta/change weighting.
    """
    burn_targets=[] if burn_targets is None else burn_targets
    if len(burn_targets)!=int(burn_in_steps):raise ValueError('burn target count mismatch')
    pred=current
    with torch.no_grad():
        for burn_step in range(int(burn_in_steps)):
            features=build_features(previous,pred,static,params,times+(.0069444*burn_step),transform,normalizer)
            pred=project_physical(transform.decode(model(features,transform.encode(pred))),active)
            previous=pred if burn_step==0 else previous
            # History must progress with two model states, not reuse an old GT.
            previous,current= current,pred
            pred=current
    previous,pred=previous,pred;reference=current if teacher_current is None else teacher_current
    losses=[];amps=[];steps=len(targets)
    for k in range(steps):
        features=build_features(previous,pred,static,params,times+(.0069444*(k+int(burn_in_steps))),transform,normalizer)
        encoded=transform.encode(pred)
        diagnostics_supported=hasattr(model,'bounded_residual')
        fn=(lambda f,e:model(f,e,return_diagnostics=True)) if diagnostics_supported else (lambda f,e:(model(f,e),None))
        with torch.autocast(device_type=pred.device.type,enabled=amp_dtype is not None,dtype=amp_dtype): out,raw_over_bound=checkpoint(fn,features,encoded,use_reentrant=False) if gradient_checkpointing and len(targets)>=2 else fn(features,encoded)
        nextp=project_physical(transform.decode(out),active);target_t=transform.encode(targets[k]);reference_t=transform.encode(reference)
        source_active=bool(float((times[0]+(.0069444*(k+int(burn_in_steps)))).detach().cpu())*1440<30)
        scales=None if delta_normalization is None else delta_normalization.scales
        term,detail=step_loss(out,target_t,reference,nextp,targets[k],active,cell_area,encoded,reference_t,scales,static,source_active,weighting)
        if raw_over_bound is not None:
            rms=raw_over_bound.square().mean((0,2,3)).sqrt();saturation=(raw_over_bound.abs()>3).float().mean((0,2,3));detail.update({f'raw_delta_over_bound_rms_{name}':value for name,value in zip(('h','hu','hv','c','ice','dz'),rms)});detail.update({f'saturation_fraction_{name}':value for name,value in zip(('h','hu','hv','c','ice','dz'),saturation)});detail['max_channel_saturation_fraction']=saturation.max()
        losses.append(term);amps.append(amplitude_guard(out,target_t,active));previous,pred=pred,nextp;reference=targets[k]
    total,parts=rollout_objective(losses,amps);return total,{**parts,**detail,'burn_in_steps':int(burn_in_steps)},pred
