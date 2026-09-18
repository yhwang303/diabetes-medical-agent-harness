"""Single-seed, full-origin forecast training; patient-level physical-unit metrics."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import json
import hashlib
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from forecast_model import Forecast, ROOT, DEFAULT
from forecast_data import Data, tensor

HORIZONS = (6, 12, 24, 48)

@torch.no_grad()
def evaluate(model, data, per_patient=128):
    model.eval(); records = []
    for pid, d in data.patients.items():
        positions = np.arange(len(d['starts'])) if per_patient == 0 else np.unique(np.linspace(0, len(d['starts'])-1, min(per_patient, len(d['starts']))).astype(int))
        sums = {name: np.zeros((48, 4)) for name in ('dsenet', 'persistence', 'trend')}
        for start in range(0, len(positions), 256):
            b = tensor(data.batch(pid, positions[start:start+256]))
            y = b['target']; mask = b['mask']
            x = b['state']; physical = x[:, :, 0] * model.scale + model.mean
            observed = x[:, :, 5] > .5
            index = torch.arange(72, device=x.device)[None].expand(len(x), -1)
            last = torch.where(observed, index, -1).max(1).values
            previous = torch.where(observed & (index <= (last-6)[:, None]), index, -1).max(1).values
            previous = torch.where(previous >= 0, previous, last)
            current = physical.gather(1, last[:, None])
            slope = (current - physical.gather(1, previous[:, None])) / (last-previous).clamp_min(1)[:, None]
            predictions = {'dsenet': model(x), 'persistence': current.expand(-1, 48),
                           'trend': current + slope * torch.arange(1, 49, device=x.device)[None]}
            for name, prediction in predictions.items():
                error = (prediction-y)*18
                low = mask & (y*18 < 70)
                values = torch.stack([(error.square()*mask).sum(0), (error.abs()*mask).sum(0),
                                      mask.sum(0), (error.square()*low).sum(0)], -1)
                sums[name] += values.cpu().numpy()
            if start == 0: low_counts = np.zeros(48)
            low_counts += low.sum(0).cpu().numpy()
        item = {'patient': pid, 'evaluated_origins': len(positions), 'methods': {}}
        for name, values in sums.items():
            item['methods'][name] = {str(h*5): {'rmse_mg_dl': float(np.sqrt(values[h-1, 0]/values[h-1, 2])) if values[h-1, 2] else None,
                                             'mae_mg_dl': float(values[h-1, 1]/values[h-1, 2]) if values[h-1, 2] else None,
                                             'count': int(values[h-1, 2]),
                                             'low_rmse_mg_dl': float(np.sqrt(values[h-1, 3]/low_counts[h-1])) if low_counts[h-1] else None,
                                             'low_count': int(low_counts[h-1])} for h in HORIZONS}
        records.append(item)
    summary = {}
    for name in sums:
        summary[name] = {}
        for h in HORIZONS:
            metrics = [r['methods'][name][str(h*5)] for r in records]
            summary[name][str(h*5)] = {key: float(np.mean([m[key] for m in metrics if m[key] is not None]))
                                      if any(m[key] is not None for m in metrics) else None for key in ('rmse_mg_dl', 'mae_mg_dl', 'low_rmse_mg_dl')}
            summary[name][str(h*5)]['valid_labels'] = sum(m['count'] for m in metrics)
    return {'summary_patient_equal': summary, 'patients': records,
            'selection_score': float(np.mean([summary['dsenet'][str(h*5)]['rmse_mg_dl'] for h in HORIZONS])),
            'units': 'mg/dL; internal model mmol/L', 'split': data.split, 'per_patient_limit': per_patient}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--name', default='D01_forecast'); ap.add_argument('--batch-size', type=int, default=256); ap.add_argument('--model-config'); args = ap.parse_args()
    out = ROOT/'results'/args.name; out.mkdir(parents=True, exist_ok=False)
    assert json.loads((ROOT/'checks/forecast_cuda.json').read_text())['status'] == 'passed'
    torch.manual_seed(260915); np.random.seed(260915); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    model_config=dict(DEFAULT)
    if args.model_config:
        overrides=json.loads(Path(args.model_config).read_text())
        if set(overrides)-{'m_patch_len','m_stride','patch_len','stride'}:raise ValueError('Patch search may only change the four patch hyperparameters')
        model_config.update(overrides)
    model = Forecast(model_config).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train, validation = Data('train'), Data('validation')
    config = {'seed': 260915, 'model': model.config, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'lr': 3e-4, 'selection': 'lowest mean patient-equal validation RMSE at30/60/120/240min;128 fixed origins per patient',
              'train_origins': train.total, 'validation_origins': validation.total, 'loss': 'mean observed squared error in mmol/L',
              'patient_split_before_windows': True, 'future_action_inputs': False}
    config['parameters']=sum(p.numel() for p in model.parameters())
    config['forecast_masks_manifest_sha256']=hashlib.sha256((ROOT/'forecast_masks/manifest.json').read_bytes()).hexdigest()
    (out/'sources').mkdir()
    for name in ('train_forecast.py','forecast_model.py','forecast_data.py'):
        shutil.copy2(ROOT/name,out/'sources'/name)
    config['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(ROOT.glob('*forecast*.py'))+list((ROOT/'dsenet').rglob('*.py'))}
    (out/'config.json').write_text(json.dumps(config, indent=2)); begin=time.time(); best=float('inf'); step=0
    with (out/'history.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            model.train(); seen=0; patients=set()
            for b in train.batches(args.batch_size, 260915+epoch, horizon=48):
                patients.add(b['patient']); seen+=len(b['state']); b=tensor(b)
                prediction=model(b['state']); loss=((prediction-b['target']).square()*b['mask']).sum()/b['mask'].sum()
                if not torch.isfinite(loss): raise ValueError('Nonfinite forecast loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step(); step+=1
                if step%100 == 0:
                    event={'epoch':epoch+1,'step':step,'seen':seen,'loss':float(loss),'elapsed_seconds':time.time()-begin}
                    log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
            assert seen==1653421 and len(patients)==225
            result=evaluate(model,validation);(out/('validation_epoch%d.json'%(epoch+1))).write_text(json.dumps(result,indent=2))
            checkpoint={'model':model.state_dict(),'config':config,'epoch':epoch+1,'step':step,'optimizer':optimizer.state_dict(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
            torch.save(checkpoint,out/'last.pt')
            if result['selection_score']<best:best=result['selection_score'];torch.save(checkpoint,out/'best.pt')
            event={'epoch_complete':epoch+1,'seen':seen,'patients':len(patients),'validation_score':result['selection_score'],'summary':result['summary_patient_equal']};log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
    model.load_state_dict(torch.load(out/'best.pt')['model']); result=evaluate(model,validation,per_patient=0)
    (out/'validation_full.json').write_text(json.dumps(result,indent=2))
    (out/'completion.json').write_text(json.dumps({'status':'forecast_training_complete_not_control_validation','epochs':args.epochs,'seed':260915,'elapsed_seconds':time.time()-begin},indent=2))

if __name__=='__main__':
    main()
