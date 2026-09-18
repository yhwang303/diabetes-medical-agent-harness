"""Create a separately versioned extension; keep the original scorer/engine intact."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'evaluate_control.py').read_text()
s=s.replace("ROOT=Path(__file__).resolve().parent", "ROOT=Path(__file__).resolve().parent\nBASE=ROOT.parent\nPROJECT=BASE.parent")
s=s.replace("ROOT.parent/'RL", "PROJECT/'RL")
s=s.replace("ROOT.parent/'.venv-native/bin/python'", "PROJECT/'.venv-native/bin/python'")
s=s.replace("choices=['actor','beam','planner','hold','nominal','legacy','ditr']", "choices=['actor','beam','planner','hold','nominal','legacy','ditr','external']")
s=s.replace("ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--common-limit',type=float,default=None);")
s=s.replace("manifest={'bounded_reference':", "manifest={'common_limit_u_h':args.common_limit,'scorer_unchanged':True,'bounded_reference':")
s=s.replace("str(ROOT/('legacy_policy_worker.py'", "str(BASE/('legacy_policy_worker.py'")
s=s.replace("if args.mode=='ditr':command=", "if args.mode=='external':command=[str(PROJECT/'.venv-native/bin/python'),str(ROOT/'external_worker.py'),'--checkpoint',str(Path(args.checkpoint).resolve())]\n  if args.mode=='ditr':command=")
s=s.replace("if args.mode=='legacy':", "if args.mode in ('legacy','external'):")
s=s.replace("hold_rates={}", "hold_rates={};projection_records=[]")
old="    for (conn,_,_,_),action in zip(pending,actions):conn.send({'action_u_h':action})"
new="""    if args.common_limit is not None:
     projected=[]
     for (_,_,job,_),state,action in zip(pending,states,actions):
      key=(job['patient'],job['group'],job['seed'])
      if key not in hold_rates:
       assert state[-1,6]>.5
       hold_rates[key]=float((state[-1,1]*0.14462788945609448+0.09945811581924525)*12)
      anchor=hold_rates[key];applied=float(np.clip(action,max(0,anchor-args.common_limit),min(20,anchor+args.common_limit)))
      projection_records.append({'patient':job['patient'],'group':job['group'],'seed':job['seed'],'decision':iterations,'anchor_u_h':anchor,'raw_u_h':action,'projected_u_h':applied})
      projected.append(applied)
     actions=projected
    for (conn,_,_,_),action in zip(pending,actions):conn.send({'action_u_h':action})"""
assert old in s;s=s.replace(old,new)
s=s.replace("  timing={'scope':", "  (out/'action_projection.json').write_text(json.dumps(projection_records,allow_nan=False))\n  timing={'scope':")
(ROOT/'evaluate_extended.py').write_text(s)
