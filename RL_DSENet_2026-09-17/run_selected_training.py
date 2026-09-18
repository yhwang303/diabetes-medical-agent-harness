"""Run only the predeclared dependencies after every patch trial finishes."""
import json
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from select_patch import NAMES

def main():
    start=time.time()
    while not all((ROOT/'results'/name/'completion.json').exists() for name in NAMES):
        if time.time()-start>1800:raise TimeoutError('Patch trials did not complete')
        time.sleep(30)
    native=str(ROOT.parent/'.venv-native/bin/python');sim=str(ROOT.parent/'.venv/bin/python')
    def run(script,*args,python=native):
        print(json.dumps({'stage':script,'args':list(map(str,args))}),flush=True)
        subprocess.run([python,str(ROOT/script),*map(str,args)],check=True)
    run('select_patch.py')
    selected=json.loads((ROOT/'checks/patch_selection.json').read_text())['selected']
    run('train_world.py','--name','D05_selected_world','--forecast','results/'+selected+'/best.pt')
    run('evaluate_world.py','--name','D05_selected_world')
    assert json.loads((ROOT/'results/D05_selected_world/development_diagnostics.json').read_text())['decision_gate_passed']
    run('check_policy.py','--world','D05_selected_world')
    run('train_policy_bounded.py','--world','D05_selected_world','--name','D06_selected_policy')
    for mode,name,checkpoint in [('planner','E08_selected_planner_dev','D05_selected_world/world.pt'),('actor','E09_selected_actor_dev','D06_selected_policy/policy.pt'),('beam','E10_selected_beam_dev','D06_selected_policy/policy.pt')]:
        run('evaluate_control.py','--config',ROOT.parent/'RL_DITR创新_2026-09-16/configs/development_control.json','--mode',mode,'--bounded-reference','--checkpoint',ROOT/'results'/checkpoint,'--name',name,'--batch-size','12',python=sim)
    print(json.dumps({'status':'selected_model_development_complete','final_test_not_started':True}),flush=True)

if __name__=='__main__':main()
