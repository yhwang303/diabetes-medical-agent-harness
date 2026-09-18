from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
prefix='RL_DSENet_2026-09-17/';rev=prefix+'revision/'
legacy='RL进阶对比_2026-09-15/results/'
methods=[]
def add(key,label,venue,year,checkpoint,mode,main,native,training='Loop only'):
 methods.append(dict(key=key,label=label,venue=venue,year=year,checkpoint=checkpoint,mode=mode,main_folder=main,native_folder=native,training_information=training))
add('hold','Constant basal','Control','—',None,'hold',prefix+'results/F01_hold',prefix+'results/F01_hold','Observed warm-up basal')
add('bc','BC','Imitation','—',legacy+'R00_bc_pilot/export_0050000.npz','legacy',rev+'results/M_bc',prefix+'results/F02_BC')
add('td3bc','TD3+BC','NeurIPS',2021,rev+'results/td3bc_seed260915/last.pt','external',rev+'results/M_td3bc',rev+'results/N_td3bc')
add('iql','IQL','ICLR',2022,rev+'results/iql_seed260915/last.pt','external',rev+'results/M_iql',rev+'results/N_iql')
add('rebrac','ReBRAC','NeurIPS',2023,legacy+'R21_rebrac_bc100/export_0050000.npz','legacy',rev+'results/M_rebrac',prefix+'results/F03_ReBRAC_BC100')
add('fql','FQL','ICML',2025,legacy+'R22_fql_alpha100/export_0050000.npz','legacy',rev+'results/M_fql',prefix+'results/F04_FQL_alpha100')
add('lom','LOM','ICLR',2025,rev+'results/lom_seed260915/last.pt','external',rev+'results/M_lom',rev+'results/N_lom')
add('gfp','GFP','ICLR',2026,rev+'results/gfp_seed260915/last.pt','external',rev+'results/M_gfp',rev+'results/N_gfp')
add('ditr','RL-DITR*','Nature Medicine',2023,'RL_DITR创新_2026-09-16/results/R03_reference_categorical/policy_last.pt','ditr',rev+'results/M_ditr',rev+'results/N_ditr','Loop only; continuous basal task adaptation')
add('ours','DSENet-RL (ours)','This work','—',prefix+'results/D06_selected_policy/policy.pt','actor',prefix+'results/F11_actor',prefix+'results/F11_actor','Loop + H02 context + paired simulation')
contract={'methods':methods,'patients':list(range(1,11)),'episode_count':60,'primary_method':'ours',
          'main_common_limit_u_h':.25,'training_seed':260915,'purpose':'same evaluation and common action envelope; training information differs explicitly',
          'no_test_tuning':True,'scenario_config':str((ROOT/'configs/scenarios.json').relative_to(ROOT.parents[1]))}
(ROOT/'configs/comparison_manifest.json').write_text(json.dumps(contract,indent=2,ensure_ascii=False))
