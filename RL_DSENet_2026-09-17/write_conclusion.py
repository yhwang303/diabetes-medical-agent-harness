"""Write a manuscript paragraph from completed evidence, including negative findings."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    final=json.loads((ROOT/'paper_tables/final_control_analysis.json').read_text());risk=json.loads((ROOT/'paper_tables/control_risk_audit.json').read_text());forecast=json.loads((ROOT/'results/F00_forecast_heldout/evaluation.json').read_text())
    ours=final['methods']['F11_actor']['aggregate'];hold=final['methods']['F01_hold']['aggregate'];planner=final['methods']['F10_planner']['aggregate']
    def ci(name,key):
        r=final['paired_comparisons'][name][key]
        return '%.2f pp (95%% patient-cluster CI %.2f to %.2f)'%(r['difference'],*r['ci95'])
    rmse='/'.join('%.2f'%forecast['summary_patient_equal']['dsenet'][str(h)]['rmse_mg_dl'] for h in (30,60,120,240))
    worst=max(risk['methods']['F11_actor']['patients'],key=lambda p:p['tbr70_pct'] or 0)
    text=f'''We adapted the DSENet dual-stream predictor to patient-separated Loop histories, preserving its global bidirectional Mamba stream, local stream, router and fusion. A fixed-budget, single-seed development search selected global patch length/stride 12/3 and local patch length/stride 2/1. The frozen predictor achieved patient-equal RMSEs of {rmse} mg/dL at 30/60/120/240 minutes on held-out Loop patients (effective n=32/32/31/31). These forecasting results do not establish treatment efficacy.

For basal-rate control, we evaluated frozen policies in 60 matched three-day scenarios per method using 10 previously exposed virtual adults, two scenario seeds and three common meal-bolus conditions. All runs shared a six-hour nominal-basal warmup. The proposed policy achieved BG TIR {ours['bg']['tir_lower_bound_pct']:.2f}%, TBR70 {ours['bg']['tbr70_pct']:.2f}%, and TBR54 {ours['bg']['tbr54_pct']:.2f}%, with {ours['failures']}/60 early terminations and complete observation coverage. Relative to holding the last observed warmup basal rate, the TIR difference was {ci('F01_hold','tir_lower_bound_pct')}; however, TBR70 increased by {ci('F01_hold','tbr70_pct')}. Thus this comparison represents an efficacy-risk trade-off, not dominance on all endpoints.

Against a non-RL planner sharing the same DSENet world model, reward and seven candidate plans, the TIR difference was {ci('F10_planner','tir_lower_bound_pct')}, while the TBR70 difference was {ci('F10_planner','tbr70_pct')}. The evidence therefore supports reduced hypoglycemia relative to matched planning, but not a statistically resolved TIR advantage over that planner. Existing BC, ReBRAC, FQL and earlier project baselines were re-evaluated without retraining. Cross-system comparisons differ in training information, action support and planning budget, and should not be interpreted as an isolated algorithmic effect.

Mean outcomes conceal substantial individual risk: adult{worst['patient']:03d} had TBR70 {worst['tbr70_pct']:.2f}% and TBR54 {worst['tbr54_pct']:.2f}%, with {worst['prolonged_under54_120min_events']} episodes of continuously remaining below 54 mg/dL for at least 120 minutes. These results do not establish a safe or reliable clinical controller. All learned models use one training seed (260915); patient-cluster intervals exclude training-seed uncertainty and are not multiplicity-adjusted. The final scenarios are new, but their virtual patients are not unseen. No final result was used to change the selected model.
'''
    (ROOT/'paper_tables/results_paragraph_en.md').write_text(text)

if __name__=='__main__':main()
