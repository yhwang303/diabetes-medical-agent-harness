"""Portable candidate artifact with recorded checkpoint, preprocessing and inference sources."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 source=ROOT/'results/O07_bounded040_rl';dest=ROOT/'artifacts/O07_research_candidate';dest.mkdir(parents=True,exist_ok=False)
 for a,b in [(source/'export_0050000.npz','policy.npz'),(source/'export_0050000.json','policy.json'),(source/'config.json','training_config.json'),(source/'provenance.json','training_provenance.json'),(ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json','normalization.json'),(ROOT/'numpy_policy.py','numpy_policy.py'),(ROOT/'research_inference.py','research_inference.py'),(ROOT/'simulation_contract_v3_single_seed.json','evaluation_contract.json')]:shutil.copy2(a,dest/b)
 (dest/'requirements.txt').write_text('numpy>=1.24,<3\n')
 (dest/'README.md').write_text("""# O07 研究推理候选包

运行：`OPENBLAS_NUM_THREADS=1 python research_inference.py --bundle . < example_request.json`

输入是按本包normalization.json处理的72×22观测历史及固定的最初72条已观测5分钟基础输注量（U）。固定参考在后续调用中不滚动重算，每位患者独立绑定，最大年龄66小时对应本轮3天仿真范围。输出为接下来5分钟连续基础率U/h以及等价区间总量U，绝不是单次注射剂量；不静默泵量化。

食物/运动的缺失值、mask及年龄编码必须保留；不得把没有运动记录解释为没有运动。example_request.json来自公开模拟患者开发轨迹，仅用于可重复推理演示。

本包是研究模型交付，不是Core注册执行器。manifest中的SHA用于本包一致性核验，不是外部信任签名；patient_id一致性也不是来源认证。后续必须由Core验证真实来源、适用性、时效、两模型依赖和发布权限。本轮不修改Harness可信规则。

单训练seed260915、50,000更新、全量Loop训练数据；冻结参考＋±0.4 U/h残差ReBRAC。参考是历史实际输注，不是验证过的医学需求；范围不是医学安全界限。模拟器无运动生理机制。最终效果以外部HTML/T01封存评价为准，不能仅凭本包存在判断验收通过。
""")
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
 meta={'model_id':'loop-reference-bounded-rebrac-O07-step50000-seed260915','files_sha256':files,'normalization_sha256':files['normalization.json'],'training_source_checkpoint_sha256':json.loads((dest/'policy.json').read_text())['source_sha256'],'reference_max_age_hours':66,'scope':'adult T1D public-simulator research candidate; fixed observed6h reference, 5min basal rate; no clinical deployment or Core registration','acceptance_status':'pending_full_development_and_sealed_evaluation','clinical_ready':False}
 (dest/'manifest.json').write_text(json.dumps(meta,indent=2));print(dest)
if __name__=='__main__':main()
