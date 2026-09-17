import { useCallback, useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type Kind = 'input' | 'prediction' | 'policy' | 'safety' | 'reviews' | 'release';
const scenarios = { candidate: '正常占位', abstain: '策略弃权', unsupported: '输入不支持', error: '策略返回错误', invalid_output: '输出格式错误', parent_mismatch: '父预测错配', safety_rejected: '工程边界超限', timeout: '执行超时' };
type Scenario = keyof typeof scenarios;
type Card = { kind: Kind; status: 'verified' | 'waiting' | 'blocked'; reason: string | null; reference_id: string | null; digest: string | null; producer: string | null; version: string | null; origin: string | null };
type Evidence = { case_id: string; snapshot_id: string; source: 'synthetic' | 'historical' | 'simulation'; run_id: string | null; run_state: string; mode: 'eval' | 'research' | null; fixture_scenario: Scenario | null; missing_count: number; checked_at: number; release_allowed: boolean; clinical_validation: false; cards: Card[] };
const labels: Record<Kind, string> = { input: '输入快照', prediction: '血糖预测证据', policy: '策略证据', safety: '工程安全检查', reviews: '双审核绑定', release: '报告发布条件' };
const reasons: Record<string, string> = {
  MISSING_INPUT: '存在缺失输入，分析条件未满足。', MODEL_NOT_CONFIGURED: '真实模型尚未配置，没有可用模型证据。',
  RUN_NOT_STARTED: '尚未启动分析任务。', MISSING_PREDICTION: '尚未取得有效预测证据。', MISSING_POLICY: '尚未取得有效策略证据。',
  MODEL_EVIDENCE_REQUIRED: '需要有效的双模型证据。', REVIEW_REQUIRED: '需要同一报告版本的两项审核均通过。',
  REVIEW_REJECTED: '审核未通过，不能发布。', REVIEW_INVALID: '审核绑定失效，不能复用旧审核。',
  CANCELLED: '任务已取消，原有证据不再授予发布资格。', SNAPSHOT_CHANGED: '输入已更新，旧任务绑定已失效。',
  VERSION_REVOKED: '依赖版本已撤销。', RUN_EXPIRED: '任务已过期，需要重新分析。', FIXTURES_DISABLED: '当前服务未启用工程替身。',
  EXECUTOR_TIMEOUT: '执行超时，没有接受迟到结果。', EXECUTOR_ERROR: '执行失败，没有有效结果。',
  EXECUTOR_BUSY: '执行器暂不可用。', INVALID_MODEL_OUTPUT: '模型输出不符合合同。',
  ENGINEERING_SAFETY_REJECTED: '候选未通过工程边界检查。', RESEARCH_RULES_NOT_CONFIGURED: '真实研究规则尚未配置。',
  POLICY_ABSTAIN: '策略已弃权，没有候选动作。', POLICY_UNSUPPORTED: '策略不支持此输入。', POLICY_ERROR: '策略执行失败。',
  FORECAST_PARENT_MISMATCH: '策略引用的预测与本次任务不一致。', INPUT_DIGEST_MISMATCH: '模型结果的输入摘要不一致。',
  TEMPLATE_MISMATCH: '报告与受控模板不一致。', EVIDENCE_CHANGED: '绑定证据已改变，需要重新审核。',
  FIXTURE_REQUIRES_SYNTHETIC_CASE: '占位检查仅接受声明为合成的数据。',
  INVALID_STAGE: '该任务正在执行或当前阶段不允许操作，请刷新状态后重试。',
  CORE_UNAVAILABLE: '无法确认执行结果，请恢复连接后重试；重试会复用本次请求标识。',
  ATTEMPT_BUDGET_EXHAUSTED: '本任务已用完三次尝试，不能继续重试。',
  WORKER_CPU_LIMIT: '执行进程超过 CPU 时间限制，已终止。', WORKER_MEMORY_LIMIT: '执行进程超过内存限制，已终止。',
  WORKER_OUTPUT_LIMIT: '执行输出超过大小限制，结果未接受。', WORKER_FILE_LIMIT: '执行进程触及单文件大小限制。',
  WORKER_FD_LIMIT: '执行进程触及文件句柄限制。', WORKER_CRASHED: '执行进程异常退出，没有有效结果。',
  WORKER_PROTOCOL_ERROR: '执行进程响应不符合传输合同，结果未接受。',
  WORKER_MONITOR_UNAVAILABLE: '资源监测不可用，已停止执行。', WORKER_LIMITS_UNAVAILABLE: '资源限制未能启用，已停止执行。',
  WORKER_EXECUTOR_NOT_REGISTERED: '执行器未通过独立进程注册校验。', CORE_SHUTTING_DOWN: 'Core 正在关闭，执行已停止。',
};
const verified: Record<Kind, string> = {
  input: '快照结构完整且摘要一致；不代表数据的医学真实性已验证。', prediction: '登记作业、执行器版本与输入绑定已复核。',
  policy: '登记作业、输入与父预测绑定已复核。', safety: '已通过当前工程规则；不代表临床安全。',
  reviews: '同一最终稿及证据摘要的两项审核绑定均有效。', release: '本次读取时前置条件满足；实际发布仍需 Core 再次核验。',
};
export function EvidencePanel({ caseId, snapshotId, onConnected }: { caseId: string; snapshotId: string; onConnected: (value: boolean) => void }) {
  const [data, setData] = useState<Evidence | null>(null);
  const [error, setError] = useState('');
  const [checking, setChecking] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [executionMessage, setExecutionMessage] = useState('');
  const [scenario, setScenario] = useState<Scenario>('abstain');
  const pending = useRef<{ key: string; scenario: Scenario | null } | null>(null);
  const busy = useRef(false);
  const mounted = useRef(false);
  const sequence = useRef(0);
  const refresh = useCallback(async () => {
    const current = ++sequence.current;
    setChecking(true);
    try {
      const next = await invoke<Evidence>('case_evidence', { caseId });
      if (current !== sequence.current) return;
      onConnected(true);
      if (next.case_id !== caseId || next.snapshot_id !== snapshotId) {
        setData(null); setError('病例快照已变化，请重新连接病例库并选择当前记录。');
      } else { setData(next); setError(''); }
    } catch {
      if (current === sequence.current) { onConnected(false); setData(null); setError('当前证据状态无法核验，已撤下旧状态。请重新打开应用恢复本地服务。'); }
    } finally { if (current === sequence.current) setChecking(false); }
  }, [caseId, snapshotId, onConnected]);
  useEffect(() => {
    mounted.current = true;
    void refresh();
    const timer = setInterval(() => { if (!document.hidden) void refresh(); }, 5000);
    return () => { mounted.current = false; sequence.current++; clearInterval(timer); };
  }, [refresh]);
  const checkFixture = async (requestedScenario: Scenario | null) => {
    if (busy.current || !data || data.source !== 'synthetic' || data.missing_count > 0) return;
    busy.current = true; setExecuting(true); setExecutionMessage('');
    try {
      if (!pending.current || pending.current.scenario !== requestedScenario) pending.current = { key: crypto.randomUUID().replaceAll('-', ''), scenario: requestedScenario };
      const result = await invoke<{ state: string; reason: string | null; currently_valid: boolean }>(requestedScenario === null ? 'check_prediction' : 'run_rl_demo', {
        caseId, snapshotId, idempotencyKey: pending.current.key, ...(requestedScenario === null ? {} : { scenario: requestedScenario }),
      });
      pending.current = null;
      if (mounted.current) setExecutionMessage(result.currently_valid && result.state === 'SAFETY_ACCEPTED'
        ? '占位双模型与工程检查通过；尚未执行审核或发布，不代表临床安全。'
        : result.state === 'PREDICTION_ACCEPTED' && result.currently_valid ? '预测占位证据已登记。策略、审核与发布仍未执行。'
        : reasons[result.reason ?? ''] ?? '本次任务未取得可用预测证据，请查看当前闸门状态。');
    } catch (failure) {
      const code = typeof failure === 'string' ? failure : 'CORE_UNAVAILABLE';
      if (!['CORE_UNAVAILABLE', 'EXECUTOR_TIMEOUT', 'EXECUTOR_BUSY', 'EXECUTOR_ERROR', 'INVALID_STAGE'].includes(code)) pending.current = null;
      if (mounted.current) setExecutionMessage(reasons[code] ?? '本次检查被拒绝，没有放行后续步骤。');
    } finally {
      busy.current = false;
      if (mounted.current) { setExecuting(false); await refresh(); }
    }
  };
  return <section className="evidence-panel" aria-label="证据与闸门">
    <div className="timeline-heading"><h2>证据与闸门</h2><button className="evidence-refresh" onClick={() => void refresh()} disabled={checking}>{checking ? '正在复核…' : '刷新证据状态'}</button></div>
    <p className="timeline-caption">证据状态每 5 秒只读刷新；执行检查需要单独点击下方按钮。</p>
    {error && <p role="alert" className="evidence-unavailable">{error}</p>}
    {!data && !error && <p className="timeline-caption">正在读取 Core 证据状态…</p>}
    {data && <>
      <div className="fixture-notice"><strong>预测占位检查 · 仅合成数据</strong><p>将最后一个观测值重复为 6 个点，仅验证调用与证据登记，不是真实预测。不会启动 RL、审核或报告发布。</p><button className="evidence-refresh" disabled={executing || data.source !== 'synthetic' || data.missing_count > 0} onClick={() => void checkFixture(null)}>{pending.current?.scenario === null ? '重试本次预测占位检查' : '运行预测占位检查'}</button>{data.source !== 'synthetic' && <p>当前来源不是合成数据，禁止运行占位检查。</p>}{data.missing_count > 0 && <p>输入有缺失点，需先补充有效输入。</p>}</div>
      <div className="fixture-notice rl-demo"><strong>RL 工程替身 · 故障演示</strong><p>每次新演示建立独立任务，依次检查预测占位、策略占位与工程规则。故障由 Core 固定执行器产生；不训练 RL，不进行医学审核或发布报告。</p><label>演示场景 <select aria-label="RL 演示场景" value={scenario} disabled={executing} onChange={e => setScenario(e.target.value as Scenario)}>{Object.entries(scenarios).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><button className="evidence-refresh" disabled={executing || data.source !== 'synthetic' || data.missing_count > 0} onClick={() => void checkFixture(scenario)}>{pending.current?.scenario === scenario ? '重试本次 RL 演示' : '运行 RL 场景'}</button>{scenario === 'timeout' && <p>独立执行进程将在 Core 的 5 秒期限到达时被终止。</p>}{executing && <p role="status">正在执行工程检查，请等待 Core 返回状态…</p>}</div>
      {executionMessage && <p role="status" className="timeline-caption">{executionMessage}</p>}
      <div className={`gate-summary ${data.release_allowed ? 'gate-ready' : ''}`} role="status"><strong>{data.release_allowed ? '当前发布前置条件满足' : '发布条件尚未满足'}</strong><span>{data.run_id ? '已关联最近一次分析任务' : '尚未启动分析任务'}{data.missing_count > 0 ? ` · ${data.missing_count} 个缺失点` : ''}</span></div>
      {data.mode === 'eval' && <p className="fixture-notice">工程替身任务：模型与审核通过只表示工程验证，不是真实模型或医学审核。</p>}
      {data.fixture_scenario && <p className="fixture-notice">当前任务场景：{scenarios[data.fixture_scenario]} · {data.fixture_scenario}</p>}
      <div className="evidence-grid">{data.cards.map(card => <article key={card.kind} className={`evidence-card ${card.status}`}><div className="card-heading"><h3>{labels[card.kind]}</h3><span>{card.status === 'verified' ? '已复核' : card.status === 'blocked' ? '已阻断' : '待前置'}</span></div><p>{card.status === 'verified' ? verified[card.kind] : reasons[card.reason ?? ''] ?? '当前依赖尚未通过核验，不能据此放行。'}</p>
        {card.origin === 'fixture' && <span className="fixture-origin">fixture · 工程替身</span>}
        <details><summary>核验标识</summary><dl>{card.reason && <><dt>原因码</dt><dd>{card.reason}</dd></>}{card.reference_id && <><dt>引用 ID</dt><dd>{card.reference_id}</dd></>}{card.digest && <><dt>摘要</dt><dd>{card.digest}</dd></>}{card.producer && <><dt>执行器</dt><dd>{card.producer}</dd></>}{card.version && <><dt>版本</dt><dd>{card.version}</dd></>}{!card.reason && !card.reference_id && !card.version && <dd>依据其余前置卡片的本次核验结果。</dd>}</dl></details>
      </article>)}</div>
      <p className="evidence-stamp">最近核验：{new Date(data.checked_at * 1000).toLocaleTimeString('zh-CN', { hour12: false })} · {checking ? '正在更新' : '仅反映上述时点'}{data.run_id && <><br />任务：{data.run_id} · {data.run_state}</>}</p>
    </>}
  </section>;
}
