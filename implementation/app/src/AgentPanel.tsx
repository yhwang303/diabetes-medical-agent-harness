import { useCallback, useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type Task = { id: string; case_id: string; snapshot_id: string; state: string; phase: string; reason: string | null;
  run_state: string; active: boolean; currently_valid: boolean; invalid_reason: string | null; checked_at: number;
  progress: { sequence: number; kind: string; tool?: string; accepted?: boolean; code?: string; session_id?: string }[] };
type Permission = { snapshot_id: string; permissions: { report: boolean } };
type Config = { credential_ready: boolean; model: string };
const states: Record<string, string> = { QUEUED: '任务已接受', RUNNING: '正在运行', CANCELLING: '正在停止并回收进程',
  CANCELLED: '任务已取消', FAILED: '任务未完成', SUCCEEDED: '工程草稿已准备，等待双审核' };
const tools: Record<string, string> = { inspect_report_contract: '读取报告合同', inspect_report_status: '检查报告任务状态', submit_report_proposal: '提交受限章节提案' };
const errors: Record<string, string> = { AGENT_NOT_CONFIGURED: '宿主尚未配置可用的 Flash 凭据。', EXECUTOR_BUSY: '已有任务正在运行，请稍后再试。',
  SDK_BUDGET_EXHAUSTED: '开发预算已用完，本次已停止；需在宿主侧核验预算。', SNAPSHOT_CHANGED: '病例快照已变化，请重新连接病例库。',
  SERVICE_RESTARTED: '服务重启，旧任务已失效；不会自动恢复模型调用。', CANCELLED: '任务已取消，后续结果不能登记。',
  DATA_PERMISSION_REQUIRED: '需要为当前病例资料授权；重启或更新资料后须重新授权。', DATA_PERMISSION_CHANGED: '外发授权已变化，旧任务不能继续。',
  CORE_UNAVAILABLE: '无法核验当前任务，请刷新恢复连接；不要重复创建任务。' };

export function AgentPanel({ caseId, snapshotId, eligible }: { caseId: string; snapshotId: string; eligible: boolean }) {
  const [task, setTask] = useState<Task | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [error, setError] = useState('');
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [consent, setConsent] = useState(false);
  const [permission, setPermission] = useState<Permission | null>(null);
  useEffect(() => { setConsent(false); setPermission(null); }, [caseId, snapshotId]);
  const pending = useRef<string | null>(null);
  const sequence = useRef(0);
  const mounted = useRef(true);
  const sending = useRef(false);
  const refresh = useCallback(async () => {
    const seq = ++sequence.current;
    try {
      const [cfg, latest, grant] = await Promise.all([invoke<Config>('agent_configuration'), invoke<Task | null>('agent_latest', { caseId }), invoke<Permission>('data_permissions', { caseId })]);
      if (!mounted.current || seq !== sequence.current) return;
      setConfig(cfg); setTask(latest); setPermission(grant); setConnected(true); setError(previous => previous === errors.CORE_UNAVAILABLE ? '' : previous);
    } catch {
      if (mounted.current && seq === sequence.current) { setConnected(false); setError(errors.CORE_UNAVAILABLE); }
    }
  }, [caseId]);
  useEffect(() => {
    mounted.current = true; void refresh();
    const timer = setInterval(() => { if (!document.hidden) void refresh(); }, 1500);
    return () => { mounted.current = false; ++sequence.current; clearInterval(timer); };
  }, [refresh]);
  async function start() {
    if (sending.current || task?.active || !connected || !eligible || !config?.credential_ready || !(consent || authorized)) return;
    sending.current = true; setBusy(true); setError(''); ++sequence.current;
    if (!pending.current) pending.current = crypto.randomUUID().replaceAll('-', '');
    try {
      if (!authorized) {
        const grant = await invoke<Permission>('data_permission_set', { caseId, snapshotId, allowed: true });
        if (mounted.current) setPermission(grant);
      }
      const result = await invoke<Task>('agent_start', { caseId, snapshotId, idempotencyKey: pending.current });
      pending.current = null;
      if (mounted.current) { setTask(result); setConnected(true); }
    } catch (err) {
      if (mounted.current) setError(errors[String(err)] ?? `任务被拒绝：${String(err)}`);
      // Keep the same idempotency key after an uncertain acknowledgement; retries cannot rebill.
    } finally { sending.current = false; if (mounted.current) setBusy(false); }
  }
  async function cancel() {
    if (!task || sending.current) return;
    sending.current = true; setBusy(true); ++sequence.current;
    try {
      const result = await invoke<Task>('agent_cancel', { taskId: task.id });
      if (mounted.current) { setTask(result); setError(''); }
    } catch { if (mounted.current) setError('取消尚未确认，请重试取消或刷新任务状态。'); }
    finally { sending.current = false; if (mounted.current) setBusy(false); }
  }
  async function withdraw() {
    if (sending.current) return;
    sending.current = true; setBusy(true); ++sequence.current;
    try {
      const grant = await invoke<Permission>('data_permission_set', { caseId, snapshotId, allowed: false });
      if (mounted.current) { setPermission(grant); setConsent(false); setError(''); }
    } catch { if (mounted.current) setError('撤回尚未确认，请重试或刷新。'); }
    finally { sending.current = false; if (mounted.current) setBusy(false); }
  }
  const authorized = permission?.snapshot_id === snapshotId && permission.permissions.report;
  const stale = task && task.snapshot_id !== snapshotId;
  return <section className="agent-panel" aria-label="Agent 任务">
    <div className="timeline-heading"><h2>Agent 任务</h2><span>{config?.credential_ready ? 'Flash · 宿主已配置' : '等待宿主配置'}</span></div>
    <p className="timeline-caption">准备工程草稿：Core 检查合成输入与占位证据，真实 Agent 读取合同并提交章节提案。会调用付费 Flash；本步不自动审核或发布。</p>
    <p className="timeline-caption">外发到 DeepSeek 官方：报告合同、证据摘要、任务状态、章节提案及模型生成的会话内容；不发送原始血糖、用药历史或文件名。服务重启或更新资料后需重新授权。新报告/审核任务的 SDK 临时目录在进程结束后清理，失败会记录待处理状态；撤回停止后续请求，不删除已经发出的内容。</p>
    {!authorized && <label><input type="checkbox" checked={consent} disabled={busy} onChange={event => setConsent(event.target.checked)} />允许当前病例资料用于上述报告任务</label>}
    {authorized && <p role="status">当前资料已授权报告任务。<button className="evidence-refresh" disabled={busy} onClick={() => void withdraw()}>撤回外发授权</button></p>}
    <div className="agent-actions"><button className="evidence-refresh" disabled={busy || !connected || !eligible || !config?.credential_ready || task?.active || !(consent || authorized)} onClick={() => void start()}>{pending.current ? '确认上次启动请求' : '启动报告 Agent'}</button>
      {task?.active && <button className="evidence-refresh" disabled={busy || task.state === 'CANCELLING'} onClick={() => void cancel()}>取消当前任务</button>}
      <button className="evidence-refresh" onClick={() => void refresh()}>刷新任务</button></div>
    {!eligible && <p className="fixture-notice">当前仅接受观测完整的合成病例，真实预测和 RL 尚未接入。</p>}
    {error && <p role="alert" className="evidence-unavailable">{error}</p>}
    {!task && connected && <p role="status">尚未启动 Agent；打开页面和刷新不会调用模型。</p>}
    {task && <div className="agent-status" role="status"><strong>{!connected ? '任务状态暂不可核验' : stale ? '以下为旧快照的任务' : !task.currently_valid && task.state === 'SUCCEEDED' ? '旧草稿资格已失效' : states[task.state] ?? '任务状态待核验'}</strong>
      <p>{task.active ? task.phase === 'evidence' ? '正在核验占位模型与工程证据' : '正在执行 SDK 报告任务' : '当前不会自动继续执行下一步。'}</p>
      {(task.reason || task.invalid_reason) && <p>{errors[task.invalid_reason ?? task.reason ?? ''] ?? (task.invalid_reason ?? task.reason)}</p>}
      <ol className="agent-progress">{task.progress.map(item => <li key={item.sequence}>{item.kind === 'report_agent_tool' ? `${tools[item.tool ?? ''] ?? '受控工具'} · ${item.accepted ? '已接受' : '已拒绝'}` : item.kind === 'report_agent_completed' ? 'SDK 会话正常完成' : item.kind === 'worker_started' ? '独立执行进程已启动' : '独立执行进程已回收'}{item.code && `（${item.code}）`}</li>)}</ol>
      <small>任务 {task.id} · Core {task.run_state}<br />最后核验 {new Date(task.checked_at * 1000).toLocaleTimeString('zh-CN', { hour12: false })}；未审核正文不会显示在此处。</small>
    </div>}
  </section>;
}
