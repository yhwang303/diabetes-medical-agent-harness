import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { EvidencePanel } from './EvidencePanel';
import { AgentPanel } from './AgentPanel';

type Case = { case_id: string; snapshot_id: string; file_name: string | null; imported_at: number | null };
type Point = { time: string; value: number | null };
type Timeline = {
  case_id: string; snapshot_id: string; snapshot_hash: string;
  snapshot: { source: 'synthetic' | 'historical' | 'simulation'; source_ref: string; history: Point[]; decision_time: string; unit: string };
  import: { file_name: string; file_hash: string; byte_count: number; imported_at: number } | null;
};
const sources = { synthetic: '合成数据（声明）', historical: '历史数据（声明）', simulation: '仿真数据（声明）' };
const messages: Record<string, string> = {
  INVALID_JSON: '文件不是有效 JSON，或包含重复字段。请检查原文件。',
  INVALID_SCHEMA: '文件不符合病例合同。请检查单位、时区、5 分钟间隔、缺失标记和字段。',
  PAYLOAD_TOO_LARGE: '文件超过 64 KiB，请使用单份结构化病例。',
  CORE_UNAVAILABLE: '本地服务暂不可用。请重新连接；仍失败时重新打开应用。',
  AUTH_REQUIRED: '本地服务认证失败，请重新打开应用。',
  AUDIT_UNAVAILABLE: '审计存储不可用，导入未获确认。恢复后可重试同一文件。',
  STORAGE_UNAVAILABLE: '存储不可用，导入未获确认。恢复后可重试同一文件。',
};
function errorText(error: unknown) { return messages[String(error)] ?? '本次操作未完成，请检查文件或重新连接。'; }

function Plot({ points }: { points: Point[] }) {
  const values = points.flatMap(p => p.value === null ? [] : [p.value]);
  const low = Math.max(0, Math.min(...values) - 10), high = Math.min(1000, Math.max(...values) + 10);
  const x = (i: number) => 46 + i * 642 / (points.length - 1);
  const y = (v: number) => 138 - (v - low) * 112 / Math.max(1, high - low);
  const segments: string[][] = [[]];
  points.forEach((p, i) => { if (p.value === null) segments.push([]); else segments[segments.length - 1].push(`${x(i)},${y(p.value)}`); });
  return <svg className="glucose-plot" viewBox="0 0 720 174" role="img" aria-label="原始血糖时间线，空缺处不连线">
    {values.length ? [low, (low + high) / 2, high].map(v => <g key={v}><line x1="46" x2="688" y1={y(v)} y2={y(v)} stroke="#e3e6ed" /><text x="36" y={y(v) + 4} textAnchor="end">{v.toFixed(0)}</text></g>) : <text x="360" y="85" textAnchor="middle">全部观测缺失，未生成曲线</text>}
    {segments.filter(s => s.length > 1).map((s, i) => <polyline key={i} points={s.join(' ')} fill="none" stroke="#5667a1" strokeWidth="2" />)}
    {points.map((p, i) => p.value === null ? <g key={i}><line x1={x(i)} x2={x(i)} y1="26" y2="138" stroke="#c8ad83" strokeDasharray="3 4" /><title>{p.time}：缺失</title></g> : <circle key={i} cx={x(i)} cy={y(p.value)} r="2.8" fill="#5667a1"><title>{p.time}：{p.value} mg/dL</title></circle>)}
    <text x="46" y="163">{points[0].time.slice(11, 16)}</text><text x="688" y="163" textAnchor="end">{points.at(-1)!.time.slice(11, 16)}</text>
  </svg>;
}

export function InputWorkspace({ onConnected }: { onConnected: (connected: boolean) => void }) {
  const [view, setView] = useState<'input' | 'evidence' | 'agent'>('input');
  const [cases, setCases] = useState<Case[]>([]);
  const [selected, setSelected] = useState('');
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [loading, setLoading] = useState(true);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const sequence = useRef(0);

  async function select(id: string) {
    const current = ++sequence.current;
    setSelected(id); setTimeline(null); setError(''); setConfirmDelete(false);
    try {
      const result = await invoke<Timeline>('input_timeline', { caseId: id });
      if (current === sequence.current) setTimeline(result);
    } catch (err) { if (current === sequence.current) setError(errorText(err)); }
  }
  async function refresh() {
    setLoading(true); setError('');
    try {
      const result = await invoke<Case[]>('list_cases');
      setCases(result); onConnected(true);
      if (result.length) await select(result[0].case_id);
      else { ++sequence.current; setSelected(''); setTimeline(null); setConfirmDelete(false); }
    } catch (err) { onConnected(false); setError(errorText(err)); }
    finally { setLoading(false); }
  }
  useEffect(() => { void refresh(); return () => { sequence.current++; }; }, []);

  async function importFile(file?: File) {
    if (!file || importing) return;
    setError(''); setNotice('');
    if (!file.name.toLowerCase().endsWith('.json')) { setError('本步仅支持 UTF-8 JSON 病例文件。'); return; }
    if (file.size > 65536) { setError(messages.PAYLOAD_TOO_LARGE); return; }
    setImporting(true);
    try {
      let content: string;
      try { content = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer()); }
      catch { setError('文件不是有效 UTF-8 文本，请检查编码。'); return; }
      const result = await invoke<{ case_id: string; duplicate: boolean }>('import_case', { fileName: file.name, content });
      // Import acknowledgement is separate from a later list/read failure.
      setNotice(result.duplicate ? '该文件已导入，已打开现有病例。' : '导入已保存；仅展示原始记录，未运行模型。');
      setCases(await invoke<Case[]>('list_cases')); onConnected(true);
      await select(result.case_id);
    } catch (err) { setError(errorText(err)); }
    finally { setImporting(false); }
  }

  async function deleteCurrent() {
    if (!timeline || !confirmDelete || deleting) return;
    const target = { caseId: timeline.case_id, snapshotId: timeline.snapshot_id };
    setDeleting(true); setError(''); ++sequence.current;
    try {
      const result = await invoke<{ state: string }>('delete_case', target);
      setTimeline(null); setSelected(''); setConfirmDelete(false);
      setNotice(result.state === 'LOCAL_CONTENT_REMOVED'
        ? '本地病例正文已删除，旧任务已失效；保留删除凭证和审计。已有备份、导出和供应商副本不在此次删除范围内。'
        : '本地病例正文已删除，旧任务已失效；SDK 临时副本尚待清理，请等待任务结束或重启服务后核验。');
      await refresh();
    } catch { setError('删除尚未确认。请重试同一病例的删除或重新连接核验；不要据此认为资料已经全部清除。'); }
    finally { setDeleting(false); }
  }

  return <section className="workspace-panel" aria-label="病例导入与时间线">
    <div className="panel-label"><span>病例工作区 · 原始输入</span><div className="import-actions"><button onClick={() => void refresh()} disabled={loading || importing || deleting}>重新连接</button><label className={`import-button ${loading || importing ? 'disabled' : ''}`}>{importing ? '正在校验…' : '导入 JSON 病例'}<input aria-label="导入 JSON 病例" type="file" accept=".json,application/json" disabled={loading || importing || deleting} onChange={e => { void importFile(e.target.files?.[0]); e.target.value = ''; }} /></label></div></div>
    <p className="import-help">UTF-8 JSON · 最大 64 KiB · 5 分钟血糖记录 · 文件通过校验后保存在本地</p>
    {error && <p role="alert" className="import-error">{error}</p>}{notice && <p role="status" className="import-notice">{notice}</p>}
    {loading ? <div className="input-empty">正在连接本地病例库…</div> : !cases.length ? <div className="input-empty"><h2>导入第一份病例</h2><p>选择符合病例合同的 JSON 文件。<br />可以先使用项目 examples 目录中的 synthetic-case.json 合成示例。</p></div> : <div className="input-layout">
      <aside className="case-list" aria-label="已保存病例"><p>最近 100 份病例 · {cases.length} 份</p>{cases.map(c => <button key={c.case_id} className={selected === c.case_id ? 'active' : ''} onClick={() => void select(c.case_id)} disabled={importing || deleting} aria-pressed={selected === c.case_id}><strong>{c.file_name ?? '结构化病例'}</strong><small>{c.case_id.slice(0, 12)}</small></button>)}</aside>
      {timeline ? <div className="timeline-panel"><div className="case-tabs" aria-label="病例视图"><button aria-pressed={view === 'input'} onClick={() => setView('input')}>原始记录</button><button aria-pressed={view === 'evidence'} onClick={() => setView('evidence')}>证据与闸门</button><button aria-pressed={view === 'agent'} onClick={() => setView('agent')}>Agent 任务</button></div>{view === 'agent' ? <AgentPanel key={timeline.snapshot_id} caseId={timeline.case_id} snapshotId={timeline.snapshot_id} eligible={timeline.snapshot.source === 'synthetic' && timeline.snapshot.history.every(p => p.value !== null)} /> : view === 'evidence' ? <EvidencePanel key={timeline.snapshot_id} caseId={timeline.case_id} snapshotId={timeline.snapshot_id} onConnected={onConnected} /> : <><div className="timeline-heading"><h2>原始血糖时间线</h2><span>{sources[timeline.snapshot.source]}</span></div><p className="timeline-caption">{timeline.snapshot.history.length} 个采样点 · {timeline.snapshot.history.filter(p => p.value === null).length} 个缺失点 · mg/dL</p>
        <Plot points={timeline.snapshot.history} />
        <p className="timeline-caption">时间保留文件原始时区；缺失处断开，不插值。此图不包含预测或用药建议。</p>
        <div className="observation-table"><table><thead><tr><th>原始时间（含时区）</th><th>血糖 mg/dL</th><th>记录状态</th></tr></thead><tbody>{timeline.snapshot.history.map(p => <tr key={p.time}><td>{p.time}</td><td>{p.value ?? '—'}</td><td>{p.value === null ? '缺失' : '已记录'}</td></tr>)}</tbody></table></div>
        <details className="source-details"><summary>查看输入来源与保存标识</summary><dl><dt>来源声明</dt><dd>{timeline.snapshot.source_ref}</dd><dt>输入截止</dt><dd>{timeline.snapshot.decision_time}</dd><dt>病例 ID</dt><dd>{timeline.case_id}</dd><dt>快照 ID</dt><dd>{timeline.snapshot_id}</dd><dt>文件 SHA-256</dt><dd>{timeline.import?.file_hash ?? '未通过文件导入，无文件摘要'}</dd><dt>快照 SHA-256</dt><dd>{timeline.snapshot_hash}</dd></dl></details></>}
        <details className="source-details"><summary>本地资料保留与删除</summary>
          <p>病例资料保留至主动删除。删除会移除该病例的全部输入版本、模型产物和报告正文，撤销授权并停止旧任务；保留必要的删除凭证和审计记录。</p>
          <p>已有备份、导出、历史测试目录及供应商副本需另行处理，此操作不保证物理介质和内存彻底擦除。</p>
          <label><input type="checkbox" checked={confirmDelete} disabled={deleting} onChange={e => setConfirmDelete(e.target.checked)} />确认删除当前病例的本地资料</label>
          <button disabled={!confirmDelete || deleting || importing} onClick={() => void deleteCurrent()}>{deleting ? '正在删除…' : '删除本地病例资料'}</button>
        </details>
      </div> : <div className="input-empty">{error ? '时间线未能加载。' : '正在读取所选病例…'}</div>}
    </div>}
  </section>;
}
