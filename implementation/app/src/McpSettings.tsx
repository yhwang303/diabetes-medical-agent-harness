import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type McpStatus = {
  sdk_version: string | null;
  expected_sdk_version: string;
  tools: string[];
  prediction_entry_enabled: boolean;
  real_prediction_configured: boolean;
  rl_mcp_configured: boolean;
  rl_entry_enabled: boolean;
  online_provider_checked: boolean;
  last_check: { state: 'NOT_CHECKED' | 'RUNNING' | 'PASSED' | 'FAILED'; checked_at: number | null; duration_ms?: number; error: string | null };
};

export function McpSettings() {
  const [status, setStatus] = useState<McpStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(false);
  const sequence = useRef(0);
  const mounted = useRef(true);
  async function request(check = false) {
    const id = ++sequence.current;
    setError(false);
    if (check) {
      setChecking(true);
      setStatus(previous => previous ? {...previous, last_check: {state: 'RUNNING', checked_at: null, error: null}} : null);
    }
    else setLoading(true);
    try {
      const result = await invoke<McpStatus>(check ? 'mcp_check' : 'mcp_status');
      if (mounted.current && id === sequence.current) setStatus(result);
    } catch {
      if (mounted.current && id === sequence.current) { setError(true); setStatus(null); }
    } finally {
      if (mounted.current && id === sequence.current) { setLoading(false); setChecking(false); }
    }
  }
  useEffect(() => {
    mounted.current = true;
    void request();
    return () => { mounted.current = false; sequence.current++; };
  }, []);
  useEffect(() => {
    if (status?.last_check.state !== 'RUNNING') return;
    const timer = window.setTimeout(() => { void request(); }, 1000);
    return () => window.clearTimeout(timer);
  }, [status]);
  const result = status?.last_check;
  const running = checking || result?.state === 'RUNNING';
  const passed = result?.state === 'PASSED';
  const stateLabel = running ? '正在检查' : passed ? '本地自检通过' : result?.state === 'FAILED' ? '本地自检失败' : '尚未检查';
  return <section className="mcp-settings" aria-labelledby="mcp-title">
    <p className="intro">查看预测工具的连接情况。真实模型接入进度与本地链路检查分别显示。</p>
    <div className="mcp-panel">
      <div className="mcp-panel-heading"><div><p className="eyebrow">连接检查</p><h2 id="mcp-title">受控预测 MCP</h2></div><span className="mcp-scope">本地 · 合成数据</span></div>
      {error && <p className="mcp-error" role="alert">无法读取本地服务，当前连接状态未知。请刷新状态；刷新不会重跑自检。</p>}
      <div className="mcp-stages" aria-live="polite" aria-busy={running}>
        <article><span className="mcp-step">01 / 工具</span><h3>工具定义</h3><strong>{status ? `${status.tools.length} 个工具已定义` : loading ? '正在读取' : '等待确认'}</strong><p>定义存在不代表调用已通过。</p></article>
        <article className={passed ? 'mcp-passed' : ''}><span className="mcp-step">02 / 链路</span><h3>调用检查</h3><strong>{error ? '状态未知' : stateLabel}</strong><p>实际 SDK → MCP → Core → 占位预测</p></article>
        <article><span className="mcp-step">03 / 模型</span><h3>真实预测模型</h3><strong>{status ? status.real_prediction_configured ? '已配置，待专项验证' : '未接入' : '等待确认'}</strong><p>当前自检只验证工程链路。</p></article>
      </div>
      <div className="mcp-check-row"><div><h3>手动本地自检</h3><p>使用系统自带的合成数据和离线上游回复；不读取当前病例，不调用付费模型。通常数秒，执行限时 20 秒。</p></div><button className="mcp-check-button" disabled={running || loading || !status} onClick={() => { void request(true); }}>{running ? '检查中…' : '运行本地自检'}</button></div>
      <div className="mcp-result" role="status">
        {result?.checked_at ? <span>上次检查：{new Date(result.checked_at * 1000).toLocaleString('zh-CN')} · {((result.duration_ms ?? 0) / 1000).toFixed(1)} 秒</span> : <span>{error ? '检查结果暂不可用。' : running ? '正在执行隔离的合成检查…' : loading ? '正在读取检查记录…' : '本次服务启动后尚无检查结果。'}</span>}
        {result?.state === 'FAILED' && <span className="mcp-error">检查失败（{result.error}），请修复本地环境后重试。</span>}
        {passed && <span>请求与查询返回同一份 fixture 证据；线上语言模型连接未检测。</span>}
      </div>
      <dl className="mcp-details">
        <div><dt>Agent SDK</dt><dd>{status ? `Claude Agent SDK ${status.sdk_version ?? '未安装'}（要求 ${status.expected_sdk_version}）` : '等待本地服务'}</dd></div>
        <div><dt>预测工具</dt><dd>{status?.tools.map(name => <code key={name}>{name}</code>) ?? '等待确认'}</dd></div>
        <div><dt>预测任务入口</dt><dd>{status ? status.prediction_entry_enabled ? '已启用 · 仍需病例授权与 Core 检查' : '未启用 · 自检入口独立可用' : '等待确认'}</dd></div>
        <div><dt>RL 工具</dt><dd>{status ? status.rl_mcp_configured ? `工具已定义 · 任务入口${status.rl_entry_enabled ? '已启用' : '未启用'} · 本页自检仅检查预测` : '尚未接入 MCP' : '等待确认'}</dd></div>
      </dl>
      <div className="mcp-bottom"><span>检查结果仅属于本次本地服务；重启后需重新检查。</span><button disabled={running || loading} onClick={() => { void request(); }}>刷新状态</button></div>
    </div>
  </section>;
}
