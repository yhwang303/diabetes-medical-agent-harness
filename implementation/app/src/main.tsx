import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import './style.css';
import { InputWorkspace } from './InputWorkspace';
import { McpSettings } from './McpSettings';

type ShellInfo = {
  version: string;
  platform: string;
  architecture: string;
};

const stages = ['输入核验', '血糖预测', 'RL 候选', '独立审核', '报告发布'];

function App() {
  const [page, setPage] = useState<'workspace' | 'about' | 'settings'>('workspace');
  const [info, setInfo] = useState<ShellInfo | null>(null);
  const [nativeError, setNativeError] = useState(false);
  const [coreConnected, setCoreConnected] = useState(false);

  useEffect(() => {
    invoke<ShellInfo>('shell_info').then(setInfo).catch(() => setNativeError(true));
  }, []);

  return (
    <div className="app-frame">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark" aria-hidden="true">H<span>·</span></span><div>研究工作台<small>Medical Harness</small></div></div>
        <p className="nav-caption">本地工作区</p>
        <nav aria-label="主导航">
          <button className={page === 'workspace' ? 'nav-item selected' : 'nav-item'} aria-current={page === 'workspace' ? 'page' : undefined} onClick={() => setPage('workspace')}><span aria-hidden="true">▦</span> 工作区概览</button>
          <button className={page === 'about' ? 'nav-item selected' : 'nav-item'} aria-current={page === 'about' ? 'page' : undefined} onClick={() => setPage('about')}><span aria-hidden="true">ⓘ</span> 本步说明</button>
          <button className={page === 'settings' ? 'nav-item selected' : 'nav-item'} aria-current={page === 'settings' ? 'page' : undefined} onClick={() => setPage('settings')}><span aria-hidden="true">⚙</span> 设置</button>
        </nav>
        <div className="sidebar-bottom"><span className="small-dot" />研究演示环境<p>模型输出与临床结论分开。<br />当前未提供诊疗功能。</p></div>
      </aside>

      <main>
        <header className="topbar"><span>糖尿病医疗 Agent</span><span className="shell-tag">受控 Agent</span></header>
        <div className="page-content">
          <div className="page-heading"><div><p className="eyebrow">{page === 'workspace' ? '工作区概览' : page === 'settings' ? '设置' : '本步说明'}</p><h1>{page === 'workspace' ? '糖尿病研究工作台' : page === 'settings' ? '工具与模型连接' : '真实 Agent，受控执行'}</h1></div><span className="version">v{info?.version ?? '0.1.0'}</span></div>
          {page === 'workspace' ? <>
            <p className="intro">在这里整理病例、检查证据，并追溯每一次研究分析。</p>
            <InputWorkspace onConnected={setCoreConnected} />
            <section className="flow-section" aria-labelledby="flow-title">
              <div className="section-heading"><h2 id="flow-title">分析流程</h2><span>流程说明 · 实际进度见证据卡</span></div>
              <ol className="flow-track">{stages.map((label, index) => <li key={label}><span className="step-number">{index + 1}</span><span>{label}</span>{index < stages.length - 1 && <span className="flow-arrow" aria-hidden="true">→</span>}</li>)}</ol>
            </section>
            <section className="connection-strip" aria-label="当前连接状态">
              <div><span className="status-label">桌面窗口</span><strong className={info ? 'native-ready' : ''}>{info ? '原生桥接已连接' : nativeError ? '原生桥接不可用' : '正在确认'}</strong></div>
              <div><span className="status-label">Harness Core</span><strong>{coreConnected ? '病例接口已连接' : '病例接口未连接'}</strong></div>
              <div><span className="status-label">预测 / RL</span><strong>真实模型未接入</strong></div>
              <div><span className="status-label">语言模型</span><strong>按需启用 · 见任务页</strong></div>
            </section>
          </> : page === 'settings' ? <McpSettings /> : <section className="about-panel">
            <p className="about-lead">本步将受限报告 Agent 接入原生桌面。</p>
            <dl><div><dt>当前任务</dt><dd>阶段 3 · Agent 任务入口、运行状态与取消</dd></div><div><dt>窗口运行环境</dt><dd>{info ? `${info.platform} / ${info.architecture} · 原生桥接已确认` : '尚未确认原生环境'}</dd></div><div><dt>已有研究核心</dt><dd>执行器在独立进程运行，超时或任务失效即终止；证据仍由 Core 校验和登记。</dd></div><div><dt>模型状态</dt><dd>真实预测和 RL 尚未接入。占位器仅用于合成数据的工程检查。</dd></div><div><dt>接下来</dt><dd>SDK 与 Flash 已接通；本步完成后再验收全通道发布限制。</dd></div></dl>
            <div className="scope-note">当前开放病例导入、原始时间线、证据状态、占位演示及报告 Agent 任务。任务支持状态查询和取消，生成草稿后等待双审核；没有桌面发布入口。</div>
          </section>}
          <footer><span>仅供研究与工程演示，不用于实际诊疗。</span><span>{info ? `${info.platform} · ${info.architecture}` : '本地桌面'}</span></footer>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
