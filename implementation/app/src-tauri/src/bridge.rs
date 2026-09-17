use serde_json::{json, Value};
use std::{io::Write, net::TcpListener, path::PathBuf, process::{Child, Command, Stdio}, sync::Mutex};
use std::os::unix::fs::OpenOptionsExt;

pub struct Backend {
    pub port: u16,
    child: Mutex<Option<Child>>,
}

fn root() -> PathBuf {
    // Development app: fixed build-time project runtime, not a WebView-provided path.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("Project runtime missing")
}

impl Backend {
    pub fn start() -> Result<Self, Box<dyn std::error::Error>> {
        let root = root();
        let data = root.join("runtime/desktop-core");
        std::fs::create_dir_all(&data)?;
        if !data.canonicalize()?.starts_with(&root) { return Err("Invalid runtime path".into()); }
        let log_path = data.join("service.log");
        if log_path.is_symlink() { return Err("Invalid log path".into()); }
        let log = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(log_path)?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let child = Command::new(root.join(".venv/bin/python"))
            .args(["-m", "medical_harness.cli", "serve", "--enable-fixtures", "--enable-agent", "--port", &port.to_string(), "--data-dir"])
            .arg(&data).current_dir(&root).stdin(Stdio::null())
            .stdout(log.try_clone()?).stderr(log).spawn()?;
        Ok(Self { port, child: Mutex::new(Some(child)) })
    }

    pub fn stop(&self) {
        if let Some(mut child) = self.child.lock().unwrap().take() {
            // Let Core fence tasks and reap SDK workers before the service exits.
            let _ = Command::new("/bin/kill").args(["-TERM", &child.id().to_string()]).status();
            for _ in 0..100 {
                if matches!(child.try_wait(), Ok(Some(_))) { return; }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            let _ = child.kill(); // Unresponsive service fallback, not normal task cancellation.
            let _ = child.wait();
        }
    }
}

impl Drop for Backend { fn drop(&mut self) { self.stop(); } }

async fn request(port: u16, operation: &'static str, payload: Value) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = root();
        let mut child = Command::new(root.join(".venv/bin/python"))
            .args(["-m", "medical_harness.desktop_client", &port.to_string(), operation])
            .current_dir(root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
            .spawn().map_err(|_| "CORE_UNAVAILABLE")?;
        child.stdin.take().ok_or("CORE_UNAVAILABLE")?.write_all(payload.to_string().as_bytes()).map_err(|_| "CORE_UNAVAILABLE")?;
        let output = child.wait_with_output().map_err(|_| "CORE_UNAVAILABLE")?;
        let result: Value = serde_json::from_slice(&output.stdout).map_err(|_| "CORE_UNAVAILABLE")?;
        if result["ok"] == true { Ok(result["data"].clone()) }
        else { Err(result["error"].as_str().unwrap_or("CORE_UNAVAILABLE").to_string()) }
    }).await.map_err(|_| "CORE_UNAVAILABLE".to_string())?
}

#[tauri::command]
pub async fn list_cases(backend: tauri::State<'_, Backend>) -> Result<Value, String> {
    request(backend.port, "cases", json!({})).await
}

#[tauri::command]
pub async fn input_timeline(backend: tauri::State<'_, Backend>, case_id: String) -> Result<Value, String> {
    if case_id.len() != 32 || !case_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "timeline", json!({"case_id": case_id})).await
}

#[tauri::command]
pub async fn case_evidence(backend: tauri::State<'_, Backend>, case_id: String) -> Result<Value, String> {
    if case_id.len() != 32 || !case_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "evidence", json!({"case_id": case_id})).await
}

#[tauri::command]
pub async fn check_prediction(backend: tauri::State<'_, Backend>, case_id: String, snapshot_id: String, idempotency_key: String) -> Result<Value, String> {
    if [&case_id, &snapshot_id, &idempotency_key].iter().any(|s| s.len() != 32 || !s.bytes().all(|c| c.is_ascii_hexdigit())) {
        return Err("DESKTOP_OPERATION_FORBIDDEN".into());
    }
    request(backend.port, "prediction", json!({"case_id": case_id, "snapshot_id": snapshot_id, "idempotency_key": idempotency_key})).await
}

#[tauri::command]
pub async fn run_rl_demo(backend: tauri::State<'_, Backend>, case_id: String, snapshot_id: String, idempotency_key: String, scenario: String) -> Result<Value, String> {
    if [&case_id, &snapshot_id, &idempotency_key].iter().any(|s| s.len() != 32 || !s.bytes().all(|c| c.is_ascii_hexdigit()))
        || !["candidate", "abstain", "unsupported", "error", "invalid_output", "parent_mismatch", "safety_rejected", "timeout"].contains(&scenario.as_str()) {
        return Err("DESKTOP_OPERATION_FORBIDDEN".into());
    }
    request(backend.port, "rl-demo", json!({"case_id": case_id, "snapshot_id": snapshot_id, "idempotency_key": idempotency_key, "scenario": scenario})).await
}

#[tauri::command]
pub async fn import_case(backend: tauri::State<'_, Backend>, file_name: String, content: String) -> Result<Value, String> {
    if content.len() > 65536 || file_name.len() > 800 { return Err("PAYLOAD_TOO_LARGE".into()); }
    request(backend.port, "import", json!({"file_name": file_name, "content": content})).await
}

#[tauri::command]
pub async fn agent_configuration(backend: tauri::State<'_, Backend>) -> Result<Value, String> {
    request(backend.port, "agent-configuration", json!({})).await
}

#[tauri::command]
pub async fn agent_start(backend: tauri::State<'_, Backend>, case_id: String, snapshot_id: String, idempotency_key: String) -> Result<Value, String> {
    if [&case_id, &snapshot_id, &idempotency_key].iter().any(|s| s.len() != 32 || !s.bytes().all(|c| c.is_ascii_hexdigit())) {
        return Err("DESKTOP_OPERATION_FORBIDDEN".into());
    }
    request(backend.port, "agent-start", json!({"case_id": case_id, "snapshot_id": snapshot_id, "idempotency_key": idempotency_key})).await
}

#[tauri::command]
pub async fn agent_latest(backend: tauri::State<'_, Backend>, case_id: String) -> Result<Value, String> {
    if case_id.len() != 32 || !case_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "agent-latest", json!({"case_id": case_id})).await
}

#[tauri::command]
pub async fn agent_status(backend: tauri::State<'_, Backend>, task_id: String) -> Result<Value, String> {
    if task_id.len() != 32 || !task_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "agent-status", json!({"task_id": task_id})).await
}

#[tauri::command]
pub async fn agent_cancel(backend: tauri::State<'_, Backend>, task_id: String) -> Result<Value, String> {
    if task_id.len() != 32 || !task_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "agent-cancel", json!({"task_id": task_id})).await
}

#[tauri::command]
pub async fn data_permissions(backend: tauri::State<'_, Backend>, case_id: String) -> Result<Value, String> {
    if case_id.len() != 32 || !case_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "data-permissions", json!({"case_id": case_id})).await
}

#[tauri::command]
pub async fn data_permission_set(backend: tauri::State<'_, Backend>, case_id: String, snapshot_id: String, allowed: bool) -> Result<Value, String> {
    if [&case_id, &snapshot_id].iter().any(|s| s.len() != 32 || !s.bytes().all(|c| c.is_ascii_hexdigit())) {
        return Err("DESKTOP_OPERATION_FORBIDDEN".into());
    }
    request(backend.port, "data-permission-set", json!({"case_id": case_id, "snapshot_id": snapshot_id,
        "purpose": "report", "allowed": allowed, "policy_version": "engineering-data-policy-v1"})).await
}

#[tauri::command]
pub async fn case_lifecycle(backend: tauri::State<'_, Backend>, case_id: String) -> Result<Value, String> {
    if case_id.len() != 32 || !case_id.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("NOT_FOUND".into()); }
    request(backend.port, "case-lifecycle", json!({"case_id": case_id})).await
}

#[tauri::command]
pub async fn delete_case(backend: tauri::State<'_, Backend>, case_id: String, snapshot_id: String) -> Result<Value, String> {
    if [&case_id, &snapshot_id].iter().any(|s| s.len() != 32 || !s.bytes().all(|c| c.is_ascii_hexdigit())) {
        return Err("DESKTOP_OPERATION_FORBIDDEN".into());
    }
    request(backend.port, "case-delete", json!({"case_id": case_id, "snapshot_id": snapshot_id,
        "contract_version": "engineering-lifecycle-v1"})).await
}

#[tauri::command]
pub async fn mcp_status(backend: tauri::State<'_, Backend>) -> Result<Value, String> {
    request(backend.port, "mcp-status", json!({})).await
}

#[tauri::command]
pub async fn mcp_check(backend: tauri::State<'_, Backend>) -> Result<Value, String> {
    request(backend.port, "mcp-check", json!({})).await
}
