use serde::Serialize;
use tauri::Manager;
mod bridge;

#[derive(Serialize)]
struct ShellInfo {
    version: &'static str,
    platform: &'static str,
    architecture: &'static str,
}

#[tauri::command]
fn shell_info() -> ShellInfo {
    // This command identifies the shell only. It does not claim backend readiness.
    ShellInfo {
        version: env!("CARGO_PKG_VERSION"),
        platform: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
    }
}

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| { app.manage(bridge::Backend::start()?); Ok(()) })
        .invoke_handler(tauri::generate_handler![shell_info, bridge::list_cases, bridge::input_timeline, bridge::case_evidence, bridge::check_prediction, bridge::run_rl_demo, bridge::import_case, bridge::agent_configuration, bridge::agent_start, bridge::agent_latest, bridge::agent_status, bridge::agent_cancel, bridge::data_permissions, bridge::data_permission_set, bridge::case_lifecycle, bridge::delete_case, bridge::mcp_status, bridge::mcp_check])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                if let Err(error) = window.hide() {
                    eprintln!("Unable to hide window: {error}");
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("Unable to initialize the desktop shell");

    app.run(|handle, event| {
        if let tauri::RunEvent::Exit = event { handle.state::<bridge::Backend>().stop(); }
        #[cfg(target_os = "macos")]
        if let tauri::RunEvent::Reopen { .. } = event {
            if let Some(window) = handle.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
    });
}
