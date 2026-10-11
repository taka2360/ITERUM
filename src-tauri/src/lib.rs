pub mod deepzoom;

/// Forwards a diagnostic line from the WebView to stdout (tests and benchmarks read it).
#[tauri::command]
fn diag(line: String) {
    println!("[iterum] {line}");
}

/// Launch mode from the ITERUM_MODE environment variable ("", "bench", "gputest").
#[tauri::command]
fn launch_mode() -> String {
    std::env::var("ITERUM_MODE").unwrap_or_default()
}

/// Reads an environment variable (test harness configuration).
#[tauri::command]
fn env_var(name: String) -> Option<String> {
    if name.starts_with("ITERUM_") { std::env::var(name).ok() } else { None }
}

/// Writes test artifacts (captures, benchmark reports). Only used by the test modes.
#[tauri::command]
fn save_file(path: String, data: Vec<u8>) -> Result<(), String> {
    if std::env::var("ITERUM_MODE").unwrap_or_default().is_empty() {
        return Err("save_file is only available in test modes".into());
    }
    std::fs::write(&path, data).map_err(|e| e.to_string())
}

#[tauri::command]
fn exit_app(app: tauri::AppHandle, code: i32) {
    app.exit(code);
}

pub fn run() {
    tauri::Builder::default()
        .manage(deepzoom::DeepZoomState::default())
        .invoke_handler(tauri::generate_handler![
            diag,
            launch_mode,
            env_var,
            save_file,
            exit_app,
            deepzoom::reference_orbit,
            deepzoom::bla_table,
            deepzoom::cancel_orbit,
            deepzoom::direct_iterate,
            deepzoom::direct_batch
        ])
        .run(tauri::generate_context!())
        .expect("error while running ITERUM");
}
