mod commands;
mod transcripts;

use commands::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::model_status,
            commands::download_model,
            commands::start_recording,
            commands::stop_recording,
            commands::transcribe_current_recording,
            commands::transcribe_file,
            commands::format_transcript,
            commands::save_transcript,
            commands::list_transcripts,
            commands::load_transcript,
            commands::delete_transcript,
            commands::get_transcript_audio_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
