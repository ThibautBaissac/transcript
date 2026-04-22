# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Build everything (first build compiles whisper.cpp from source via whisper-rs-sys — takes minutes)
cargo build --release

# Run the CLI locally without installing
cargo run -p transcript-cli --release -- jfk.wav --model base.en

# Install the CLI to ~/.cargo/bin (this is what the /transcribe and /speak slash commands invoke)
cargo install --path crates/transcript-cli --locked

# Rust tests — unit tests live inline (`#[cfg(test)]`) in each module across all crates
cargo test --workspace
cargo test -p transcript-core
cargo test -p transcript-core -- resample::tests::resample_48k_to_16k_rough_length   # single test

# Frontend typecheck
pnpm check

# Tauri dev (hot-reload SvelteKit, fast rebuilds for the Rust backend)
pnpm tauri dev

# Tauri release bundle → src-tauri/target/release/bundle/{macos,dmg}/
pnpm tauri build
```

Smoke check after changes to the core or CLI: `transcript jfk.wav --model base.en` should return the JFK quote.

## Architecture

Three front-ends, one Rust core. Everything is Apple Silicon / macOS only, runs fully offline, and caches models in a single shared directory.

```
crates/transcript-core/   # Rust library — audio, resample, models, whisper engine
crates/transcript-cli/    # `transcript` binary (clap + tokio)
src-tauri/                # Tauri 2 backend (exposes core via #[tauri::command])
src/                      # SvelteKit 5 frontend (runes mode)
.claude/commands/         # project-scoped /transcribe and /speak slash commands
```

### Core library (`crates/transcript-core`)

- `models.rs` — Whisper model catalog (`ModelId` enum), HF download, CoreML zip unpacking. `cache_dir()` resolves to `~/Library/Application Support/com.thibautbaissac.transcript/models/` and is the single source of truth shared by the CLI, the GUI, and the slash commands.
- `audio.rs` — two responsibilities:
  - `decode_file()` using symphonia for file input.
  - Live mic capture via cpal. **cpal's `Stream` is `!Send`**, so `start_capture` spawns a dedicated OS thread that owns the stream and parks on a stop channel. `CaptureHandle` is the `Send` handle the callers hold. Don't try to move the `Stream` across tasks.
  - `record_until_silence()` is the auto-stop primitive the CLI's `--record` and `/speak` use.
- `resample.rs` — downmixes to mono and resamples to 16 kHz (Whisper's required input) using rubato's sinc resampler.
- `transcribe.rs` — `Engine` wraps `WhisperContext`. Inference is synchronous and CPU/ANE-bound; callers run it on `tokio::task::spawn_blocking`.

### CoreML sidecar convention

`whisper-rs` picks up the CoreML encoder automatically when a sibling directory `ggml-<model>-encoder.mlmodelc/` sits next to `ggml-<model>.bin`. The core fetches both from HuggingFace (`ggerganov/whisper.cpp`) on first use — the `.bin` plus a zipped `.mlmodelc` that is unpacked with the system `unzip` binary (intentional — avoids a zip crate dependency). If you add a new model, make sure both paths exist upstream; `ModelId::has_coreml_encoder()` is the hook to opt out when they don't.

### Tauri backend (`src-tauri/src/commands.rs`)

- `AppState` holds three locks: active `CaptureHandle`, cached `Engine`, last recording buffer. `ensure_engine()` reloads only when the requested `ModelId` differs from the cached one.
- Mic level events are throttled to ~30 Hz via an `AtomicU64` checked on the audio thread — without this, the Tauri IPC channel gets flooded (cpal callbacks fire at buffer rate, often 100+ Hz).
- Heavy work (`Engine::load`, `transcribe_*`) always goes through `tokio::task::spawn_blocking`.
- IPC events: `recording-level` (throttled RMS), `download-progress` (ggml then coreml stages).

### Frontend (`src/`)

- SvelteKit with `@sveltejs/adapter-static` → SPA mode (Tauri has no Node server). Build output goes to `build/`, which `src-tauri/tauri.conf.json` points at as `frontendDist`.
- All IPC is typed in `src/lib/ipc.ts` — keep the types there in sync with `commands.rs` serde structs when changing commands or events.
- `+page.svelte` uses Svelte 5 runes (`$state`, `$derived`). VU meter decouples IPC frequency from draw rate via a `vuDirty` flag + `requestAnimationFrame` loop.

### CLI conventions

- **stdout carries only the transcript.** All progress, logs, and status go to stderr. This is what makes `transcript foo.mp3 > out.txt` work and is what the `/transcribe` slash command relies on. Don't `println!` progress; use `eprintln!`.
- `--record` uses `afplay` for Tink/Pop start/stop cues and `osascript` for Notification Center banners. Both are best-effort and silently ignore failures (missing permissions, sandboxed contexts).

### Slash commands (`.claude/commands/`)

`/transcribe` and `/speak` shell out to `$HOME/.cargo/bin/transcript`. `/speak` embeds the CLI invocation directly in the prompt via `!…` — the transcript text becomes part of the user's prompt. If you change CLI flags or output semantics, check both command files.

## Notable constraints

- **First `cargo build` is slow** because `whisper-rs-sys` compiles whisper.cpp from source. Don't mistake this for a hang. Subsequent builds are incremental.
- **`use_gpu = true` in `Engine::load`** — on Apple Silicon this enables Metal in whisper.cpp alongside the CoreML encoder. Don't flip it off without a reason.
- **Model cache is shared across all three interfaces.** If you add model-management logic, thread it through `transcript-core::models` rather than duplicating paths.
- **Tauri config allowlists** — the frontend uses `plugin-dialog` and `plugin-fs` for Save dialogs. Adding new filesystem/dialog capabilities requires updating `src-tauri/capabilities/`.
