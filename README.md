# Transcript

Offline speech-to-text on macOS Apple Silicon, powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp) via [`whisper-rs`](https://github.com/tazz4843/whisper-rs) with CoreML + Apple Neural Engine acceleration.

Three interfaces, one shared Rust core:

- **`transcript` CLI** — transcribe an audio file, or record from the mic with silence auto-stop.
- **Transcript.app** — Tauri 2 native GUI: record, stop, transcript pane, model picker.
- **Claude Code slash commands** — `/transcribe <path>` and `/speak` for dictation.

All processing is local. No audio or text ever leaves the machine.

## Prerequisites

```bash
brew install rustup cmake node pnpm
rustup-init -y --default-toolchain stable
```

Also required (usually pre-installed):

- **Xcode Command Line Tools** — `xcode-select --install`
- **Apple Silicon Mac** (M1 or newer) on macOS 12+

Add Cargo to your `PATH` so `cargo` and the built CLI binary are discoverable:

```bash
echo 'export PATH="$HOME/.cargo/bin:/opt/homebrew/opt/rustup/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## First build

```bash
git clone <this-repo> transcript && cd transcript
pnpm install
cargo build --release
```

On first build Cargo fetches `whisper-rs-sys` and compiles whisper.cpp from source (takes a few minutes). Subsequent builds are incremental.

## CLI — `transcript`

Install the binary to `~/.cargo/bin`:

```bash
cargo install --path crates/transcript-cli --locked
```

Transcribe a file:

```bash
transcript '/Users/thba/Downloads/Lundi matin .m4a' --model medium.en
transcript lecture.mp3 --model medium.en --format srt > lecture.srt
transcript interview.wav --lang fr --prompt "proper nouns: Dupont, Martin"
```

Record from the mic (auto-stops after 2.5 s of silence, max 60 s):

```bash
transcript --record
transcript --record --max-secs 120 --model large-v3-turbo
transcript --record --no-cue   # skip the Tink/Pop sounds + notifications
```

Full usage: `transcript --help`.

### Models

| slug | size | multilingual | use case |
|---|---|---|---|
| `base.en` | ~150 MB | no | quick English tests |
| `small.en` | ~500 MB | no | everyday English dictation |
| `medium.en` | ~1.5 GB | no | high-accuracy English |
| `large-v3` | ~3 GB | yes | highest accuracy, any language |
| `large-v3-turbo` (default) | ~1.5 GB | yes | best speed/quality balance |

First use of any model downloads both the ggml weights and a CoreML-compiled encoder from Hugging Face into `~/Library/Application Support/com.thibautbaissac.transcript/models/`. That cache is shared across the CLI, GUI, and slash commands.

### Output formats

- `--format txt` (default) — plain text, one paragraph
- `--format srt` — SubRip subtitles with timestamps
- `--format json` — full structured result (text, segments with start/end, detected language)

stdout carries only the transcript; progress logs and status go to stderr, so `transcript foo.mp3 > out.txt` works as expected.

## GUI — Transcript.app

Run in dev (hot-reload for frontend, fast rebuilds for Rust):

```bash
pnpm tauri dev
```

Build a release bundle:

```bash
pnpm tauri build
```

Output goes to `src-tauri/target/release/bundle/`:

- `macos/Transcript.app` — standalone app (double-click to run)
- `dmg/Transcript_0.1.0_aarch64.dmg` — installer disk image

### First launch

macOS will ask for **Microphone** permission the first time you hit Record. Approve it in System Settings → Privacy & Security → Microphone if you miss the prompt.

The app downloads the selected model on first use (shown as a progress bar) and caches it in the same directory as the CLI.

## Claude Code slash commands

Two user-global slash commands wrap the CLI (installed at `~/.claude/commands/`):

- **`/transcribe <path>`** — transcribe an audio file already on disk. Accepts any CLI flag: `/transcribe foo.mp3 --model medium.en --lang en`.
- **`/speak [max-seconds]`** — dictate your next prompt. Plays a Tink, records from the mic (auto-stops on silence), plays a Pop, and injects the transcript into the Claude Code conversation so Claude responds to what you said.

No configuration needed — as long as `~/.cargo/bin/transcript` exists (after `cargo install`), both commands work.

## Project layout

```
.
├── Cargo.toml                    # workspace root
├── crates/
│   ├── transcript-core/          # shared library (audio, resample, models, whisper engine)
│   └── transcript-cli/           # `transcript` CLI binary
├── src-tauri/                    # Tauri 2 app (Rust backend + packaging)
├── src/                          # SvelteKit frontend
│   ├── routes/+page.svelte       # main UI
│   └── lib/ipc.ts                # typed Tauri command wrappers
└── .claude/commands/             # project-scoped slash commands
```

## Testing

```bash
cargo test -p transcript-core     # unit + integration tests for the core
pnpm check                        # svelte-check + TypeScript
```

Smoke-test the CLI against a known sample:

```bash
curl -LO https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav
transcript jfk.wav --model base.en
# → "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```

## Troubleshooting

**First-run download seems to hang.**  The `large-v3-turbo` pair is ~2.3 GB total. Watch the size grow:

```bash
watch -n 2 'du -sh "$HOME/Library/Application Support/com.thibautbaissac.transcript/models/"'
```

**Recording returns "Thank you. Thank you." or other hallucinations.**  Whisper produces filler text when the audio is near-silent. Usually a microphone permission / input level issue:

- System Settings → Privacy & Security → Microphone — ensure the terminal app (Ghostty/iTerm/Terminal.app) or Transcript.app is enabled.
- System Settings → Sound → Input — check the right mic is selected and its level is > 0 when you speak.

**`cargo build` fails with `cmake not found`.**  `brew install cmake`.

**`pnpm` complains about using yarn.**  Your home directory has a `~/package.json` with a `packageManager` field that's taking precedence. The project already declares `packageManager: pnpm@10.x` in its own `package.json`; run `pnpm` commands from inside the project directory.

## License

MIT.
