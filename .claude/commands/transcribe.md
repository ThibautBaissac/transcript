---
description: Transcribe an audio file locally with Whisper (MLX/CoreML, no cloud).
argument-hint: <audio-path> [--model base.en|small.en|medium.en|large-v3|large-v3-turbo] [--lang en|fr|...]
allowed-tools: Bash($HOME/.cargo/bin/transcript:*)
---

Run the local `transcript` CLI on the user's audio file and present the transcript.

Steps:

1. Resolve the audio path from `$ARGUMENTS`. If the first token is a relative path, keep it relative to the current working directory. Expand `~` to `$HOME`. If the path doesn't exist, tell the user and stop — don't guess.

2. Invoke the CLI with Bash:
   ```
   $HOME/.cargo/bin/transcript $ARGUMENTS --format txt
   ```
   Any extra flags in `$ARGUMENTS` (e.g. `--model`, `--lang`, `--prompt`) pass straight through. Model downloads stream progress to stderr on first use of each model; the transcript itself is the stdout.

3. If the command exits non-zero, surface stderr so the user can see the underlying error.

4. When it succeeds, present the transcript back to the user in a code block. Then ask whether they want you to do anything else with it — don't do follow-up work unsolicited.
