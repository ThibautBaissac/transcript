---
description: Dictate your next prompt by voice. Records from the mic, transcribes locally, treats the transcript as your message.
argument-hint: [max-seconds] — optional hard cap, default 60
allowed-tools: Bash($HOME/.cargo/bin/transcript:*)
---

The user is dictating their next prompt. Below is the transcript of what they just said — captured from the microphone and auto-stopped after ~2.5s of silence.

!DUR="$ARGUMENTS"; DUR="${DUR:-60}"; "$HOME/.cargo/bin/transcript" --record --max-secs "$DUR" --format txt

Treat the text above as if the user had typed it. Respond to it directly — don't re-ask what they meant, don't transcribe or reformat it, just act on the content. If the transcript is empty or obviously garbled, say so and ask them to try again rather than guessing.
