<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { writeTextFile } from "@tauri-apps/plugin-fs";
  import type { UnlistenFn } from "@tauri-apps/api/event";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import {
    api,
    events,
    type ModelEntry,
    type TranscriptResult,
  } from "$lib/ipc";

  type Status =
    | { kind: "idle" }
    | { kind: "recording"; startedAt: number }
    | { kind: "transcribing" }
    | { kind: "downloading"; model: string; pct: number; stage: string }
    | { kind: "error"; message: string };

  let models = $state<ModelEntry[]>([]);
  let selectedModel = $state<string>("large-v3-turbo");
  let status = $state<Status>({ kind: "idle" });
  let elapsed = $state<number>(0);
  let level = $state<number>(0);
  let transcript = $state<TranscriptResult | null>(null);
  let transcriptText = $state<string>("");
  let droppedFile = $state<string | null>(null);
  let dragging = $state<boolean>(false);
  let copied = $state<boolean>(false);

  const selectedEntry = $derived(
    models.find((m) => m.id === selectedModel) ?? null,
  );
  const needsDownload = $derived(
    selectedEntry ? !selectedEntry.ggml_present : false,
  );
  const busy = $derived(
    status.kind === "recording" ||
      status.kind === "transcribing" ||
      status.kind === "downloading",
  );

  let timerId: number | undefined;
  let pendingListeners: Promise<UnlistenFn>[] = [];

  onMount(() => {
    (async () => { models = await api.listModels(); })();

    const lvl = events.onLevel((l) => {
      level = Math.min(1, l.rms * 4);
    });
    const dl = events.onDownloadProgress((p) => {
      if (status.kind !== "downloading") return;
      const pct = p.total ? Math.round((p.downloaded / p.total) * 100) : 0;
      status = { kind: "downloading", model: p.model, pct, stage: p.stage };
    });
    // Tauri-level drag-drop: gives us real filesystem paths (HTML5 DnD in a webview
    // only exposes file names). We listen on the webview, not the DOM.
    const drop = getCurrentWebview().onDragDropEvent((event) => {
      const e = event.payload;
      if (e.type === "enter" || e.type === "over") {
        if (!busy) dragging = true;
      } else if (e.type === "leave") {
        dragging = false;
      } else if (e.type === "drop") {
        dragging = false;
        const first = e.paths?.[0];
        if (!first) return;
        if (busy) {
          status = { kind: "error", message: "Busy — finish the current operation first." };
          return;
        }
        transcribeDroppedFile(first);
      }
    });
    pendingListeners = [lvl, dl, drop];
  });

  onDestroy(async () => {
    if (timerId) clearInterval(timerId);
    for (const p of pendingListeners) {
      try { (await p)(); } catch {}
    }
  });

  async function refreshModels() {
    models = await api.listModels();
  }

  async function ensureModel(id: string) {
    status = { kind: "downloading", model: id, pct: 0, stage: "ggml" };
    await api.downloadModel(id);
    await refreshModels();
  }

  async function runTranscribe(call: () => Promise<TranscriptResult>) {
    status = { kind: "transcribing" };
    try {
      const result = await call();
      transcript = result;
      transcriptText = result.text;
      status = { kind: "idle" };
    } catch (e) {
      status = { kind: "error", message: String(e) };
    }
  }

  async function downloadSelected() {
    try {
      await ensureModel(selectedModel);
      status = { kind: "idle" };
    } catch (e) {
      status = { kind: "error", message: String(e) };
    }
  }

  async function startRecording() {
    try {
      await api.startRecording();
      status = { kind: "recording", startedAt: performance.now() };
      elapsed = 0;
      timerId = window.setInterval(() => {
        if (status.kind === "recording") {
          elapsed = (performance.now() - status.startedAt) / 1000;
        }
      }, 250);
    } catch (e) {
      status = { kind: "error", message: String(e) };
    }
  }

  async function stopAndTranscribe() {
    if (status.kind !== "recording") return;
    if (timerId) {
      clearInterval(timerId);
      timerId = undefined;
    }
    droppedFile = null;
    try {
      await api.stopRecording();
    } catch (e) {
      status = { kind: "error", message: String(e) };
      return;
    }
    await runTranscribe(() => api.transcribeCurrent(selectedModel));
  }

  async function transcribeDroppedFile(path: string) {
    if (busy) return;
    droppedFile = path;
    try {
      if (needsDownload) await ensureModel(selectedModel);
    } catch (e) {
      status = { kind: "error", message: String(e) };
      return;
    }
    await runTranscribe(() => api.transcribeFile(path, selectedModel));
  }

  async function copyToClipboard() {
    await navigator.clipboard.writeText(transcriptText);
    copied = true;
    setTimeout(() => (copied = false), 1400);
  }

  async function saveTranscript() {
    const path = await save({
      defaultPath: `transcript-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`,
      filters: [{ name: "Text", extensions: ["txt"] }],
    });
    if (!path) return;
    await writeTextFile(path, transcriptText);
  }

  function handleMainClick() {
    if (status.kind === "recording") stopAndTranscribe();
    else if (!busy && !needsDownload) startRecording();
  }

  function basename(path: string): string {
    const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return i >= 0 ? path.slice(i + 1) : path;
  }

  function fmtDuration(seconds: number): string {
    const s = Math.floor(seconds % 60).toString().padStart(2, "0");
    const m = Math.floor(seconds / 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  const statusLabel = $derived.by(() => {
    switch (status.kind) {
      case "recording": return fmtDuration(elapsed);
      case "transcribing":
        return droppedFile ? `Transcribing ${basename(droppedFile)}…` : "Transcribing…";
      case "downloading":
        return `Downloading ${status.stage} · ${status.pct}%`;
      case "error": return status.message;
      case "idle":
        if (needsDownload) return "Model not downloaded";
        if (transcript) return `${transcript.language} · ${transcript.segments.length} segments`;
        return "Tap to record · or drop an audio file";
    }
  });
</script>

<div class="titlebar" data-tauri-drag-region></div>

<main class:has-transcript={!!transcript}>
  <header class="top">
    <div class="brand">Transcript</div>
    <div class="model-pill" class:warn={needsDownload}>
      <label for="model" class="sr-only">Model</label>
      <select id="model" bind:value={selectedModel} disabled={busy}>
        {#each models as m}
          <option value={m.id}>{m.display}</option>
        {/each}
      </select>
      {#if needsDownload && status.kind !== "downloading"}
        <button class="pill-action" onclick={downloadSelected}>Download</button>
      {/if}
    </div>
  </header>

  <section class="stage">
    <button
      class="record"
      class:recording={status.kind === "recording"}
      class:disabled={needsDownload || (busy && status.kind !== "recording")}
      onclick={handleMainClick}
      disabled={needsDownload || (busy && status.kind !== "recording")}
      style="--glow: {Math.min(1, level * 1.2)}"
      aria-label={status.kind === "recording" ? "Stop and transcribe" : "Start recording"}
    >
      <span class="glow" aria-hidden="true"></span>
      <span class="icon" aria-hidden="true">
        {#if status.kind === "recording"}
          <svg viewBox="0 0 24 24" width="26" height="26">
            <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>
          </svg>
        {:else if status.kind === "transcribing" || status.kind === "downloading"}
          <svg viewBox="0 0 24 24" width="26" height="26" class="spin">
            <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-dasharray="42 100"/>
          </svg>
        {:else}
          <svg viewBox="0 0 24 24" width="26" height="26">
            <path d="M12 3.5a3.5 3.5 0 0 0-3.5 3.5v5a3.5 3.5 0 0 0 7 0V7A3.5 3.5 0 0 0 12 3.5Zm7 8.5a.75.75 0 0 1 1.5 0 8.5 8.5 0 0 1-7.75 8.47V22a.75.75 0 0 1-1.5 0v-1.53A8.5 8.5 0 0 1 3.5 12a.75.75 0 0 1 1.5 0 7 7 0 1 0 14 0Z" fill="currentColor"/>
          </svg>
        {/if}
      </span>
    </button>

    <p class="status-line" class:error={status.kind === "error"}>{statusLabel}</p>
  </section>

  <section class="reader" class:empty={!transcript}>
    {#if transcript}
      <div class="reader-actions">
        <button class="icon-btn" onclick={copyToClipboard} title="Copy">
          {#if copied}
            <svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 12.5l4 4 10-10" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
          {:else}
            <svg viewBox="0 0 24 24" width="16" height="16"><rect x="8" y="8" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.7" fill="none"/><rect x="4" y="4" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.7" fill="none"/></svg>
          {/if}
        </button>
        <button class="icon-btn" onclick={saveTranscript} title="Save .txt">
          <svg viewBox="0 0 24 24" width="16" height="16"><path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" stroke="currentColor" stroke-width="1.7" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
      </div>
      <textarea class="reader-text" bind:value={transcriptText} spellcheck="false"></textarea>
    {:else}
      <div class="placeholder">
        <p>Press the button to record, or drop an audio file here.</p>
        <p class="faint">Supports wav, mp3, m4a, flac, opus, webm, ogg.</p>
      </div>
    {/if}
  </section>
</main>

{#if dragging}
  <div class="drop-overlay" aria-hidden="true">
    <div class="drop-card">
      <svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <span>Drop audio to transcribe</span>
    </div>
  </div>
{/if}

<style>
  :global(:root) {
    --bg: #f7f7f8;
    --surface: #ffffffc0;
    --surface-solid: #ffffff;
    --border: rgba(0, 0, 0, 0.08);
    --text: #171719;
    --text-soft: #6c6c72;
    --text-faint: #a0a0a6;
    --accent: #4a63e7;
    --accent-soft: rgba(74, 99, 231, 0.12);
    --record: #e54b4b;
    --record-soft: rgba(229, 75, 75, 0.15);
    --danger: #c0392b;
    --radius: 14px;
    --radius-sm: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
    font-size: 14px;
    color: var(--text);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
  }
  :global(body) { margin: 0; }
  :global(*) { box-sizing: border-box; }

  .titlebar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 28px;
    z-index: 10;
    -webkit-app-region: drag;
  }

  main {
    display: grid;
    grid-template-rows: auto auto 1fr;
    gap: 18px;
    padding: 40px 28px 24px;
    min-height: 100vh;
  }

  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
  }
  .model-pill {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  }
  .model-pill.warn { border-color: rgba(229, 165, 0, 0.4); }
  .model-pill select {
    appearance: none;
    -webkit-appearance: none;
    border: none;
    background: transparent;
    padding: 6px 26px 6px 12px;
    font: inherit;
    color: var(--text);
    cursor: pointer;
    border-radius: 999px;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'><path d='M3 5l3 3 3-3' stroke='%236c6c72' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    background-repeat: no-repeat;
    background-position: right 8px center;
  }
  .model-pill select:disabled { color: var(--text-faint); cursor: not-allowed; }
  .pill-action {
    border: none;
    background: var(--accent);
    color: #fff;
    padding: 6px 12px;
    border-radius: 999px;
    font: inherit;
    font-weight: 500;
    cursor: pointer;
  }
  .pill-action:hover { filter: brightness(1.05); }

  .stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
    padding: 6px 0 2px;
  }

  .record {
    position: relative;
    width: 88px;
    height: 88px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--surface-solid);
    color: var(--accent);
    cursor: pointer;
    display: grid;
    place-items: center;
    transition: transform 0.18s ease, box-shadow 0.18s ease, color 0.18s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 8px 24px rgba(0, 0, 0, 0.04);
  }
  .record:hover:not(:disabled) { transform: translateY(-1px); }
  .record:active:not(:disabled) { transform: translateY(0); }
  .record.recording {
    color: #fff;
    background: var(--record);
    border-color: transparent;
    box-shadow:
      0 0 0 calc(4px + 14px * var(--glow, 0)) var(--record-soft),
      0 8px 28px rgba(229, 75, 75, 0.35);
  }
  .record.disabled { opacity: 0.45; cursor: not-allowed; }
  .record .glow {
    position: absolute;
    inset: -10px;
    border-radius: 50%;
    pointer-events: none;
    background: radial-gradient(closest-side, var(--record-soft), transparent 70%);
    opacity: calc(0.25 + 0.6 * var(--glow, 0));
    transition: opacity 0.08s linear;
  }
  .record:not(.recording) .glow { display: none; }
  .record .icon { display: inline-flex; transition: transform 0.25s ease; }
  .record:not(:disabled):active .icon { transform: scale(0.92); }

  .spin { animation: spin 1.1s linear infinite; transform-origin: center; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .status-line {
    margin: 0;
    color: var(--text-soft);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    text-align: center;
    min-height: 18px;
  }
  .status-line.error { color: var(--danger); }

  .reader {
    position: relative;
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    min-height: 220px;
    display: flex;
    flex-direction: column;
  }
  .reader.empty { background: transparent; border-style: dashed; }

  .reader-actions {
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    gap: 4px;
    z-index: 2;
  }
  .icon-btn {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-soft);
    display: grid;
    place-items: center;
    cursor: pointer;
    transition: background 0.12s ease, color 0.12s ease;
  }
  .icon-btn:hover { background: var(--accent-soft); color: var(--accent); }

  .reader-text {
    flex: 1;
    width: 100%;
    padding: 26px 28px;
    border: none;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "New York", Georgia, serif;
    font-size: 15px;
    line-height: 1.65;
    color: var(--text);
    resize: none;
    outline: none;
  }

  .placeholder {
    margin: auto;
    text-align: center;
    padding: 32px;
    color: var(--text-faint);
  }
  .placeholder p { margin: 0 0 4px; font-size: 13px; }
  .placeholder .faint { color: var(--text-faint); font-size: 12px; }

  .drop-overlay {
    position: fixed;
    inset: 16px;
    border: 2px dashed var(--accent);
    border-radius: 18px;
    background: var(--accent-soft);
    display: grid;
    place-items: center;
    pointer-events: none;
    z-index: 50;
    animation: drop-in 0.15s ease-out;
  }
  .drop-card {
    background: var(--accent);
    color: #fff;
    font-size: 14px;
    font-weight: 500;
    padding: 12px 18px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
  }
  @keyframes drop-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .sr-only {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  @media (prefers-color-scheme: dark) {
    :global(:root) {
      --bg: #1a1a1c;
      --surface: rgba(44, 44, 48, 0.7);
      --surface-solid: #2a2a2d;
      --border: rgba(255, 255, 255, 0.08);
      --text: #f1f1f3;
      --text-soft: #a0a0a6;
      --text-faint: #6c6c72;
      --accent: #7b91ff;
      --accent-soft: rgba(123, 145, 255, 0.14);
      --record: #ff5c5c;
      --record-soft: rgba(255, 92, 92, 0.18);
      --danger: #ff6b6b;
    }
    .model-pill select {
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'><path d='M3 5l3 3 3-3' stroke='%23a0a0a6' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    .record {
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2), 0 8px 24px rgba(0, 0, 0, 0.35);
    }
  }
</style>
