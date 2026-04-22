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
  let vuCanvas: HTMLCanvasElement | undefined = $state();

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
  let rafId: number | undefined;
  let vuDirty = false;
  let pendingListeners: Promise<UnlistenFn>[] = [];

  onMount(() => {
    (async () => { models = await api.listModels(); })();

    // Stash the pending listener promises so onDestroy can await them — avoids
    // leaking a listener if the component unmounts before `listen()` resolves.
    const lvl = events.onLevel((l) => {
      level = Math.min(1, l.rms * 4); // gentle amplification for visibility
      vuDirty = true;
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

    // rAF-driven canvas redraw: decouples drawing from IPC frequency, and naturally
    // caps at display refresh rate regardless of how fast level events arrive.
    const tick = () => {
      if (vuDirty) {
        drawVu();
        vuDirty = false;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  });

  onDestroy(async () => {
    if (timerId) clearInterval(timerId);
    if (rafId) cancelAnimationFrame(rafId);
    for (const p of pendingListeners) {
      try { (await p)(); } catch {}
    }
  });

  function drawVu() {
    const c = vuCanvas;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = c.width;
    const h = c.height;
    ctx.clearRect(0, 0, w, h);
    const barW = w * level;
    const hue = level > 0.8 ? 0 : 120 - level * 50;
    ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
    ctx.fillRect(0, 0, barW, h);
  }

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
      }, 1000);
    } catch (e) {
      status = { kind: "error", message: String(e) };
    }
  }

  async function transcribeDroppedFile(path: string) {
    if (busy) return;
    droppedFile = path;
    try {
      // Auto-fetch the model on drop — a drop is an explicit transcribe intent,
      // so block on download rather than forcing the user to notice a disabled button.
      if (needsDownload) await ensureModel(selectedModel);
    } catch (e) {
      status = { kind: "error", message: String(e) };
      return;
    }
    await runTranscribe(() => api.transcribeFile(path, selectedModel));
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

  async function copyToClipboard() {
    await navigator.clipboard.writeText(transcriptText);
  }

  async function saveTranscript() {
    const path = await save({
      defaultPath: `transcript-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`,
      filters: [{ name: "Text", extensions: ["txt"] }],
    });
    if (!path) return;
    await writeTextFile(path, transcriptText);
  }

  function basename(path: string): string {
    const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return i >= 0 ? path.slice(i + 1) : path;
  }

  function fmtDuration(seconds: number): string {
    const s = Math.floor(seconds % 60)
      .toString()
      .padStart(2, "0");
    const m = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    return `${m}:${s}`;
  }
</script>

<main>
  <header class="top-bar">
    <h1>Transcript</h1>
    <div class="model-picker">
      <label for="model">Model</label>
      <select id="model" bind:value={selectedModel} disabled={busy}>
        {#each models as m}
          <option value={m.id}>
            {m.display} {m.ggml_present ? "✓" : "(not downloaded)"}
          </option>
        {/each}
      </select>
      {#if needsDownload && status.kind !== "downloading"}
        <button class="secondary" onclick={downloadSelected}>Download</button>
      {/if}
    </div>
  </header>

  <section class="record-panel">
    <canvas
      bind:this={vuCanvas}
      class="vu-meter"
      width="600"
      height="16"
      aria-label="Input level"
    ></canvas>

    <div class="controls">
      {#if status.kind === "recording"}
        <button class="big stop" onclick={stopAndTranscribe}>
          <span class="dot"></span> Stop &amp; Transcribe
        </button>
      {:else if status.kind === "transcribing"}
        <button class="big" disabled>Transcribing…</button>
      {:else if status.kind === "downloading"}
        <button class="big" disabled>
          Downloading {status.stage} {status.pct}%
        </button>
      {:else}
        <button
          class="big record"
          onclick={startRecording}
          disabled={needsDownload}
          title={needsDownload ? "Download the model first" : "Start recording"}
        >
          <span class="dot"></span> Record
        </button>
      {/if}
      <span class="elapsed">{fmtDuration(elapsed)}</span>
    </div>

    <p class="hint">
      {#if status.kind === "transcribing" && droppedFile}
        Transcribing <code>{basename(droppedFile)}</code>…
      {:else}
        or drop an audio file anywhere in the window
      {/if}
    </p>

    {#if status.kind === "error"}
      <div class="error">{status.message}</div>
    {/if}
  </section>

  {#if dragging}
    <div class="drop-overlay" aria-hidden="true">
      <div class="drop-card">Drop audio to transcribe</div>
    </div>
  {/if}

  <section class="transcript-panel">
    <div class="panel-header">
      <h2>Transcript</h2>
      <div class="panel-actions">
        <button class="secondary" disabled={!transcript} onclick={copyToClipboard}>Copy</button>
        <button class="secondary" disabled={!transcript} onclick={saveTranscript}>Save .txt</button>
      </div>
    </div>
    <textarea
      class="transcript-text"
      bind:value={transcriptText}
      placeholder={transcript
        ? ""
        : "Press Record to capture audio, then Stop & Transcribe. The text will appear here."}
      spellcheck="false"
    ></textarea>
    {#if transcript}
      <div class="meta">
        Language: {transcript.language} · {transcript.segments.length} segments
      </div>
    {/if}
  </section>
</main>

<style>
  :global(:root) {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", sans-serif;
    font-size: 15px;
    color: #1a1a1a;
    background: #f6f6f6;
  }
  :global(body) {
    margin: 0;
  }
  main {
    display: flex;
    flex-direction: column;
    gap: 18px;
    padding: 18px 24px;
    min-height: 100vh;
    box-sizing: border-box;
  }
  .top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .top-bar h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
  }
  .model-picker {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .model-picker label {
    color: #555;
  }
  select {
    padding: 6px 8px;
    border-radius: 6px;
    border: 1px solid #cfcfcf;
    background: #fff;
    font-size: 14px;
    min-width: 280px;
  }
  .record-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    align-items: center;
    padding: 20px;
    border-radius: 12px;
    background: #fff;
    border: 1px solid #e5e5e5;
  }
  .vu-meter {
    width: 100%;
    max-width: 600px;
    height: 16px;
    background: #eee;
    border-radius: 8px;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .big {
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    background: #2952e3;
    color: #fff;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .big:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .big.record {
    background: #2952e3;
  }
  .big.stop {
    background: #c8322b;
  }
  .big .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #fff;
    display: inline-block;
  }
  .elapsed {
    font-variant-numeric: tabular-nums;
    color: #333;
    min-width: 52px;
  }
  .hint {
    margin: 0;
    color: #777;
    font-size: 13px;
  }
  .hint code {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    background: rgba(0, 0, 0, 0.05);
    padding: 1px 6px;
    border-radius: 4px;
  }
  .error {
    color: #b00020;
    font-size: 13px;
  }
  .drop-overlay {
    position: fixed;
    inset: 0;
    background: rgba(41, 82, 227, 0.08);
    border: 3px dashed #2952e3;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
    z-index: 50;
  }
  .drop-card {
    background: #2952e3;
    color: #fff;
    font-size: 18px;
    font-weight: 600;
    padding: 18px 28px;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  }
  .transcript-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .panel-header h2 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }
  .panel-actions {
    display: flex;
    gap: 8px;
  }
  .secondary {
    padding: 6px 12px;
    border: 1px solid #cfcfcf;
    background: #fff;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
  }
  .secondary:hover:not(:disabled) {
    background: #f0f0f0;
  }
  .secondary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .transcript-text {
    flex: 1;
    min-height: 260px;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #e5e5e5;
    background: #fff;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 14px;
    line-height: 1.55;
    resize: none;
    color: inherit;
  }
  .meta {
    color: #666;
    font-size: 12px;
  }
  @media (prefers-color-scheme: dark) {
    :global(:root) {
      color: #f0f0f0;
      background: #1c1c1e;
    }
    .record-panel,
    .transcript-text,
    select,
    .secondary {
      background: #2b2b2e;
      border-color: #3a3a3d;
      color: #f0f0f0;
    }
    .vu-meter {
      background: #1a1a1c;
    }
    .meta,
    .model-picker label,
    .hint {
      color: #bbb;
    }
    .hint code {
      background: rgba(255, 255, 255, 0.08);
    }
    .secondary:hover:not(:disabled) {
      background: #333336;
    }
  }
</style>
