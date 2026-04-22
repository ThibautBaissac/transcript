import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type ModelEntry = {
  id: string;
  display: string;
  ggml_present: boolean;
  coreml_present: boolean;
};

export type DownloadProgress = {
  model: string;
  stage: "ggml" | "coreml";
  downloaded: number;
  total: number | null;
};

export type RecordingLevel = { rms: number };

export type RecordingInfo = {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  samples_len: number;
};

export type Segment = { start: number; end: number; text: string };
export type TranscriptResult = {
  text: string;
  segments: Segment[];
  language: string;
};

export type TranscriptSource =
  | { kind: "recording" }
  | { kind: "file"; value: string };

type TranscriptMeta = {
  id: string;
  created_at: string;
  model: string;
  source: TranscriptSource;
  duration_secs: number | null;
};

export type TranscriptRecord = TranscriptMeta & { result: TranscriptResult };

export type TranscriptSummary = TranscriptMeta & {
  language: string;
  preview: string;
};

export const api = {
  listModels: () => invoke<ModelEntry[]>("list_models"),
  modelStatus: (id: string) => invoke<ModelEntry>("model_status", { id }),
  downloadModel: (id: string) => invoke<ModelEntry>("download_model", { id }),
  startRecording: () => invoke<void>("start_recording"),
  stopRecording: () => invoke<RecordingInfo>("stop_recording"),
  transcribeCurrent: (model: string, lang?: string) =>
    invoke<TranscriptResult>("transcribe_current_recording", { model, lang }),
  transcribeFile: (path: string, model: string, lang?: string) =>
    invoke<TranscriptResult>("transcribe_file", { path, model, lang }),
  saveTranscript: (
    model: string,
    source: TranscriptSource,
    durationSecs: number | null,
    result: TranscriptResult,
  ) =>
    invoke<TranscriptRecord>("save_transcript", {
      model,
      source,
      duration_secs: durationSecs,
      result,
    }),
  listTranscripts: () => invoke<TranscriptSummary[]>("list_transcripts"),
  loadTranscript: (id: string) => invoke<TranscriptRecord>("load_transcript", { id }),
  deleteTranscript: (id: string) => invoke<void>("delete_transcript", { id }),
};

const on = <T,>(name: string, cb: (payload: T) => void): Promise<UnlistenFn> =>
  listen<T>(name, (e) => cb(e.payload));

export const events = {
  onLevel: (cb: (level: RecordingLevel) => void) => on<RecordingLevel>("recording-level", cb),
  onDownloadProgress: (cb: (progress: DownloadProgress) => void) =>
    on<DownloadProgress>("download-progress", cb),
};
