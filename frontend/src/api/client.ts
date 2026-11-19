import type { Analysis, CleanState, Health, Job, JobCreated, JobEvent, Lyrics, Mode } from "./types";

/** A progress stream that says nothing for this long is treated as stuck. */
const STREAM_SILENCE_MS = 15_000;

/** Error with a message that is safe to show to the user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  try {
    const body: unknown = await res.json();
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string") message = detail;
    else if (Array.isArray(detail) && detail.length) {
      message = detail.map((d: { msg?: string }) => d.msg ?? "invalid request").join("; ");
    }
  } catch {
    // not JSON (e.g. the proxy could not reach the backend)
    if (res.status >= 500) message = "The Unmix backend is not reachable. Is it (or the Colab notebook) still running?";
  }
  return new ApiError(message, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("The Unmix backend is not reachable. Is it running on port 8000?", 0);
  }
  if (!res.ok) throw await errorFrom(res);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export interface CreateJobOptions {
  mode: Mode;
  splitVoices?: boolean;
  lyrics?: boolean;
  /** 0..1 upload progress */
  onUploadProgress?: (fraction: number) => void;
}

export const api = {
  health: () => request<Health>("/api/health"),

  /** XHR instead of fetch because fetch still has no upload progress events. */
  createJob(file: Blob, filename: string, opts: CreateJobOptions): Promise<JobCreated> {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file, filename);
      form.append("mode", opts.mode);
      form.append("split_voices", String(opts.splitVoices ?? false));
      form.append("lyrics", String(opts.lyrics ?? true));

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs");
      xhr.responseType = "json";
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) opts.onUploadProgress?.(e.loaded / e.total);
      };
      xhr.onerror = () => reject(new ApiError("The Unmix backend is not reachable. Is it running on port 8000?", 0));
      xhr.onload = () => {
        const body = xhr.response as { job_id?: string; detail?: unknown } | null;
        if (xhr.status >= 200 && xhr.status < 300 && body?.job_id) {
          resolve({ job_id: body.job_id });
        } else {
          const detail = body?.detail;
          const message =
            typeof detail === "string"
              ? detail
              : xhr.status >= 500 || xhr.status === 0
                ? "The Unmix backend is not reachable. Is it running on port 8000?"
                : `Upload failed (${xhr.status})`;
          reject(new ApiError(message, xhr.status));
        }
      };
      xhr.send(form);
    });
  },

  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),

  getAnalysis: (id: string) => request<Analysis>(`/api/jobs/${id}/analysis`),

  getLyrics: (url: string) => request<Lyrics>(url),

  requestClean: (jobId: string, stem: string) => request<CleanState>(`/api/jobs/${jobId}/clean/${stem}`, { method: "POST" }),

  getClean: (jobId: string, stem: string) => request<CleanState>(`/api/jobs/${jobId}/clean/${stem}`),

  deleteJob: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),

  /**
   * Subscribe to job progress. Returns an unsubscribe function.
   * The stream is closed on the terminal event (otherwise EventSource would
   * auto-reconnect forever); on a network drop we fall back to polling once so
   * a finished job is never missed.
   */
  watchJob(id: string, onEvent: (e: JobEvent) => void, onLost: (err: ApiError) => void): () => void {
    // Live updates over SSE. If the stream dies for good (the browser stops
    // retrying, e.g. after a proxy error), fall back to polling until the job
    // is finished, so the progress screen can never freeze.
    let stopped = false;
    let source: EventSource | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    const stop = () => {
      stopped = true;
      source?.close();
      clearTimeout(timer);
    };
    const deliver = (event: JobEvent) => {
      if (stopped) return;
      if (event.status === "done" || event.status === "error") {
        clearTimeout(watchdog);
        stop();
      }
      onEvent(event);
    };
    const poll = () => {
      if (stopped) return;
      api.getJob(id).then(
        (job) => {
          failures = 0;
          deliver({ status: job.status, stage: job.stage, progress: job.progress, message: job.message, error: job.error });
          if (!stopped) timer = setTimeout(poll, 2000);
        },
        (err: unknown) => {
          if (stopped) return;
          // A 404 is final (job deleted); network errors get a few retries with backoff.
          if ((err instanceof ApiError && err.status === 404) || ++failures >= 5) {
            stop();
            onLost(err instanceof ApiError ? err : new ApiError("Lost connection to the backend.", 0));
          } else timer = setTimeout(poll, 1000 * 2 ** failures);
        },
      );
    };

    // Some proxies (e.g. the Cloudflare quick tunnel used for a Colab backend)
    // hold a stream back instead of failing it. The server always sends the
    // current state first, so silence means the stream is stuck: poll instead.
    const switchToPolling = () => {
      clearTimeout(watchdog);
      if (stopped || !source) return;
      source.close();
      source = null;
      poll();
    };
    const armWatchdog = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(switchToPolling, STREAM_SILENCE_MS);
    };

    source = new EventSource(`/api/jobs/${id}/events`);
    armWatchdog();
    source.onmessage = (msg) => {
      armWatchdog();
      try {
        deliver(JSON.parse(msg.data as string) as JobEvent);
      } catch {
        // ignore a malformed message; the next one (or polling) catches up
      }
    };
    source.onerror = () => {
      // While readyState is CONNECTING the browser is retrying by itself.
      if (stopped || source?.readyState !== EventSource.CLOSED) return;
      switchToPolling();
    };
    return () => {
      clearTimeout(watchdog);
      stop();
    };
  },

  /** Server stems get a download disposition; browser-made (blob:) stems are used as they are. */
  downloadUrl: (stemUrl: string) => (stemUrl.startsWith("blob:") ? stemUrl : `${stemUrl}?download=true`),
};
