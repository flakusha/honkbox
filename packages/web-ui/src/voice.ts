const MODEL_KEY = "sd-rip.voice.model";
const RATE = 16000;

function encodeWav(samples: Float32Array): Blob {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const ascii = (off: number, s: string): void => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, RATE, true);
  v.setUint32(28, RATE * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  ascii(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
    v.setInt16(44 + i * 2, s * 32767, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// Records 16kHz mono float chunks via AudioContext; downsamples on the
// fly when the device runs at a higher rate.
async function record16k(stop: () => Promise<void>): Promise<Float32Array> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext({ sampleRate: RATE });
  const src = ctx.createMediaStreamSource(stream);
  // Deprecated but dependency-free and universally supported; a
  // worklet would need a separate module file for the same capture.
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  const ratio = ctx.sampleRate / RATE;
  proc.onaudioprocess = (ev) => {
    const ch = ev.inputBuffer.getChannelData(0);
    if (ratio === 1) {
      chunks.push(new Float32Array(ch));
    } else {
      const len = Math.floor(ch.length / ratio);
      const down = new Float32Array(len);
      for (let i = 0; i < len; i++) down[i] = ch[Math.floor(i * ratio)] ?? 0;
      chunks.push(down);
    }
  };
  src.connect(proc);
  proc.connect(ctx.destination);
  await stop();
  src.disconnect();
  proc.disconnect();
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  stream.getTracks().forEach((t) => t.stop());
  await ctx.close();
  return out;
}

export function mountVoice(host: HTMLElement): void {
  host.innerHTML = "";
  const slug = document.createElement("input");
  slug.placeholder = "model slug";
  slug.value = localStorage.getItem(MODEL_KEY) ?? "";
  slug.addEventListener("change", () =>
    localStorage.setItem(MODEL_KEY, slug.value),
  );
  const lang = document.createElement("input");
  lang.placeholder = "language (optional)";
  const btn = document.createElement("button");
  btn.textContent = "Record";
  const status = document.createElement("p");
  const player = document.createElement("audio");
  player.controls = true;
  host.append(slug, lang, btn, status, player);

  let stopFn: (() => void) | null = null;
  btn.addEventListener("click", () => {
    if (stopFn) {
      stopFn();
      return;
    }
    localStorage.setItem(MODEL_KEY, slug.value);
    btn.textContent = "Stop";
    status.textContent = "recording…";
    let resolveStop: () => void;
    const stopped = new Promise<void>((r) => {
      resolveStop = r;
    });
    stopFn = () => resolveStop();
    record16k(() => stopped)
      .then(async (samples) => {
        status.textContent = "transcribing…";
        const q = new URLSearchParams({ model: slug.value });
        if (lang.value.trim()) q.set("language", lang.value.trim());
        const tres = await fetch(`/v1/transcribe?${q}`, {
          method: "POST",
          headers: { "content-type": "audio/wav" },
          body: encodeWav(samples),
        });
        if (!tres.ok) throw new Error(`transcribe ${tres.status}`);
        const t = (await tres.json()) as { text: string };
        host.append(Object.assign(document.createElement("p"), { textContent: `you: ${t.text}` }));
        status.textContent = "chatting…";
        const cres = await fetch("/v1/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: slug.value,
            messages: [{ role: "user", content: t.text }],
          }),
        });
        if (!cres.ok) throw new Error(`chat ${cres.status}`);
        const c = (await cres.json()) as { text: string };
        host.append(Object.assign(document.createElement("p"), { textContent: `assistant: ${c.text}` }));
        status.textContent = "synthesizing…";
        const sres = await fetch("/v1/synthesize", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: slug.value, text: c.text }),
        });
        if (!sres.ok) throw new Error(`synthesize ${sres.status}`);
        const wav = new Blob([await sres.arrayBuffer()], { type: "audio/wav" });
        player.src = URL.createObjectURL(wav);
        await player.play();
        status.textContent = "done";
      })
      .catch((err: unknown) => {
        status.textContent = `error: ${err instanceof Error ? err.message : err}`;
      })
      .finally(() => {
        stopFn = null;
        btn.textContent = "Record";
      });
  });
}
