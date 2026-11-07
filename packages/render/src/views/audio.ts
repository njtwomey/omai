/**
 * The views of an `AudioClip`: the waveform as an envelope of minimum and maximum per pixel column above its
 * spectrogram (computed here by a windowed transform when the record has none), with the playhead as a line through
 * both at the time of the viewed step, as a scene. `spectrogramOf` and `waveformEnvelope` are the pure parts.
 */

import { maxOf } from 'mdml-engine/base'
import { type AudioClip, audioDuration, playheadAt, type Spectrogram } from 'mdml-engine/exposition'
import { rfft } from 'mdml-engine/numerics/fourier'
import { image, line, type Pixels, polygon, type Scene, scene, text } from 'mdml-engine/scene'
import { toArray } from 'mdml-engine/tensor'
import { parseColour, sampleRamp, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'

/**
 * The magnitude spectrogram of a signal by a Hann-windowed transform.
 *
 * @param samples - The signal.
 * @param sampleRate - Samples per second.
 * @param options - The window length (a power of two, default 256) and the hop between windows (default half of it).
 * @returns The spectrogram (frequencies 0 to Nyquist, windows centred on their times).
 * @example
 * ```ts
 * import { spectrogramOf } from 'mdml-render/views'
 * const tone = Array.from({ length: 512 }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / 8000))
 * const s = spectrogramOf(tone, 8000, { window: 64, hop: 32 })
 * s.frequencies[s.magnitude.map((row) => row[3] ?? 0).indexOf(Math.max(...s.magnitude.map((row) => row[3] ?? 0)))] // => 1000
 * ```
 */
export function spectrogramOf(
  samples: readonly number[],
  sampleRate: number,
  options: { readonly window?: number; readonly hop?: number } = {},
): Spectrogram {
  const n = options.window ?? 256
  const hop = options.hop ?? Math.max(1, n / 2)
  const hann = Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n))
  const starts: number[] = []
  for (let s = 0; s + n <= samples.length; s += hop) starts.push(s)
  const bins = n / 2 + 1
  const magnitude: number[][] = Array.from({ length: bins }, () => [])
  for (const s of starts) {
    const spec = rfft(samples.slice(s, s + n).map((v, i) => v * (hann[i] ?? 0)))
    const re = toArray(spec.re) as number[]
    const im = toArray(spec.im) as number[]
    for (let f = 0; f < bins; f++) magnitude[f]?.push(Math.hypot(re[f] ?? 0, im[f] ?? 0))
  }
  return {
    times: starts.map((s) => (s + n / 2) / sampleRate),
    frequencies: Array.from({ length: bins }, (_, f) => (f * sampleRate) / n),
    magnitude,
  }
}

/**
 * The waveform as a minimum and maximum per column.
 *
 * @param samples - The signal.
 * @param columns - How many columns.
 * @returns One `[min, max]` per column.
 * @example
 * ```ts
 * import { waveformEnvelope } from 'mdml-render/views'
 * waveformEnvelope([0, 1, -1, 0.5], 2) // => [[0, 1], [-1, 0.5]]
 * ```
 */
export function waveformEnvelope(samples: readonly number[], columns: number): [number, number][] {
  return Array.from({ length: columns }, (_, c) => {
    const a = Math.floor((c * samples.length) / columns)
    const b = Math.max(a + 1, Math.floor(((c + 1) * samples.length) / columns))
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = a; i < b && i < samples.length; i++) {
      const v = samples[i] ?? 0
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    return lo <= hi ? [lo, hi] : [0, 0]
  })
}

/**
 * A spectrogram as pixels in decibels over a dynamic range below its peak, low frequencies at the bottom.
 *
 * @param s - The spectrogram.
 * @param mode - The colour mode, for the ramp.
 * @param range - Decibels below the peak at the bottom of the ramp (default 70).
 * @returns The pixels.
 * @example
 * ```ts
 * import { spectrogramPixels } from 'mdml-render/views'
 * spectrogramPixels({ times: [0, 1], frequencies: [0, 1, 2], magnitude: [[1, 1], [1, 1], [1, 1]] }, 'dark').height // => 3
 * ```
 */
export function spectrogramPixels(s: Spectrogram, mode: ColourMode, range = 70): Pixels {
  const rows = s.frequencies.length
  const cols = s.times.length
  const peak = Math.max(1e-12, maxOf(s.magnitude.flat()))
  const lut = sampleRamp(sequentialRamp('magma', mode), 256).map(parseColour)
  const rgba = new Uint8ClampedArray(rows * cols * 4)
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const db = 20 * Math.log10(Math.max(s.magnitude[rows - 1 - r]?.[c] ?? 0, 1e-12) / peak)
      const k = Math.round(Math.max(0, Math.min(1, 1 + db / range)) * 255)
      const col = lut[k] ?? [0, 0, 0, 1]
      rgba.set([col[0] ?? 0, col[1] ?? 0, col[2] ?? 0, 255], 4 * (r * cols + c))
    }
  return { width: cols, height: rows, rgba }
}

/** Options of {@link audioScene}. */
export interface AudioSceneOptions {
  /** The playhead, in seconds (default none). */
  readonly time?: number
  readonly width?: number
  readonly height?: number
  readonly mode?: ColourMode
  /** A spectrogram already computed (the record's, or one from {@link spectrogramOf}). */
  readonly spectrogram?: Spectrogram
}

/** Where the plots of an audio scene sit: the left margin, the width, and the waveform's and spectrogram's boxes. */
export function audioLayout(
  width = 440,
  height = 320,
): { left: number; plot: number; wave: [number, number]; spec: [number, number] } {
  const left = 40
  const plot = width - left - 8
  const waveH = Math.round((height - 60) * 0.4)
  return { left, plot, wave: [8, 8 + waveH], spec: [8 + waveH + 14, height - 28] }
}

/**
 * The scene of a clip: the waveform envelope above the spectrogram, a time axis, the markers and the playhead through
 * both. The scene's title says what is drawn and where the playhead stands, for assistive technology.
 *
 * @param clip - The clip.
 * @param options - The playhead, size, mode and a spectrogram.
 * @returns The scene.
 * @example
 * ```ts
 * import { audioClip } from 'mdml-engine/exposition'
 * import { audioScene } from 'mdml-render/views'
 * const clip = audioClip({ sampleRate: 100, samples: Array.from({ length: 200 }, (_, i) => Math.sin(i)) })
 * audioScene(clip, { time: 1 }).title // => 'Waveform and spectrogram of a 2 s clip, playhead at 1 s'
 * ```
 */
export function audioScene(clip: AudioClip, options: AudioSceneOptions = {}): Scene {
  const { width = 440, height = 320, mode = 'light' } = options
  const dur = audioDuration(clip)
  const L = audioLayout(width, height)
  const x = (t: number): number => L.left + (dur > 0 ? Math.min(1, Math.max(0, t / dur)) : 0) * L.plot
  const mid = (L.wave[0] + L.wave[1]) / 2
  const half = (L.wave[1] - L.wave[0]) / 2
  const env = waveformEnvelope(clip.samples, Math.max(1, Math.floor(L.plot)))
  const spec = options.spectrogram ?? clip.spectrogram ?? spectrogramOf(clip.samples, clip.sampleRate)
  const nyquist = clip.sampleRate / 2
  const t = options.time
  const children = [
    line(L.left, mid, L.left + L.plot, mid, { stroke: 'grid', strokeWidth: 1 }),
    ...env.map(([lo, hi], c) =>
      line(L.left + c + 0.5, mid - hi * half, L.left + c + 0.5, mid - lo * half - 0.5, {
        stroke: 'tab0',
        strokeWidth: 1,
      }),
    ),
    image(L.left, L.spec[0], L.plot, L.spec[1] - L.spec[0], spectrogramPixels(spec, mode), {
      smoothing: false,
      title: 'spectrogram',
    }),
    text(L.left - 4, mid, '0', { anchor: 'end', baseline: 'middle', size: 10, fill: 'muted' }),
    text(L.left - 4, L.spec[1], '0 Hz', { anchor: 'end', baseline: 'middle', size: 10, fill: 'muted' }),
    text(L.left - 4, L.spec[0] + 4, `${Math.round(nyquist)}`, {
      anchor: 'end',
      baseline: 'middle',
      size: 10,
      fill: 'muted',
    }),
    text(L.left, height - 12, '0 s', { anchor: 'start', size: 10, fill: 'muted' }),
    text(L.left + L.plot, height - 12, `${Number(dur.toFixed(2))} s`, { anchor: 'end', size: 10, fill: 'muted' }),
    ...(clip.markers ?? []).flatMap((m) => [
      line(x(m.time), L.wave[0], x(m.time), L.spec[1], { stroke: 'muted', strokeWidth: 1, dash: [2, 3] }),
      text(x(m.time) + 3, L.wave[0] + 8, m.label, { size: 10, fill: 'muted' }),
    ]),
    ...(t === undefined
      ? []
      : [
          line(x(t), L.wave[0] - 4, x(t), L.spec[1] + 2, { stroke: 'tab3', strokeWidth: 2 }),
          // A triangle head, so the playhead is findable without its colour.
          polygon(
            [
              [x(t) - 5, L.wave[0] - 6],
              [x(t) + 5, L.wave[0] - 6],
              [x(t), L.wave[0] + 2],
            ],
            { fill: 'tab3' },
          ),
        ]),
  ]
  return scene(width, height, children, {
    title: `Waveform and spectrogram of a ${Number(dur.toFixed(2))} s clip${t === undefined ? '' : `, playhead at ${Number(t.toFixed(2))} s`}`,
  })
}

/**
 * The scene of a clip at a tape step: the playhead at the time the step stands at.
 *
 * @param clip - The clip.
 * @param step - The tape step.
 * @param options - Size, mode and a spectrogram.
 * @returns The scene.
 * @example
 * ```ts
 * import { audioClip } from 'mdml-engine/exposition'
 * import { audioSceneAt } from 'mdml-render/views'
 * const clip = audioClip({ sampleRate: 100, samples: new Array(200).fill(0), frames: [{ step: 0, time: 0 }, { step: 4, time: 2 }] })
 * audioSceneAt(clip, 2).title // => 'Waveform and spectrogram of a 2 s clip, playhead at 1 s'
 * ```
 */
export function audioSceneAt(clip: AudioClip, step: number, options: Omit<AudioSceneOptions, 'time'> = {}): Scene {
  return audioScene(clip, { ...options, time: playheadAt(clip, step) })
}
