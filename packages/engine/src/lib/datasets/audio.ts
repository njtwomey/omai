/**
 * Synthetic audio with known content: spoken digits as a stream of utterances, each a sequence of vowel-like sounds
 * (harmonics shaped by formant resonances) with a speaker's pitch, speed and vocal-tract scale, in noise, with the
 * digit and the speaker of every stretch of samples as the truth. For spotting keywords and recognising digits.
 */

import { type DatasetGenerator, type TimeSeriesData, timeSeries } from '../data/index.ts'
import { foldIn, normal, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { stamp } from './_util.ts'

/** The three (F1, F2) formant pairs, in Hz, that make up each digit (a toy phonology: distinct trajectories, not English). */
export const DIGIT_FORMANTS: readonly (readonly (readonly [number, number])[])[] = [
  [
    [300, 2300],
    [700, 1200],
    [350, 800],
  ],
  [
    [650, 1100],
    [300, 2200],
    [500, 1500],
  ],
  [
    [400, 900],
    [750, 1300],
    [300, 2300],
  ],
  [
    [500, 1800],
    [350, 700],
    [700, 1100],
  ],
  [
    [300, 800],
    [600, 1900],
    [450, 1000],
  ],
  [
    [750, 1200],
    [500, 2000],
    [320, 900],
  ],
  [
    [350, 2200],
    [350, 2200],
    [650, 1000],
  ],
  [
    [600, 1000],
    [400, 1700],
    [300, 2400],
  ],
  [
    [450, 1400],
    [700, 1000],
    [450, 1900],
  ],
  [
    [320, 1000],
    [550, 2100],
    [680, 1250],
  ],
]

/** One utterance in the stream: where it lies and who said which digit. */
export interface Utterance {
  readonly start: number
  readonly end: number
  readonly digit: number
  readonly speaker: number
  /** The speaker's pitch in Hz. */
  readonly pitch: number
  readonly speed: number
}

/**
 * A stream of spoken digits, one channel of audio at `samplingRate`: each utterance is a random digit by a random
 * speaker, made of three vowel-like segments whose formants step from one to the next with a short glide, over a harmonic source at
 * the speaker's pitch, faster or slower by the speaker's speed, with white noise throughout and quiet gaps between
 * utterances. The `segments` label is the digit during an utterance and -1 elsewhere. The truth lists every utterance
 * with its start, end, digit, speaker, pitch and speed, and the formant table.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { spokenDigits } from 'mdml-engine/datasets'
 * (generate(spokenDigits, { knobs: { utterances: 3 } }).truth.extra?.['utterances'] as unknown[]).length // => 3
 * ```
 */
export const spokenDigits: DatasetGenerator<TimeSeriesData> = {
  name: 'spokenDigits',
  kind: 'timeseries',
  description: 'A stream of synthetic spoken digits with speakers of different pitch and speed, in noise.',
  knobs: dict({
    utterances: int(1, 10_000, {
      label: 'utterances',
      description: 'The number of spoken digits in the stream.',
      default: 30,
    }),
    speakers: int(1, 100, { label: 'speakers', description: 'The number of distinct speakers.', default: 4 }),
    samplingRate: int(4000, 48_000, { label: 'sampling rate', description: 'Samples per second.', default: 8000 }),
    noise: real(0, 2, {
      label: 'noise standard deviation',
      description: 'The white noise, relative to unit-amplitude speech.',
      default: 0.05,
    }),
    gap: real(0, 2, { label: 'gap', description: 'Seconds of quiet between utterances.', default: 0.15 }),
  }),
  generate: (k, knobs) => {
    const count = knobs['utterances'] as number
    const speakers = knobs['speakers'] as number
    const rate = knobs['samplingRate'] as number
    const gapSamples = Math.round((knobs['gap'] as number) * rate)
    const people = Array.from({ length: speakers }, (_, s) => {
      const u = uniform(foldIn(foldIn(k, 0), s), [3]).data
      return { pitch: 90 + 130 * (u[0] ?? 0), speed: 0.8 + 0.45 * (u[1] ?? 0), scale: 0.9 + 0.2 * (u[2] ?? 0) }
    })
    const pick = uniform(foldIn(k, 1), [count, 2]).data
    const pieces: { digit: number; speaker: number }[] = Array.from({ length: count }, (_, i) => ({
      digit: Math.min(9, Math.floor((pick[2 * i] ?? 0) * 10)),
      speaker: Math.min(speakers - 1, Math.floor((pick[2 * i + 1] ?? 0) * speakers)),
    }))
    const lengths = pieces.map((p) => Math.round((0.5 / (people[p.speaker]?.speed ?? 1)) * rate))
    const total = lengths.reduce((a, b) => a + b, 0) + gapSamples * (count + 1)
    const y = new Float64Array(total)
    const segments = new Int32Array(total).fill(-1)
    const utterances: Utterance[] = []
    let at = gapSamples
    pieces.forEach((p, i) => {
      const person = people[p.speaker] as { pitch: number; speed: number; scale: number }
      const len = lengths[i] ?? 0
      const formants = DIGIT_FORMANTS[p.digit] as readonly (readonly [number, number])[]
      const harmonics = Math.floor(3400 / person.pitch)
      let phase = 0
      for (let t = 0; t < len; t++) {
        // Three plateaus, one per anchor pair, entered by a short glide from the previous one.
        const third = (3 * t) / Math.max(1, len)
        const q = Math.min(2, Math.floor(third))
        const glide = q > 0 ? Math.min(1, (third - q) / 0.25) : 1
        const [f1q, f2q] = formants[q] as readonly [number, number]
        const [f1p, f2p] = formants[Math.max(0, q - 1)] as readonly [number, number]
        const f1 = person.scale * (f1p + glide * (f1q - f1p))
        const f2 = person.scale * (f2p + glide * (f2q - f2p))
        phase += (2 * Math.PI * person.pitch) / rate
        let s = 0
        for (let h = 1; h <= harmonics; h++) {
          const freq = h * person.pitch
          const gain = Math.exp(-(((freq - f1) / 130) ** 2)) + 0.6 * Math.exp(-(((freq - f2) / 180) ** 2))
          s += gain * Math.sin(h * phase)
        }
        const window = Math.sin((Math.PI * (t + 0.5)) / len) ** 2
        y[at + t] = 0.35 * s * window
        segments[at + t] = p.digit
      }
      utterances.push({
        start: at,
        end: at + len,
        digit: p.digit,
        speaker: p.speaker,
        pitch: person.pitch,
        speed: person.speed,
      })
      at += len + gapSamples
    })
    const z = normal(foldIn(k, 2), [total]).data
    const sd = knobs['noise'] as number
    for (let t = 0; t < total; t++) y[t] = (y[t] ?? 0) + sd * (z[t] ?? 0)
    return {
      data: timeSeries(y, {
        samplingRate: rate,
        segments,
        channelNames: ['audio'],
        meta: stamp('spokenDigits', spokenDigits.description, knobs),
      }),
      truth: { noiseScale: sd, extra: { utterances, formants: DIGIT_FORMANTS, speakers: people } },
    }
  },
}
