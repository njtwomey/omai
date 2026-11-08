/**
 * A pulse train: a signal made of a few known short pulses (filters) placed at known, well-separated positions with
 * random signs and sizes, in white noise, which is the generative model of convolutional sparse coding. The truth holds
 * the filters, the sparse activation maps and the clean signal, so a learnt dictionary can be compared with what made
 * the data (`matchFilters` in `optim`: equal up to a shift and a sign).
 */

import { OmaiError } from '../base/index.ts'
import { type DatasetGenerator, type TimeSeriesData, timeSeries } from '../data/index.ts'
import { foldIn, normal, permutation, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { stamp } from './_util.ts'

/** The shapes the pulses take, as functions of the position `u` in `(0, 1)` along the pulse: not yet unit norm. */
const PULSE_SHAPES: readonly ((u: number) => number)[] = [
  // A bump.
  (u) => Math.exp(-0.5 * ((u - 0.5) / 0.15) ** 2),
  // A biphasic wave, the derivative of a bump.
  (u) => (u - 0.5) * Math.exp(-0.5 * ((u - 0.5) / 0.15) ** 2),
  // A fast rise and a slow decay.
  (u) => (1 - Math.exp(-30 * u)) * Math.exp(-4 * u),
  // A plateau with soft edges.
  (u) => 1 / (1 + Math.exp(-40 * (u - 0.2))) - 1 / (1 + Math.exp(-40 * (u - 0.8))),
]

/** The `index`-th pulse shape sampled at `length` points, scaled to unit Euclidean norm. */
function pulse(index: number, length: number): number[] {
  const shape = PULSE_SHAPES[index] as (u: number) => number
  const raw = Array.from({ length }, (_, l) => shape((l + 0.5) / length))
  const norm = Math.hypot(...raw)
  return raw.map((v) => v / norm)
}

/**
 * A pulse train for convolutional sparse coding and dictionary learning: `pulses` different unit-norm pulses of
 * `length` samples (a bump, a biphasic wave, a fast-rise slow-decay spike, a plateau), each placed `occurrences` times (fewer when
 * the signal is too short for that many) at distinct, well-separated positions (no two copies overlap) with a random sign and a size between 0.5 and 2, added
 * to white noise of standard deviation `noise`. The signal is `y = sum_k z_k * d_k + noise`, the truth in `extra`:
 * the filters `d` (`[pulses, length]`), the maps `z` (`[pulses, n - length + 1]`, non-zero where a copy starts), the
 * start positions and sizes of each pulse and the clean signal.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { pulseTrain } from 'omai-engine/datasets'
 * const g = generate(pulseTrain, { knobs: { n: 120, pulses: 2, length: 8, occurrences: 3 } })
 * g.data.values.shape // => [120, 1]
 * (g.truth.extra?.['positions'] as number[][]).map((p) => p.length) // => [3, 3]
 * ```
 */
export const pulseTrain: DatasetGenerator<TimeSeriesData> = {
  name: 'pulseTrain',
  kind: 'timeseries',
  description: 'Known short pulses placed at sparse, separated positions with random signs and sizes, in noise.',
  knobs: dict({
    n: int(40, 1_000_000, { label: 'length', description: 'The number of samples.', default: 240 }),
    pulses: int(1, 4, { label: 'pulses', description: 'The number of different pulse shapes.', default: 2 }),
    length: int(4, 500, { label: 'pulse length', description: 'The number of samples of a pulse.', default: 8 }),
    occurrences: int(1, 1000, {
      label: 'occurrences',
      description: 'How many times each pulse occurs.',
      default: 5,
    }),
    noise: real(0, 5, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the white noise.',
      default: 0.02,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const K = knobs['pulses'] as number
    const L = knobs['length'] as number
    const noise = knobs['noise'] as number
    const M = n - L + 1
    // Cells of 2 L samples, each holding at most one copy that starts in its first half: copies never overlap, so a
    // short signal holds fewer copies than asked for (as many of each pulse as fit).
    const cells = Math.floor(n / (2 * L))
    const occ = Math.min(knobs['occurrences'] as number, Math.floor(cells / K))
    if (occ < 1)
      throw new OmaiError(
        'value',
        `pulseTrain: ${K} pulses of ${L} samples need at least ${2 * K * L} samples to be placed once each, got ${n}`,
      )
    const order = permutation(foldIn(k, 0), cells).data
    const jitter = uniform(foldIn(k, 1), [K * occ]).data
    const size = uniform(foldIn(k, 2), [K * occ], { low: 0.5, high: 2 }).data
    const flip = uniform(foldIn(k, 3), [K * occ]).data
    const white = normal(foldIn(k, 4), [n]).data
    const filters = Array.from({ length: K }, (_, p) => pulse(p, L))
    const codes = Array.from({ length: K }, () => new Array<number>(M).fill(0))
    const copies: { start: number; amplitude: number }[][] = Array.from({ length: K }, () => [])
    const clean = new Float64Array(n)
    for (let j = 0; j < K * occ; j++) {
      const p = j % K
      const start = (order[j] ?? 0) * 2 * L + Math.floor((jitter[j] ?? 0) * L)
      const amplitude = (flip[j] ?? 0) < 0.5 ? -(size[j] ?? 1) : (size[j] ?? 1)
      copies[p]?.push({ start, amplitude })
      const row = codes[p] as number[]
      row[start] = amplitude
      for (let l = 0; l < L; l++) clean[start + l] = (clean[start + l] ?? 0) + amplitude * (filters[p]?.[l] ?? 0)
    }
    for (const list of copies) list.sort((a, b) => a.start - b.start)
    const positions = copies.map((list) => list.map((c) => c.start))
    const amplitudes = copies.map((list) => list.map((c) => c.amplitude))
    const y = Float64Array.from(clean, (v, i) => v + noise * (white[i] ?? 0))
    return {
      data: timeSeries(y, { meta: stamp('pulseTrain', pulseTrain.description, knobs) }),
      truth: {
        noiseScale: noise,
        extra: { filters, codes, positions, amplitudes, clean: Array.from(clean), pulseLength: L },
      },
    }
  },
}
