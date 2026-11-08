/**
 * The Abelian sandpile of Bak, Tang and Wiesenfeld, the model that made self-organised criticality: grains dropped on a
 * grid one at a time, a site holding four or more topples (loses four, gives one to each neighbour), grains fall off
 * the edge, and one grain can set off an avalanche of any size. Stabilisation (whose result does not depend on the
 * order of topplings: that is the Abelian property), the group the stable recurrent configurations form with its
 * identity element, and the sandpile as a traced algorithm that logs each avalanche's size and duration.
 */

import { OmaiError } from '../../base/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, scalar, tensor } from '../../trace/index.ts'
import { codec, draws, positiveInt } from './_core.ts'

/** What stabilising a pile did. */
export interface Stabilised {
  /** The stable heights, row-major. */
  readonly heights: Int32Array
  /** How many times each site toppled (the odometer). */
  readonly topplings: Int32Array
  /** Parallel rounds it took (every unstable site topples in each). */
  readonly rounds: number
  /** Grains that fell off the edge. */
  readonly lost: number
}

/**
 * Topple a pile until every site holds fewer than four grains. Each round, every unstable site gives one
 * grain to each of its four neighbours per four it holds; grains pushed past the edge are lost. The result is the same
 * whatever the order of topplings.
 *
 * @param heights - The grains at each site, row-major.
 * @param width - Columns.
 * @param height - Rows.
 * @returns The stable pile, the toppling counts, the rounds and the grains lost.
 * @throws OmaiError with code `shape` for a pile of the wrong size.
 * @concept sandpile
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * const pile = Int32Array.of(0, 0, 0, 0, 4, 0, 0, 0, 0)
 * const r = automata.stabilise(pile, 3, 3)
 * [Array.from(r.heights), r.rounds, r.lost] // => [[0, 1, 0, 1, 0, 1, 0, 1, 0], 1, 0]
 * ```
 */
export function stabilise(heights: ArrayLike<number>, width: number, height: number): Stabilised {
  if (heights.length !== width * height) throw new OmaiError('shape', 'stabilise: the pile must be width x height')
  const threshold = 4
  const h = Int32Array.from(heights)
  const topplings = new Int32Array(h.length)
  const mark = new Uint8Array(h.length)
  let active: number[] = []
  for (let i = 0; i < h.length; i++) if ((h[i] ?? 0) >= threshold) active.push(i)
  let rounds = 0
  let lost = 0
  while (active.length > 0) {
    rounds++
    const next: number[] = []
    const fire = active.filter((i) => (h[i] ?? 0) >= threshold)
    const give = (j: number): void => {
      h[j] = (h[j] ?? 0) + 1
      if ((h[j] ?? 0) >= threshold && mark[j] === 0) {
        mark[j] = 1
        next.push(j)
      }
    }
    // Every site of this round topples by the amount it held at the start, then the neighbours are credited.
    const amounts = fire.map((i) => Math.floor((h[i] ?? 0) / 4))
    fire.forEach((i, k) => {
      const a = amounts[k] ?? 0
      h[i] = (h[i] ?? 0) - 4 * a
      topplings[i] = (topplings[i] ?? 0) + a
    })
    fire.forEach((i, k) => {
      const a = amounts[k] ?? 0
      const r = Math.floor(i / width)
      const c = i % width
      for (let g = 0; g < a; g++) {
        if (r > 0) give(i - width)
        else lost++
        if (r < height - 1) give(i + width)
        else lost++
        if (c > 0) give(i - 1)
        else lost++
        if (c < width - 1) give(i + 1)
        else lost++
      }
    })
    for (const j of next) mark[j] = 0
    active = next
  }
  return { heights: h, topplings, rounds, lost }
}

/**
 * The sum of two piles in the sandpile group: add site by site, then stabilise.
 *
 * @param a - A stable pile.
 * @param b - A stable pile.
 * @param width - Columns.
 * @param height - Rows.
 * @returns The stable sum.
 * @concept sandpile
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * Array.from(automata.sandpileAdd(Int32Array.of(3, 3), Int32Array.of(1, 0), 2, 1)) // => [1, 0]
 * ```
 */
export function sandpileAdd(a: ArrayLike<number>, b: ArrayLike<number>, width: number, height: number): Int32Array {
  const sum = Int32Array.from(a, (x, i) => x + (b[i] ?? 0))
  return stabilise(sum, width, height).heights
}

/**
 * The identity element of the sandpile group on a grid: the stable pile `e` with `e + c = c` for every recurrent pile
 * `c`. It is found as `stabilise(2 * top - stabilise(2 * top))`, where `top` has three grains everywhere; the result is
 * a symmetric, fractal-looking picture.
 *
 * @param width - Columns.
 * @param height - Rows.
 * @returns The identity pile, row-major.
 * @concept sandpile
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * Array.from(automata.sandpileIdentity(3, 3)) // => [2, 1, 2, 1, 0, 1, 2, 1, 2]
 * ```
 */
export function sandpileIdentity(width: number, height: number): Int32Array {
  const top = new Int32Array(width * height).fill(6)
  const s = stabilise(top, width, height).heights
  return stabilise(
    Int32Array.from(s, (x) => 6 - x),
    width,
    height,
  ).heights
}

/** Where the next grain falls: the middle, a random site, or a given one. */
export type SandpileDrop = 'centre' | 'random' | readonly [number, number]

/** Options of {@link sandpileAlgorithm}. */
export interface SandpileOptions {
  /** Columns and rows (default 51 by 51). */
  readonly width?: number
  readonly height?: number
  /** Where grains fall (default `'centre'`). */
  readonly drop?: SandpileDrop
  /** The starting pile: empty, or `'top'` (three grains everywhere, the largest stable pile) (default `'empty'`). */
  readonly start?: 'empty' | 'top'
}

/** The state of {@link sandpileAlgorithm}. */
export interface SandpileState {
  readonly heights: Int32Array
  /** Grains dropped so far. */
  readonly grains: number
  /** Grains lost off the edge so far. */
  readonly lost: number
  /** Topplings in the last avalanche (0 if the grain did not topple anything). */
  readonly size: number
  /** Parallel rounds of the last avalanche. */
  readonly duration: number
  /** Sites that toppled in the last avalanche, 1 for each. */
  readonly avalanche: Uint8Array
}

/** The paths of {@link sandpileAlgorithm}. */
export type SandpilePath =
  | 'pile/board'
  | 'pile/avalanche'
  | 'pile/grains'
  | 'pile/avalancheSize'
  | 'pile/avalancheDuration'

/** The log contract of {@link sandpileAlgorithm}. */
export const sandpileContract: Contract<SandpileState, SandpilePath> = defineContract({
  'pile/board': tensor({ role: 'state', shape: ['height', 'width'], description: 'grains at each site' }),
  'pile/avalanche': tensor({
    role: 'state',
    shape: ['height', 'width'],
    description: 'sites that toppled in the last avalanche',
  }),
  'pile/grains': scalar({ description: 'grains dropped so far' }),
  'pile/avalancheSize': scalar({ description: 'topplings in the last avalanche' }),
  'pile/avalancheDuration': scalar({ description: 'parallel rounds of the last avalanche' }),
})

/**
 * The sandpile as a traced algorithm: one grain dropped per step, then stabilised; the avalanche's size (topplings),
 * duration (parallel rounds) and the sites it reached are on the tape. Grains are conserved: `grains` equals the sum of
 * the heights plus `lost`. The state serialises.
 *
 * @param options - The size, where grains fall and the starting pile.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a bad size or drop site outside the grid.
 * @concept sandpile
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // Four grains on one site of a 3 by 3 pile make the first avalanche.
 * const s = runAlgorithm(automata.sandpileAlgorithm({ width: 3, height: 3 }), { seed: 0, maxSteps: 4 }).state
 * [s.size, Array.from(s.heights)] // => [1, [0, 1, 0, 1, 0, 1, 0, 1, 0]]
 * ```
 */
export function sandpileAlgorithm(options: SandpileOptions = {}): Algorithm<SandpileState, SandpilePath> {
  const width = options.width ?? 51
  const height = options.height ?? 51
  const drop = options.drop ?? 'centre'
  const start = options.start ?? 'empty'
  positiveInt('sandpileAlgorithm', 'width', width)
  positiveInt('sandpileAlgorithm', 'height', height)
  if (Array.isArray(drop)) {
    const [x, y] = drop as readonly [number, number]
    if (!(Number.isInteger(x) && Number.isInteger(y) && x >= 0 && x < width && y >= 0 && y < height))
      throw new OmaiError('value', 'sandpileAlgorithm: the drop site must be on the grid')
  }
  return {
    name: 'dynamics/automata/sandpile',
    config: { width, height, drop: Array.isArray(drop) ? [...drop] : (drop as string), start },
    schema: contractSchema(sandpileContract),
    init: () => ({
      heights: new Int32Array(width * height).fill(start === 'top' ? 3 : 0),
      grains: 0,
      lost: 0,
      size: 0,
      duration: 0,
      avalanche: new Uint8Array(width * height),
    }),
    step: (s, ctx) => {
      let site: number
      if (drop === 'centre') site = Math.floor(height / 2) * width + Math.floor(width / 2)
      else if (drop === 'random')
        site = Math.min(width * height - 1, Math.floor((draws(ctx.key, 1)[0] ?? 0) * width * height))
      else site = (drop as readonly [number, number])[1] * width + (drop as readonly [number, number])[0]
      const heights = Int32Array.from(s.heights)
      heights[site] = (heights[site] ?? 0) + 1
      const r = stabilise(heights, width, height)
      let size = 0
      for (const t of r.topplings) size += t
      return {
        heights: r.heights,
        grains: s.grains + 1,
        lost: s.lost + r.lost,
        size,
        duration: r.rounds,
        avalanche: Uint8Array.from(r.topplings, (t) => (t > 0 ? 1 : 0)),
      }
    },
    observe: (s) => ({
      metrics: {
        'pile/grains': s.grains,
        'pile/avalancheSize': s.size,
        'pile/avalancheDuration': s.duration,
      },
      tensors: {
        'pile/board': { role: 'state', value: s.heights, shape: [height, width] },
        'pile/avalanche': { role: 'state', value: s.avalanche, shape: [height, width] },
      },
    }),
    ...codec<SandpileState>(),
  }
}
