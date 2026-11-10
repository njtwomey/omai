/**
 * The exposition record of a pair plot of posterior draws from several inference methods (`posteriorPairs`): for a few
 * coordinates of the posterior, each method's draws thinned to a few hundred points per method (the same points in
 * every pair, so one table of columns serves the whole plot), the divergent transitions kept apart, and the statistics
 * a corner plot draws (a kernel density per coordinate on a shared grid, the mean and standard deviation of each
 * coordinate, the correlation of each pair) at a modest number of positions of a cursor along the run, so a view can
 * play the draws arriving and show the marginals and correlations of the prefix without computing anything. Each draw
 * carries its step (an iteration, a wall-clock time, a fraction of the run: one unit across the methods; by default the
 * draw's position in its chain), a method with fewer steps stops earlier and keeps its last state. The record is plain
 * numbers sized by `maxPoints`, `frames`, `gridPoints` and the number of coordinates, because it crosses a Worker
 * boundary. The kernel density (Scott's rule, as `gaussianKde` of `omai-engine/stats`) and the correlation are
 * computed here, because `exposition` may not import `stats`; the tests compare them with it.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'

/** One method's draws of the coordinates, as a sampler or a variational method with draws reports them. */
export interface PairDraws {
  /** The method's name (`nuts`, `mean-field vi`, `laplace`). */
  readonly name: string
  /** The draws: each coordinate's, all of one length. */
  readonly draws: Readonly<Record<string, ArrayLike<number>>>
  /** The step of each draw, in one unit across the methods (default the draw's position in its chain). */
  readonly steps?: ArrayLike<number>
  /** The chain of each draw (default one chain): draws of the chains are interleaved by step. */
  readonly chain?: ArrayLike<number>
  /** Whether each draw was a divergent transition (non-zero or true); these are kept apart, to mark on the plot. */
  readonly divergent?: ArrayLike<number | boolean>
}

/** One coordinate of the plot: its shared range and grid, and the reference value when there is one. */
export interface PairsCoordinate {
  readonly name: string
  /** The axis range every panel of this coordinate shares: the bulk of every method's draws and the reference value. */
  readonly range: readonly [number, number]
  /** The grid, ascending over the range, that every method's density is evaluated on. */
  readonly grid: readonly number[]
  /** The reference value (the truth, a posterior mode, a long run's mean). */
  readonly truth?: number
}

/** The draws of a method kept for the scatter, ascending in step; `values[c][i]` is coordinate `c` of point `i`. */
export interface PairsPoints {
  readonly steps: readonly number[]
  readonly values: readonly (readonly number[])[]
}

/**
 * A method's statistics at each cursor position of the record: those of every draw up to that step. Row `f` of each
 * array is for `cursors[f]`; `count[f]` is how many draws there were (under two, nothing else is meaningful, and is zero).
 */
export interface PairsFrames {
  readonly count: readonly number[]
  /** `mean[f][c]`, `sd[f][c]`: the mean and standard deviation of coordinate `c`. */
  readonly mean: readonly (readonly number[])[]
  readonly sd: readonly (readonly number[])[]
  /** `correlation[f][pairIndex(k, i, j)]`: the correlation of coordinates `i < j`, upper triangle by rows. */
  readonly correlation: readonly (readonly number[])[]
  /** `density[f][c * gridPoints + g]`: the kernel density of coordinate `c` at its grid point `g`, integrating to one. */
  readonly density: readonly (readonly number[])[]
}

/** One method: how many draws it had, where its steps run, the points to scatter and its statistics along the run. */
export interface PairsMethod {
  readonly name: string
  readonly numDraws: number
  readonly stepRange: readonly [number, number]
  readonly points: PairsPoints
  /** How many draws diverged (the points of `divergent` are at most the record's limit of them). */
  readonly numDivergent: number
  readonly divergent: PairsPoints
  readonly frames: PairsFrames
}

/** The draws of several methods over a few coordinates, thinned, with the statistics of a pair plot along the run. */
export interface PosteriorPairs {
  readonly kind: 'posteriorPairs'
  readonly coordinates: readonly PairsCoordinate[]
  readonly methods: readonly PairsMethod[]
  /** The steps the statistics are computed at, ascending, the last the end of the longest method. */
  readonly cursors: readonly number[]
  /** The first and last step of any method. */
  readonly stepRange: readonly [number, number]
  /** What the reference value is called (default `truth`). */
  readonly referenceLabel: string
}

/** Options of {@link posteriorPairs}. */
export interface PosteriorPairsOptions {
  /** The coordinates, in the order drawn (default the first method's, the first {@link MAX_PAIR_COORDINATES}). */
  readonly coordinates?: readonly string[]
  /** The most points kept per method (default 400), thinned evenly over the draws ordered by step. */
  readonly maxPoints?: number
  /** The most divergent draws kept per method (default 100). */
  readonly maxDivergent?: number
  /** How many positions of the cursor carry statistics (default 20), evenly over the steps, the last at the end. */
  readonly frames?: number
  /** The grid points of each density (default 60). */
  readonly gridPoints?: number
  /** The draws each density is estimated from (default 500, taken evenly from the prefix: the cost is quadratic in it). */
  readonly densityDraws?: number
  /** The reference value of each coordinate it is known for. */
  readonly truth?: Readonly<Record<string, number>>
  /** What the reference value is called (default `truth`). */
  readonly referenceLabel?: string
}

/** The most coordinates of a plot: beyond it the panels are too small to read. */
export const MAX_PAIR_COORDINATES = 8

/**
 * The index of the pair of coordinates `i < j` among `k` in the record's `correlation` arrays: the upper triangle by rows.
 *
 * @param k - The number of coordinates.
 * @param i - The first coordinate.
 * @param j - The second, greater than `i`.
 * @returns The index, from 0 to `k (k - 1) / 2 - 1`.
 * @example
 * ```ts
 * import { pairIndex } from 'omai-engine/exposition'
 * [pairIndex(4, 0, 1), pairIndex(4, 0, 3), pairIndex(4, 1, 2), pairIndex(4, 2, 3)] // => [0, 2, 3, 5]
 * ```
 */
export function pairIndex(k: number, i: number, j: number): number {
  return i * k - (i * (i + 1)) / 2 + (j - i - 1)
}

/**
 * How many of the points (ascending in step) are at or before a step: a binary search.
 *
 * @param steps - The steps of the points, ascending.
 * @param cursor - The step.
 * @returns The count.
 * @example
 * ```ts
 * import { drawsUpTo } from 'omai-engine/exposition'
 * [drawsUpTo([0, 1, 1, 4], 1), drawsUpTo([0, 1, 1, 4], -1), drawsUpTo([0, 1, 1, 4], 9)] // => [3, 0, 4]
 * ```
 */
export function drawsUpTo(steps: readonly number[], cursor: number): number {
  let lo = 0
  let hi = steps.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((steps[mid] as number) <= cursor) lo = mid + 1
    else hi = mid
  }
  return lo
}

/**
 * The frame of a record whose statistics a cursor shows: the last cursor position at or before it.
 *
 * @param record - The record.
 * @param cursor - A step.
 * @returns The frame index, or -1 before the first.
 * @example
 * ```ts
 * import { pairsFrameAt, posteriorPairs } from 'omai-engine/exposition'
 * const r = posteriorPairs([{ name: 'a', draws: { x: [0, 1, 2, 3], y: [1, 0, 3, 2] } }], { frames: 2 })
 * [r.cursors, pairsFrameAt(r, 0), pairsFrameAt(r, 2), pairsFrameAt(r, 99)] // => [[1.5, 3], -1, 0, 1]
 * ```
 */
export function pairsFrameAt(record: PosteriorPairs, cursor: number): number {
  return drawsUpTo(record.cursors, cursor) - 1
}

/** `m` indices spread evenly over `0 .. n - 1`, distinct and ascending (all of them when `m >= n`). */
function evenly(n: number, m: number): number[] {
  if (m >= n) return Array.from({ length: n }, (_, i) => i)
  if (m <= 1) return n > 0 ? [0] : []
  return Array.from({ length: m }, (_, i) => Math.floor((i * (n - 1)) / (m - 1)))
}

/** The draws of one method with a step each, ordered by it: the order of the draws, their steps and chain positions. */
function ordered(m: PairDraws, n: number): { order: number[]; steps: number[] } {
  const given = m.steps
  if (given !== undefined && given.length !== n)
    throw new OmaiError('shape', `posteriorPairs: ${m.name} has ${given.length} steps for ${n} draws`)
  let steps: number[]
  if (given !== undefined) steps = Array.from(given)
  else if (m.chain === undefined) steps = Array.from({ length: n }, (_, i) => i)
  else {
    if (m.chain.length !== n)
      throw new OmaiError('shape', `posteriorPairs: ${m.name} has ${m.chain.length} chain indices for ${n} draws`)
    const seen = new Map<number, number>()
    steps = Array.from({ length: n }, (_, i) => {
      const c = m.chain?.[i] ?? 0
      const at = seen.get(c) ?? 0
      seen.set(c, at + 1)
      return at
    })
  }
  for (const [i, s] of steps.entries())
    if (!Number.isFinite(s)) throw new OmaiError('value', `posteriorPairs: ${m.name} has step ${s} at draw ${i}`)
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (steps[a] as number) - (steps[b] as number) || a - b,
  )
  return { order, steps: order.map((i) => steps[i] as number) }
}

/** The p-quantile of ascending values by linear interpolation. */
function quantileOf(sorted: readonly number[], p: number): number {
  const at = p * (sorted.length - 1)
  const lo = Math.floor(at)
  const hi = Math.min(sorted.length - 1, lo + 1)
  return (sorted[lo] as number) + (at - lo) * ((sorted[hi] as number) - (sorted[lo] as number))
}

/**
 * A Gaussian kernel density on a grid, bandwidth by Scott's rule (`gaussianKde` of `omai-engine/stats`), normalised to
 * integrate to one over the grid; a sample with no spread gets a bandwidth of a sixtieth of the range.
 */
function densityOn(sample: readonly number[], sd: number, grid: readonly number[]): number[] {
  const n = sample.length
  const span = (grid[grid.length - 1] as number) - (grid[0] as number)
  const h = sd > 0 ? sd * n ** -0.2 : span / 60
  const norm = 1 / (n * h * Math.sqrt(2 * Math.PI))
  const out = grid.map((g) => {
    let s = 0
    for (const v of sample) {
      const z = (g - v) / h
      s += Math.exp(-0.5 * z * z)
    }
    return s * norm
  })
  return out
}

/**
 * The record of a pair plot of several methods' posterior draws: thinned points, kernel densities, means, standard
 * deviations and correlations of the draws up to each of `frames` positions of a cursor along the run.
 *
 * @param methods - The methods and their draws, each coordinate's draws of one length.
 * @param options - The coordinates, the sizes of the record, the reference values and their label.
 * @returns The record.
 * @throws OmaiError `shape` for no method, fewer than two coordinates, more than {@link MAX_PAIR_COORDINATES}, or draws
 *   of different lengths; `value` for a method that lacks a coordinate or has fewer than two draws.
 * @index function Methods' posterior draws for a corner plot, with densities and correlations along the run.
 * @example
 * ```ts
 * import { posteriorPairs } from 'omai-engine/exposition'
 * const draws = (shift: number) => ({ a: Array.from({ length: 200 }, (_, i) => Math.sin(i) + shift), b: Array.from({ length: 200 }, (_, i) => Math.sin(i) * 2 + Math.cos(3 * i)) })
 * const r = posteriorPairs([{ name: 'nuts', draws: draws(0) }, { name: 'vi', draws: draws(0.5) }], { maxPoints: 50, frames: 4, gridPoints: 30, truth: { a: 0 } })
 * [r.methods.map((m) => m.points.steps.length), r.cursors.length, r.coordinates[0]?.grid.length, r.coordinates[0]?.truth] // => [[50, 50], 4, 30, 0]
 * ```
 */
export function posteriorPairs(methods: readonly PairDraws[], options: PosteriorPairsOptions = {}): PosteriorPairs {
  const first = methods[0]
  if (first === undefined) throw new OmaiError('shape', 'posteriorPairs: no methods')
  const names = options.coordinates ?? Object.keys(first.draws).slice(0, MAX_PAIR_COORDINATES)
  const k = names.length
  if (k < 2) throw new OmaiError('shape', 'posteriorPairs: a pair plot needs at least two coordinates')
  if (k > MAX_PAIR_COORDINATES)
    throw new OmaiError('shape', `posteriorPairs: ${k} coordinates, at most ${MAX_PAIR_COORDINATES} fit a plot`)
  const maxPoints = Math.max(2, Math.floor(options.maxPoints ?? 400))
  const maxDivergent = Math.max(0, Math.floor(options.maxDivergent ?? 100))
  const frameCount = Math.max(1, Math.floor(options.frames ?? 20))
  const G = Math.max(8, Math.floor(options.gridPoints ?? 60))
  const kde = Math.max(10, Math.floor(options.densityDraws ?? 500))

  // Each method's draws, ordered by step, as columns.
  const prepared = methods.map((m) => {
    const cols = names.map((c) => {
      const d = m.draws[c]
      if (d === undefined) throw new OmaiError('value', `posteriorPairs: ${m.name} has no draws of ${c}`)
      return d
    })
    const n = cols[0]?.length ?? 0
    for (const [j, d] of cols.entries())
      if (d.length !== n)
        throw new OmaiError(
          'shape',
          `posteriorPairs: ${m.name} has ${d.length} draws of ${names[j]}, ${n} of ${names[0]}`,
        )
    if (n < 2) throw new OmaiError('value', `posteriorPairs: ${m.name} has fewer than two draws`)
    if (m.divergent !== undefined && m.divergent.length !== n)
      throw new OmaiError('shape', `posteriorPairs: ${m.name} has ${m.divergent.length} divergent flags for ${n} draws`)
    const { order, steps } = ordered(m, n)
    const col = cols.map((d) => Float64Array.from(order, (i) => Number(d[i])))
    for (const [j, c] of col.entries())
      for (const v of c)
        if (!Number.isFinite(v))
          throw new OmaiError('value', `posteriorPairs: ${m.name} has a non-finite draw of ${names[j]}`)
    const flags = order.map((i) => (m.divergent?.[i] === undefined ? false : Number(m.divergent[i]) !== 0))
    return { name: m.name, n, steps, col, flags }
  })

  // Cursor positions: evenly over the steps of every method, the last at the end of the longest.
  const lo = Math.min(...prepared.map((p) => p.steps[0] as number))
  const hi = Math.max(...prepared.map((p) => p.steps[p.n - 1] as number))
  const cursors =
    hi > lo
      ? Array.from({ length: frameCount }, (_, f) =>
          f === frameCount - 1 ? hi : lo + ((hi - lo) * (f + 1)) / frameCount,
        )
      : [hi]

  // The axis range of each coordinate: the bulk of every method's draws (0.2% to 99.8%) padded, with the reference value.
  const coordinates = names.map((name, j): PairsCoordinate => {
    const truth = options.truth?.[name]
    let a = truth ?? Number.POSITIVE_INFINITY
    let b = truth ?? Number.NEGATIVE_INFINITY
    for (const p of prepared) {
      const sample = evenly(p.n, kde * 4)
        .map((i) => (p.col[j] as Float64Array)[i] as number)
        .sort((x, y) => x - y)
      a = Math.min(a, quantileOf(sample, 0.002))
      b = Math.max(b, quantileOf(sample, 0.998))
    }
    if (!(b > a)) {
      a -= 0.5
      b += 0.5
    }
    const pad = 0.06 * (b - a)
    a -= pad
    b += pad
    return {
      name,
      range: [a, b],
      grid: Array.from({ length: G }, (_, g) => a + ((b - a) * g) / (G - 1)),
      ...(truth === undefined ? {} : { truth }),
    }
  })

  const out = prepared.map((p): PairsMethod => {
    const keep = evenly(p.n, maxPoints)
    const pick = (idx: readonly number[]): PairsPoints => ({
      steps: idx.map((i) => p.steps[i] as number),
      values: p.col.map((c) => idx.map((i) => c[i] as number)),
    })
    const diverged = p.flags.flatMap((f, i) => (f ? [i] : []))
    const divergentKept = evenly(diverged.length, maxDivergent).map((i) => diverged[i] as number)
    // The running sums of the shifted draws, snapshotted at each cursor: the moments of the prefix without a pass per frame.
    const shift = p.col.map((c) => c[0] as number)
    const sum = new Float64Array(k)
    const cross = new Float64Array(k * k)
    const count: number[] = []
    const mean: number[][] = []
    const sd: number[][] = []
    const correlation: number[][] = []
    const density: number[][] = []
    let at = 0
    const v = new Float64Array(k)
    for (const cursor of cursors) {
      while (at < p.n && (p.steps[at] as number) <= cursor) {
        for (let a = 0; a < k; a++) v[a] = ((p.col[a] as Float64Array)[at] as number) - (shift[a] as number)
        for (let a = 0; a < k; a++) {
          sum[a] = (sum[a] as number) + (v[a] as number)
          for (let b = a; b < k; b++)
            cross[a * k + b] = (cross[a * k + b] as number) + (v[a] as number) * (v[b] as number)
        }
        at++
      }
      count.push(at)
      if (at < 2) {
        mean.push(Array.from({ length: k }, (_, a) => (at === 1 ? ((p.col[a] as Float64Array)[0] as number) : 0)))
        sd.push(new Array<number>(k).fill(0))
        correlation.push(new Array<number>((k * (k - 1)) / 2).fill(0))
        density.push(new Array<number>(k * G).fill(0))
        continue
      }
      const m = Array.from({ length: k }, (_, a) => (sum[a] as number) / at)
      // The covariance with n - 1, from the shifted sums.
      const covar = (a: number, b: number): number =>
        ((cross[a * k + b] as number) - at * (m[a] as number) * (m[b] as number)) / (at - 1)
      const s = m.map((_, a) => Math.sqrt(Math.max(0, covar(a, a))))
      mean.push(m.map((x, a) => x + (shift[a] as number)))
      sd.push(s)
      const r: number[] = []
      for (let a = 0; a < k; a++)
        for (let b = a + 1; b < k; b++) {
          const d = (s[a] as number) * (s[b] as number)
          r.push(d > 0 ? Math.min(1, Math.max(-1, covar(a, b) / d)) : 0)
        }
      correlation.push(r)
      const sub = evenly(at, kde)
      const dens: number[] = []
      for (let a = 0; a < k; a++) {
        const sample = sub.map((i) => (p.col[a] as Float64Array)[i] as number)
        // The spread of the subsample, which the bandwidth is a multiple of.
        const mu = sample.reduce((x, y) => x + y, 0) / sample.length
        const sdSub = Math.sqrt(sample.reduce((x, y) => x + (y - mu) ** 2, 0) / Math.max(1, sample.length - 1))
        dens.push(...densityOn(sample, sdSub, (coordinates[a] as PairsCoordinate).grid))
      }
      density.push(dens)
    }
    return {
      name: p.name,
      numDraws: p.n,
      stepRange: [p.steps[0] as number, p.steps[p.n - 1] as number],
      points: pick(keep),
      numDivergent: diverged.length,
      divergent: pick(divergentKept),
      frames: { count, mean, sd, correlation, density },
    }
  })
  return seal({
    kind: 'posteriorPairs',
    coordinates,
    methods: out,
    cursors,
    stepRange: [lo, hi],
    referenceLabel: options.referenceLabel ?? 'truth',
  })
}
