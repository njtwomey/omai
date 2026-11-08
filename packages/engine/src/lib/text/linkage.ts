/**
 * Record linkage between two tables without a shared key: blocking to cut the pairs to compare (standard blocking on a
 * key, the sorted-neighbourhood method), field comparison into agreement levels (string similarities cut at
 * thresholds), the Fellegi-Sunter model (Fellegi and Sunter 1969) whose match and non-match probabilities per field and
 * level are fitted by EM on the unlabelled comparison patterns (Winkler 1988; Jaro 1989), match weights and a threshold
 * decision, the clusters that matched pairs form, and the pairwise precision and recall against a known truth.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { connectedComponents, graph } from '../graph/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { jaroWinkler } from './similarity.ts'

/** A record: its fields as text. */
export type LinkageRecord = readonly string[]

/** A candidate pair: the index of a left record and of a right record. */
export type RecordPair = readonly [left: number, right: number]

/**
 * Standard blocking: the pairs of records that share a blocking key (a record may give several keys).
 *
 * @param left - The left table.
 * @param right - The right table.
 * @param key - The key or keys of a record.
 * @returns The candidate pairs, ordered by left then right index, without repeats.
 * @concept record-linkage
 * @example
 * ```ts
 * import { standardBlocking } from 'omai-engine/text'
 * standardBlocking([['ann', 'x'], ['bob', 'y']], [['anna', 'x'], ['bo', 'z']], (r) => r[1] ?? '') // => [[0, 0]]
 * ```
 */
export function standardBlocking(
  left: readonly LinkageRecord[],
  right: readonly LinkageRecord[],
  key: (record: LinkageRecord) => string | readonly string[],
): RecordPair[] {
  const keys = (r: LinkageRecord): string[] => {
    const k = key(r)
    return typeof k === 'string' ? [k] : [...k]
  }
  const blocks = new Map<string, number[]>()
  for (const [j, r] of right.entries()) for (const k of keys(r)) blocks.set(k, [...(blocks.get(k) ?? []), j])
  const pairs = new Set<string>()
  const out: [number, number][] = []
  for (const [i, r] of left.entries())
    for (const k of keys(r))
      for (const j of blocks.get(k) ?? []) {
        const id = `${i},${j}`
        if (!pairs.has(id)) {
          pairs.add(id)
          out.push([i, j])
        }
      }
  return out.sort((a, b) => a[0] - b[0] || a[1] - b[1])
}

/**
 * Sorted-neighbourhood blocking (Hernandez and Stolfo 1995): sort all records of both tables by a key and pair every
 * left and right record that lie within `window` positions of each other in the sorted order.
 *
 * @param left - The left table.
 * @param right - The right table.
 * @param key - The sorting key of a record.
 * @param window - The window size, at least 2: records at most `window - 1` apart are paired.
 * @returns The candidate pairs, ordered by left then right index.
 * @throws OmaiError with code `value` for a window below 2.
 * @concept record-linkage
 * @example
 * ```ts
 * import { sortedNeighbourhoodBlocking } from 'omai-engine/text'
 * sortedNeighbourhoodBlocking([['smith'], ['jones']], [['smyth'], ['zed']], (r) => r[0] ?? '', 2) // => [[0, 0]]
 * ```
 */
export function sortedNeighbourhoodBlocking(
  left: readonly LinkageRecord[],
  right: readonly LinkageRecord[],
  key: (record: LinkageRecord) => string,
  window: number,
): RecordPair[] {
  if (!Number.isInteger(window) || window < 2)
    throw new OmaiError('value', 'sortedNeighbourhoodBlocking: window must be at least 2')
  const all = [
    ...left.map((r, i) => ({ k: key(r), side: 0, i })),
    ...right.map((r, j) => ({ k: key(r), side: 1, i: j })),
  ].sort((a, b) => (a.k < b.k ? -1 : a.k > b.k ? 1 : a.side - b.side || a.i - b.i))
  const pairs = new Set<string>()
  for (let p = 0; p < all.length; p++)
    for (let q = p + 1; q < Math.min(all.length, p + window); q++) {
      const a = all[p] as (typeof all)[number]
      const b = all[q] as (typeof all)[number]
      if (a.side === b.side) continue
      pairs.add(a.side === 0 ? `${a.i},${b.i}` : `${b.i},${a.i}`)
    }
  return [...pairs].map((s) => s.split(',').map(Number) as [number, number]).sort((a, b) => a[0] - b[0] || a[1] - b[1])
}

/** How one field is compared: a similarity in [0, 1] cut into agreement levels. */
export interface FieldComparator {
  /** The field's index in the records. */
  readonly field: number
  /** The similarity of two values (default Jaro-Winkler). */
  readonly compare?: (a: string, b: string) => number
  /** Descending cut points: a similarity at or above the first gives the top level, and below all of them level 0. */
  readonly thresholds: readonly number[]
}

/**
 * The comparison vector of each candidate pair: for every field, the agreement level (0 the lowest) of the pair's
 * values, the similarity cut at the field's thresholds. A field with `k` thresholds has `k + 1` levels.
 *
 * @param pairs - The candidate pairs.
 * @param left - The left table.
 * @param right - The right table.
 * @param fields - How each compared field is cut.
 * @returns One level vector per pair.
 * @concept record-linkage
 * @example
 * ```ts
 * import { comparisonVectors } from 'omai-engine/text'
 * comparisonVectors([[0, 0]], [['smith', '1980']], [['smyth', '1980']], [{ field: 0, thresholds: [0.99, 0.8] }, { field: 1, compare: (a, b) => (a === b ? 1 : 0), thresholds: [1] }]) // => [Int32Array.of(1, 1)]
 * ```
 */
export function comparisonVectors(
  pairs: readonly RecordPair[],
  left: readonly LinkageRecord[],
  right: readonly LinkageRecord[],
  fields: readonly FieldComparator[],
): Int32Array[] {
  return pairs.map(([i, j]) =>
    Int32Array.from(fields, (f) => {
      const s = (f.compare ?? ((a, b) => jaroWinkler(a, b)))(left[i]?.[f.field] ?? '', right[j]?.[f.field] ?? '')
      let level = 0
      for (const [k, t] of f.thresholds.entries()) if (s >= t) level = Math.max(level, f.thresholds.length - k)
      return level
    }),
  )
}

/** The Fellegi-Sunter parameters: the match prior and, per field and level, the probability under match and non-match. */
export interface FellegiSunterModel {
  /** The prior probability that a candidate pair is a match. */
  readonly lambda: number
  /** `m[f][l]` = P(level l of field f | match). */
  readonly m: readonly (readonly number[])[]
  /** `u[f][l]` = P(level l of field f | non-match). */
  readonly u: readonly (readonly number[])[]
}

/** The state of {@link fellegiSunterAlgorithm}. */
export interface FellegiSunterState extends FellegiSunterModel {
  /** The log likelihood of the comparison patterns. */
  readonly logLikelihood: number
  /** The largest change of a parameter in the last step. */
  readonly change: number
  readonly iteration: number
}

const FLOOR = 1e-10

/** The distinct comparison patterns and how many pairs show each. */
function patternsOf(vectors: readonly Int32Array[]): { patterns: Int32Array[]; counts: number[] } {
  const seen = new Map<string, number>()
  const patterns: Int32Array[] = []
  const counts: number[] = []
  for (const v of vectors) {
    const k = Array.from(v).join(',')
    const at = seen.get(k)
    if (at === undefined) {
      seen.set(k, patterns.length)
      patterns.push(v)
      counts.push(1)
    } else counts[at] = (counts[at] ?? 0) + 1
  }
  return { patterns, counts }
}

/**
 * The posterior probability that a pair with the given levels is a match under the model.
 *
 * @param model - The parameters.
 * @param levels - The pair's comparison vector.
 * @returns The posterior match probability.
 * @concept record-linkage
 * @example
 * ```ts
 * import { matchPosterior } from 'omai-engine/text'
 * matchPosterior({ lambda: 0.5, m: [[0.1, 0.9]], u: [[0.9, 0.1]] }, Int32Array.of(1)) // => 0.9
 * ```
 */
export function matchPosterior(model: FellegiSunterModel, levels: ArrayLike<number>): number {
  let lm = Math.log(model.lambda)
  let lu = Math.log(1 - model.lambda)
  for (let f = 0; f < model.m.length; f++) {
    lm += Math.log(model.m[f]?.[levels[f] ?? 0] ?? FLOOR)
    lu += Math.log(model.u[f]?.[levels[f] ?? 0] ?? FLOOR)
  }
  return 1 / (1 + Math.exp(lu - lm))
}

/**
 * The Fellegi-Sunter match weight of a pair: the sum over fields of log2(m / u) at the observed level, the log odds of a
 * match with the prior left out. Large positive weights are matches, large negative non-matches.
 *
 * @param model - The parameters.
 * @param levels - The pair's comparison vector.
 * @returns The weight in bits.
 * @concept record-linkage
 * @example
 * ```ts
 * import { matchWeight } from 'omai-engine/text'
 * matchWeight({ lambda: 0.1, m: [[0.1, 0.9]], u: [[0.9, 0.1]] }, Int32Array.of(1)) // => 3.169925001442312
 * ```
 */
export function matchWeight(model: FellegiSunterModel, levels: ArrayLike<number>): number {
  let w = 0
  for (let f = 0; f < model.m.length; f++)
    w += Math.log2((model.m[f]?.[levels[f] ?? 0] ?? FLOOR) / (model.u[f]?.[levels[f] ?? 0] ?? FLOOR))
  return w
}

/**
 * EM for the Fellegi-Sunter model as an {@link Algorithm}, one iteration per step, on the comparison vectors alone (no
 * labels): the E-step is each pattern's posterior match probability, the M-step re-estimates the prior and the
 * conditional probabilities of each field's levels. It starts with the top level more likely under match and the
 * bottom more likely under non-match, so the components keep their meaning.
 *
 * @param vectors - The comparison vectors ({@link comparisonVectors}).
 * @param levels - The number of levels of each field.
 * @param options - The starting prior (default 0.1) and the tolerance on parameter change (default 1e-8).
 * @returns The algorithm; its state serialises.
 * @throws OmaiError with code `value` for no pairs.
 * @concept record-linkage
 * @example
 * ```ts
 * import { fellegiSunterAlgorithm } from 'omai-engine/text'
 * fellegiSunterAlgorithm([Int32Array.of(1), Int32Array.of(0)], [2]).name // => 'fellegiSunterEm'
 * ```
 */
export function fellegiSunterAlgorithm(
  vectors: readonly Int32Array[],
  levels: readonly number[],
  options: { readonly lambda?: number; readonly tol?: number } = {},
): Algorithm<FellegiSunterState> {
  if (vectors.length === 0) throw new OmaiError('value', 'fellegiSunter: needs at least one pair')
  const { patterns, counts } = patternsOf(vectors)
  const N = vectors.length
  const tol = options.tol ?? 1e-8
  const lambda0 = options.lambda ?? 0.1
  const start = (up: boolean): number[][] =>
    levels.map((L) => {
      const w = Array.from({ length: L }, (_, l) => (up ? l + 1 : L - l) ** 2)
      const z = w.reduce((a, b) => a + b, 0)
      return w.map((x) => x / z)
    })
  const logLik = (s: FellegiSunterModel): number => {
    let ll = 0
    for (const [p, v] of patterns.entries()) {
      let a = Math.log(s.lambda)
      let b = Math.log(1 - s.lambda)
      for (let f = 0; f < levels.length; f++) {
        a += Math.log(s.m[f]?.[v[f] ?? 0] ?? FLOOR)
        b += Math.log(s.u[f]?.[v[f] ?? 0] ?? FLOOR)
      }
      const hi = Math.max(a, b)
      ll += (counts[p] ?? 0) * (hi + Math.log(Math.exp(a - hi) + Math.exp(b - hi)))
    }
    return ll
  }
  return {
    name: 'fellegiSunterEm',
    config: { pairs: N, patterns: patterns.length, levels: [...levels] } as Json,
    init: () => {
      const model = { lambda: lambda0, m: start(true), u: start(false) }
      return { ...model, logLikelihood: logLik(model), change: Number.POSITIVE_INFINITY, iteration: 0 }
    },
    step: (s) => {
      const g = patterns.map((v) => matchPosterior(s, v))
      let sumG = 0
      for (const [p, c] of counts.entries()) sumG += c * (g[p] ?? 0)
      const m = levels.map((L) => new Array<number>(L).fill(0))
      const u = levels.map((L) => new Array<number>(L).fill(0))
      for (const [p, v] of patterns.entries())
        for (let f = 0; f < levels.length; f++) {
          const c = counts[p] ?? 0
          ;(m[f] as number[])[v[f] ?? 0] = ((m[f] as number[])[v[f] ?? 0] ?? 0) + c * (g[p] ?? 0)
          ;(u[f] as number[])[v[f] ?? 0] = ((u[f] as number[])[v[f] ?? 0] ?? 0) + c * (1 - (g[p] ?? 0))
        }
      const norm = (row: number[], total: number): number[] => row.map((x) => Math.max(x / (total || 1), FLOOR))
      const nm = m.map((r) => norm(r, sumG))
      const nu = u.map((r) => norm(r, N - sumG))
      const lambda = Math.min(1 - FLOOR, Math.max(FLOOR, sumG / N))
      let change = Math.abs(lambda - s.lambda)
      for (let f = 0; f < levels.length; f++)
        for (let l = 0; l < (levels[f] ?? 0); l++)
          change = Math.max(
            change,
            Math.abs((nm[f]?.[l] ?? 0) - (s.m[f]?.[l] ?? 0)),
            Math.abs((nu[f]?.[l] ?? 0) - (s.u[f]?.[l] ?? 0)),
          )
      const model = { lambda, m: nm, u: nu }
      return { ...model, logLikelihood: logLik(model), change, iteration: s.iteration + 1 }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { logLikelihood: s.logLikelihood, change: Math.min(s.change, 1e300), lambda: s.lambda },
    }),
    serialise: (s) => ({ ...s, change: Math.min(s.change, 1e300) }) as unknown as Json,
    deserialise: (j) => j as unknown as FellegiSunterState,
  }
}

/**
 * Fit the Fellegi-Sunter model by EM on the comparison vectors.
 *
 * @param vectors - The comparison vectors.
 * @param levels - The number of levels of each field.
 * @param options - The starting prior, the tolerance and the iteration cap (default 500).
 * @returns The fitted parameters, the log likelihood and the iterations used.
 * @concept record-linkage
 * @example
 * ```ts
 * import { fellegiSunter } from 'omai-engine/text'
 * const v = [...Array(90).fill(0), ...Array(10).fill(1)].map((l) => Int32Array.of(l))
 * fellegiSunter(v, [2]).model.lambda > 0 // => true
 * ```
 */
export function fellegiSunter(
  vectors: readonly Int32Array[],
  levels: readonly number[],
  options: { readonly lambda?: number; readonly tol?: number; readonly maxIter?: number } = {},
): { readonly model: FellegiSunterModel; readonly logLikelihood: number; readonly iterations: number } {
  const r = runAlgorithm(fellegiSunterAlgorithm(vectors, levels, options), {
    maxSteps: options.maxIter ?? 500,
    seed: 0,
  })
  const { lambda, m, u } = r.state
  return { model: { lambda, m, u }, logLikelihood: r.state.logLikelihood, iterations: r.steps }
}

/** How well predicted matches agree with the truth, and what the blocking kept. */
export interface LinkageScores {
  readonly precision: number
  readonly recall: number
  readonly f1: number
  readonly truePositives: number
}

/**
 * Pairwise precision, recall and F1 of predicted matching pairs against the true ones.
 *
 * @param predicted - The pairs called matches.
 * @param truth - The true matching pairs.
 * @returns The scores.
 * @concept record-linkage
 * @example
 * ```ts
 * import { linkageScores } from 'omai-engine/text'
 * linkageScores([[0, 0], [1, 2]], [[0, 0], [1, 1]]).f1 // => 0.5
 * ```
 */
export function linkageScores(predicted: readonly RecordPair[], truth: readonly RecordPair[]): LinkageScores {
  const t = new Set(truth.map(([i, j]) => `${i},${j}`))
  const tp = predicted.filter(([i, j]) => t.has(`${i},${j}`)).length
  const precision = predicted.length === 0 ? 0 : tp / predicted.length
  const recall = truth.length === 0 ? 0 : tp / truth.length
  return {
    precision,
    recall,
    f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
    truePositives: tp,
  }
}

/**
 * The entities that matched pairs form: the connected components of the graph whose nodes are the left records
 * followed by the right ones and whose edges are the matches.
 *
 * @param matches - The pairs called matches.
 * @param nLeft - The number of left records.
 * @param nRight - The number of right records.
 * @returns The entity of each record (left records first), numbered from 0.
 * @concept record-linkage
 * @example
 * ```ts
 * import { clusterMatches } from 'omai-engine/text'
 * Array.from(clusterMatches([[0, 0]], 2, 1)) // => [0, 1, 0]
 * ```
 */
export function clusterMatches(matches: readonly RecordPair[], nLeft: number, nRight: number): Int32Array {
  const g = graph(
    matches.map(([i, j]) => [i, nLeft + j] as [number, number]),
    { nodes: nLeft + nRight },
  )
  return connectedComponents(g).labels
}
