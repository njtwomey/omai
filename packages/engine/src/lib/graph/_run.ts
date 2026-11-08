// Small helpers shared by the stepped processes of the graph domain (spreading, opinion, percolation, growth): a state
// codec for checkpoints, the nodes a process starts from, and argument checks. Not re-exported.
import { type Json, OmaiError } from '../base/index.ts'
import { type Key, subsample } from '../random/index.ts'
import { stateFromJson, stateToJson } from '../trace/index.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'

/** The `serialise` and `deserialise` of an algorithm whose state is plain objects, arrays and typed arrays. */
export function codec<S>(): { serialise: (s: S) => Json; deserialise: (j: Json) => S } {
  return { serialise: (s) => stateToJson(s), deserialise: (j) => stateFromJson(j) as S }
}

/** Resolve seed nodes: the given labels or indices, or `count` nodes drawn without replacement from the eligible ones. */
export function resolveSeeds(
  name: string,
  g: Graph,
  seeds: readonly NodeId[] | number,
  key: Key,
  excluded: ReadonlySet<number> = new Set(),
): number[] {
  if (typeof seeds !== 'number') {
    const out = [...new Set(seeds.map((v) => nodeIndex(g, v)))]
    return out.sort((a, b) => a - b)
  }
  const eligible: number[] = []
  for (let v = 0; v < g.n; v++) if (!excluded.has(v)) eligible.push(v)
  if (!(Number.isInteger(seeds) && seeds >= 0 && seeds <= eligible.length))
    throw new OmaiError('value', `${name}: cannot draw ${seeds} seeds from ${eligible.length} eligible nodes`)
  const pick = Array.from(subsample(key, eligible.length, seeds).data)
  return pick.map((i) => eligible[i] ?? 0).sort((a, b) => a - b)
}

/** Check that a rate or probability is finite and in range. */
export function checkNumber(
  name: string,
  what: string,
  x: number,
  lo: number,
  hi: number = Number.POSITIVE_INFINITY,
): void {
  if (!(x >= lo && x <= hi)) throw new OmaiError('value', `${name}: ${what} must be in [${lo}, ${hi}], got ${x}`)
}

/** The weight of adjacency entry `k` of the out-edges: its weight, or 1 for every edge when `weighted` is false. */
export function weightFunction(g: Graph, weighted: boolean | undefined): (k: number) => number {
  const w = g.out.weights
  return weighted === false ? () => 1 : (k) => w[k] ?? 0
}

/** A float array as JSON: the non-finite entries (an unreached node's infinite distance) as strings. */
export function floatsToJson(x: ArrayLike<number>): Json {
  return Array.from(x, (v) => (Number.isFinite(v) ? v : v > 0 ? 'Infinity' : v < 0 ? '-Infinity' : 'NaN'))
}

/** The inverse of {@link floatsToJson}. */
export function floatsFromJson(j: Json): Float64Array {
  return Float64Array.from(j as readonly (number | string)[], (v) => (typeof v === 'number' ? v : Number(v)))
}
