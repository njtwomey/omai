/**
 * Graph-based approximate nearest-neighbour search: HNSW (hierarchical navigable small-world graphs, Malkov and
 * Yashunin), which searches greedily down layers of ever sparser graphs, and NN-descent, which builds an approximate
 * k-nearest-neighbour graph from the idea that a neighbour of a neighbour is likely a neighbour. Searches record what
 * they visited, for figures.
 */

import { heap, type Json, MdmlError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { type Metric, metricFunction } from '../geometry/index.ts'

type Scored = readonly [number, number] // [distance, node]

/** Options of {@link hnswIndex}. */
export interface HnswOptions {
  /** Links per node on upper layers (layer 0 has 2M). Default 16. */
  readonly m?: number
  /** Beam width while building (default 200). */
  readonly efConstruction?: number
  /** Level multiplier (default 1 / ln M). */
  readonly levelMultiplier?: number
  readonly metric?: Metric
  readonly key: Key
}

/** What a graph search did, layer by layer. */
export interface GraphSearchLog {
  /** The entry point used at each layer, top first. */
  readonly path: readonly number[]
  /** Nodes whose distance was computed, per layer (top first). */
  readonly visited: readonly (readonly number[])[]
  readonly distanceEvaluations: number
}

/** A hierarchical navigable small-world graph. */
export interface HnswIndex {
  readonly size: number
  /** Each node's top layer. */
  readonly levels: readonly number[]
  /** links[node][layer]: the neighbour lists. */
  readonly links: readonly (readonly (readonly number[])[])[]
  readonly entryPoint: number
  /** Insert points `[n, d]`. */
  readonly add: (points: TensorLike) => void
  /** The k (approximate) nearest neighbours with beam width ef (default max(k, 50)). */
  readonly search: (
    q: ArrayLike<number>,
    k: number,
    options?: { readonly ef?: number; readonly log?: boolean },
  ) => { readonly indices: number[]; readonly distances: number[]; readonly log?: GraphSearchLog }
}

/** The paths {@link hnswAlgorithm} logs. */
export type HnswPath = 'hnsw/insert' | 'hnsw/layer' | 'hnsw/neighbours' | 'hnsw/entry' | 'hnsw/evaluations'

/** What the HNSW build logs: the node inserted, its top layer, its neighbours on layer 0, the entry point and the work. */
export const hnswContract: Contract<unknown, HnswPath> = defineContract({
  'hnsw/insert': scalar({ description: 'the node inserted at this step' }),
  'hnsw/layer': scalar({ description: "the new node's top layer" }),
  'hnsw/neighbours': tensor({ role: 'state', shape: ['links'], description: "the new node's links on layer 0" }),
  'hnsw/entry': scalar({ description: 'the entry point of the graph (the node on the highest layer)' }),
  'hnsw/evaluations': scalar({ description: 'distance evaluations so far' }),
})

/**
 * The state of {@link hnswAlgorithm} (and the index behind {@link hnswIndex}): the points inserted, each node's top layer
 * and neighbour lists, the entry point, and the distance evaluations. Steps append to the arrays in place; keep a copy
 * of a state you want to look at later.
 */
export interface HnswState {
  /** The inserted points, flat (n by d). */
  readonly points: number[]
  readonly levels: number[]
  /** links[node][layer]: the neighbour lists. */
  readonly links: number[][][]
  entry: number
  inserted: number
  evaluations: number
}

/** The machinery of one HNSW graph over a state: distance, layer search, neighbour selection and insertion. */
function hnswKernel(d: number, options: HnswOptions, s: HnswState) {
  const m = options.m ?? 16
  const mMax0 = 2 * m
  const ef0 = options.efConstruction ?? 200
  const mL = options.levelMultiplier ?? 1 / Math.log(Math.max(2, m))
  const dist = metricFunction(options.metric ?? 'euclidean', d)
  const { points, levels, links } = s
  const pd = (q: ArrayLike<number>, i: number): number => {
    s.evaluations++
    return dist(q, 0, points, i * d)
  }
  const pointOf = (i: number): number[] => points.slice(i * d, i * d + d)

  const searchLayer = (
    q: ArrayLike<number>,
    eps: readonly Scored[],
    ef: number,
    layer: number,
    seen?: number[],
  ): Scored[] => {
    const visited = new Set<number>(eps.map((e) => e[1]))
    for (const e of eps) seen?.push(e[1])
    const candidates = heap((a: Scored, b: Scored) => a[0] - b[0], eps)
    const results = heap((a: Scored, b: Scored) => b[0] - a[0], eps)
    while (candidates.size > 0) {
      const c = candidates.pop() as Scored
      const furthest = results.peek() as Scored
      if (c[0] > furthest[0]) break
      for (const e of links[c[1]]?.[layer] ?? []) {
        if (visited.has(e)) continue
        visited.add(e)
        seen?.push(e)
        const de = pd(q, e)
        const f = results.peek() as Scored
        if (results.size < ef || de < f[0]) {
          candidates.push([de, e])
          results.push([de, e])
          if (results.size > ef) results.pop()
        }
      }
    }
    return results.toArray().sort((a, b) => a[0] - b[0] || a[1] - b[1])
  }

  const selectNeighbours = (candidates: readonly Scored[], count: number): number[] => {
    const sorted = [...candidates].sort((a, b) => a[0] - b[0] || a[1] - b[1])
    const kept: Scored[] = []
    const pruned: Scored[] = []
    for (const c of sorted) {
      if (kept.length >= count) break
      const point = pointOf(c[1])
      // Keep c only if it is closer to the base than to every neighbour already kept.
      if (kept.every((r) => dist(point, 0, points, r[1] * d) > c[0])) kept.push(c)
      else pruned.push(c)
    }
    for (const c of pruned) {
      if (kept.length >= count) break
      kept.push(c)
    }
    return kept.map((c) => c[1])
  }

  /** Insert one point; returns its id and top layer. */
  const insert = (q: number[]): { readonly id: number; readonly level: number } => {
    const id = s.inserted++
    points.push(...q)
    const u = uniform(foldIn(options.key, id), [1]).data[0] ?? 0.5
    const level = Math.floor(-Math.log(Math.max(u, Number.MIN_VALUE)) * mL)
    levels.push(level)
    links.push(Array.from({ length: level + 1 }, () => []))
    if (s.entry < 0) {
      s.entry = id
      return { id, level }
    }
    const top = levels[s.entry] ?? 0
    let eps: Scored[] = [[pd(q, s.entry), s.entry]]
    for (let layer = top; layer > level; layer--) eps = searchLayer(q, eps, 1, layer).slice(0, 1)
    for (let layer = Math.min(top, level); layer >= 0; layer--) {
      const found = searchLayer(q, eps, ef0, layer)
      const neighbours = selectNeighbours(found, m)
      ;(links[id] as number[][])[layer] = neighbours
      const cap = layer === 0 ? mMax0 : m
      for (const e of neighbours) {
        const list = (links[e] as number[][])[layer] as number[]
        list.push(id)
        if (list.length > cap) {
          const pe = pointOf(e)
          const scored = list.map((v): Scored => [dist(pe, 0, points, v * d), v])
          ;(links[e] as number[][])[layer] = selectNeighbours(scored, cap)
        }
      }
      eps = found
    }
    if (level > top) s.entry = id
    return { id, level }
  }
  return { insert, searchLayer, pd }
}

const emptyHnsw = (): HnswState => ({ points: [], levels: [], links: [], entry: -1, inserted: 0, evaluations: 0 })

/**
 * HNSW construction as a stepped algorithm, one insertion per step: draw the new node's top layer, descend greedily
 * from the entry point to that layer, then on each layer at or below it search for candidates with beam width
 * efConstruction, link to M of them by the neighbour-selection heuristic and prune the neighbours' lists. It logs the
 * node, its layer, its layer-0 links, the entry point and the distance evaluations, and is done (`converged`) when
 * every point is in. The state serialises.
 *
 * @param data - Points `[n, d]` to insert.
 * @param options - M, efConstruction, level multiplier, metric and key.
 * @returns The algorithm.
 * @concept approximate-nearest-neighbours
 * @example
 * ```ts
 * import { hnswAlgorithm } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(hnswAlgorithm([[0, 0], [1, 0], [0, 1], [5, 5]], { m: 2, key: key(0) }), { maxSteps: 10 })
 * [run.status, run.state.inserted] // => ['converged', 4]
 * ```
 */
export function hnswAlgorithm(data: TensorLike, options: HnswOptions): Algorithm<HnswState, HnswPath> {
  const first = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'hnswIndex')
  const n = first.shape[0] ?? 0
  const d = first.shape[1] ?? 0
  const flat = toFlat(first)
  return {
    name: 'numerics/hnsw',
    config: { n, d, m: options.m ?? 16, efConstruction: options.efConstruction ?? 200 },
    init: () => emptyHnsw(),
    step: (s, ctx) => {
      const { id, level } = hnswKernel(d, options, s).insert(flat.slice(s.inserted * d, s.inserted * d + d))
      ctx.log('hnsw/insert', id)
      ctx.log('hnsw/layer', level)
      ctx.log('hnsw/neighbours', s.links[id]?.[0] ?? [])
      ctx.log('hnsw/entry', s.entry)
      ctx.log('hnsw/evaluations', s.evaluations)
      return s
    },
    done: (s) => (s.inserted >= n ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HnswState,
  }
}

/**
 * A hierarchical navigable small-world index (Malkov and Yashunin 2018): every point lives on layer 0 and on each
 * higher layer with geometrically falling probability; searches descend greedily from the sparse top layer and finish
 * with a beam search on layer 0. Links are chosen by the paper's neighbour-selection heuristic (keep a candidate only
 * if it is closer to the new point than to every link already kept, then fill with the pruned ones). The first points
 * are inserted by {@link hnswAlgorithm} run to the end.
 *
 * @param data - Points `[n, d]` to insert (more can be added later).
 * @param options - M, efConstruction, level multiplier, metric and key.
 * @returns The index.
 * @concept approximate-nearest-neighbours
 * @example
 * ```ts
 * import { hnswIndex } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * const index = hnswIndex([[0, 0], [1, 0], [0, 1], [5, 5]], { m: 2, key: key(0) })
 * index.search([4.8, 5.1], 1).indices // => [3]
 * ```
 */
export function hnswIndex(data: TensorLike, options: HnswOptions): HnswIndex {
  const first = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'hnswIndex')
  const d = first.shape[1] ?? 0
  const state = runAlgorithm(hnswAlgorithm(first, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state
  const { searchLayer, pd, insert } = hnswKernel(d, options, state)
  const { levels, links } = state

  const add = (input: TensorLike): void => {
    const t = concreteOrThrow(promoteFloat(input)[0] as Tensor, 'hnswIndex add')
    if (t.shape[1] !== d) throw new MdmlError('shape', 'hnswIndex: dimension mismatch')
    const f = toFlat(t)
    for (let i = 0; i < (t.shape[0] ?? 0); i++) insert(f.slice(i * d, i * d + d))
  }

  const self: HnswIndex = {
    get size() {
      return state.inserted
    },
    levels,
    links,
    get entryPoint() {
      return state.entry
    },
    add,
    search: (q, k, searchOptions = {}) => {
      const entry = state.entry
      if (entry < 0) return { indices: [], distances: [] }
      const ef = Math.max(k, searchOptions.ef ?? 50)
      const before = state.evaluations
      const path: number[] = []
      const visited: number[][] = []
      let eps: Scored[] = [[pd(q, entry), entry]]
      for (let layer = levels[entry] ?? 0; layer > 0; layer--) {
        path.push(eps[0]?.[1] ?? entry)
        const seen: number[] = []
        eps = searchLayer(q, eps, 1, layer, seen).slice(0, 1)
        visited.push(seen)
      }
      path.push(eps[0]?.[1] ?? entry)
      const seen: number[] = []
      const found = searchLayer(q, eps, ef, 0, seen).slice(0, k)
      visited.push(seen)
      return {
        indices: found.map((f) => f[1]),
        distances: found.map((f) => f[0]),
        ...(searchOptions.log === true
          ? { log: { path, visited, distanceEvaluations: state.evaluations - before } }
          : {}),
      }
    },
  }
  return self
}

/** State of NN-descent: each point's current neighbour list with "new" flags. */
export interface NnDescentState {
  readonly indices: Int32Array
  readonly distances: Float64Array
  readonly isNew: Uint8Array
  readonly updates: number
  readonly recall: number
}

/**
 * NN-descent (Dong, Moses and Li 2011): build an approximate k-nearest-neighbour graph by repeated local joins ("a
 * neighbour of a neighbour is likely a neighbour"), sampling rho K new and old neighbours and their reverses per point,
 * until fewer than delta n K updates happen in an iteration. A traced algorithm (updates and, with ground truth, recall
 * per iteration).
 *
 * @param data - Points `[n, d]`.
 * @param k - Neighbours per point.
 * @param options - Sample rate rho (default 0.5), delta (default 0.001), metric, key and ground-truth neighbours.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { nnDescentAlgorithm } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * nnDescentAlgorithm([[0], [1], [2], [3]], 2, { key: key(0) }).name // => 'nnDescent'
 * ```
 */
export function nnDescentAlgorithm(
  data: TensorLike,
  k: number,
  options: {
    readonly rho?: number
    readonly delta?: number
    readonly metric?: Metric
    readonly key: Key
    readonly truth?: ArrayLike<number>
  },
): Algorithm<NnDescentState> {
  const t = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'nnDescent')
  const flat = toFlat(t)
  const [n = 0, d = 0] = t.shape
  if (k >= n) throw new MdmlError('value', 'nnDescent: k must be smaller than the number of points')
  const dist = metricFunction(options.metric ?? 'euclidean', d)
  const rho = options.rho ?? 0.5
  const delta = options.delta ?? 0.001
  const sampleSize = Math.max(1, Math.round(rho * k))
  const recallOf = (indices: Int32Array): number => {
    if (options.truth === undefined) return Number.NaN
    let hit = 0
    for (let v = 0; v < n; v++) {
      const truth = new Set(Array.from({ length: k }, (_, j) => options.truth?.[v * k + j] ?? -1))
      for (let j = 0; j < k; j++) if (truth.has(indices[v * k + j] ?? -2)) hit++
    }
    return hit / (n * k)
  }
  /** Insert (u, l) into v's list if it beats the worst entry; returns 1 on change. */
  const update = (
    st: { indices: Int32Array; distances: Float64Array; isNew: Uint8Array },
    v: number,
    u: number,
    l: number,
  ): number => {
    const base = v * k
    if (u === v || l >= (st.distances[base + k - 1] ?? 0)) return 0
    for (let j = 0; j < k; j++) if (st.indices[base + j] === u) return 0
    let pos = k - 1
    while (pos > 0 && (st.distances[base + pos - 1] ?? 0) > l) {
      st.indices[base + pos] = st.indices[base + pos - 1] ?? 0
      st.distances[base + pos] = st.distances[base + pos - 1] ?? 0
      st.isNew[base + pos] = st.isNew[base + pos - 1] ?? 0
      pos--
    }
    st.indices[base + pos] = u
    st.distances[base + pos] = l
    st.isNew[base + pos] = 1
    return 1
  }
  return {
    name: 'nnDescent',
    config: { k, rho, delta } as Json,
    schema: channels({
      metrics: {
        updates: { description: 'neighbour-list updates this iteration' },
        recall: { description: 'recall against the true neighbours, when given' },
      },
    }),
    init: () => {
      const indices = new Int32Array(n * k)
      const distances = new Float64Array(n * k)
      const isNew = new Uint8Array(n * k).fill(1)
      const u = uniform(foldIn(options.key, 0), [n * k * 2]).data
      let c = 0
      for (let v = 0; v < n; v++) {
        const chosen = new Set<number>()
        while (chosen.size < k) {
          const r = Math.floor((u[c++ % u.length] ?? 0) * n)
          if (r !== v) chosen.add(r)
          if (c > u.length) for (let x = 0; chosen.size < k; x++) if (x !== v) chosen.add(x)
        }
        const sorted = [...chosen].map((w) => [dist(flat, v * d, flat, w * d), w] as const).sort((a, b) => a[0] - b[0])
        sorted.forEach(([l, w], j) => {
          indices[v * k + j] = w
          distances[v * k + j] = l
        })
      }
      return { indices, distances, isNew, updates: n * k, recall: recallOf(indices) }
    },
    step: (s, ctx) => {
      const st = {
        indices: Int32Array.from(s.indices),
        distances: Float64Array.from(s.distances),
        isNew: Uint8Array.from(s.isNew),
      }
      const rand = uniform(ctx.key, [n * k * 4]).data
      let ri = 0
      const sample = (xs: number[], size: number): number[] => {
        const copy = [...xs]
        for (let i = copy.length - 1; i > 0; i--) {
          const j = Math.floor((rand[ri++ % rand.length] ?? 0) * (i + 1))
          ;[copy[i], copy[j]] = [copy[j] as number, copy[i] as number]
        }
        return copy.slice(0, size)
      }
      const oldL: number[][] = Array.from({ length: n }, () => [])
      const newL: number[][] = Array.from({ length: n }, () => [])
      for (let v = 0; v < n; v++) {
        const fresh: number[] = []
        for (let j = 0; j < k; j++) {
          const u = st.indices[v * k + j] ?? 0
          if (st.isNew[v * k + j]) fresh.push(j)
          else oldL[v]?.push(u)
        }
        for (const j of sample(fresh, sampleSize)) {
          newL[v]?.push(st.indices[v * k + j] ?? 0)
          st.isNew[v * k + j] = 0
        }
      }
      const oldR: number[][] = Array.from({ length: n }, () => [])
      const newR: number[][] = Array.from({ length: n }, () => [])
      for (let v = 0; v < n; v++) {
        for (const u of oldL[v] ?? []) oldR[u]?.push(v)
        for (const u of newL[v] ?? []) newR[u]?.push(v)
      }
      let updates = 0
      for (let v = 0; v < n; v++) {
        const olds = [...new Set([...(oldL[v] ?? []), ...sample(oldR[v] ?? [], sampleSize)])]
        const news = [...new Set([...(newL[v] ?? []), ...sample(newR[v] ?? [], sampleSize)])]
        for (let a = 0; a < news.length; a++) {
          const u1 = news[a] as number
          const partners = [...news.slice(a + 1), ...olds]
          for (const u2 of partners) {
            if (u1 === u2) continue
            const l = dist(flat, u1 * d, flat, u2 * d)
            updates += update(st, u1, u2, l) + update(st, u2, u1, l)
          }
        }
      }
      return { ...st, updates, recall: recallOf(st.indices) }
    },
    done: (s) => (s.updates < delta * n * k ? 'converged' : undefined),
    observe: (s) => ({ metrics: { updates: s.updates, ...(Number.isNaN(s.recall) ? {} : { recall: s.recall }) } }),
  }
}

/**
 * Build an approximate k-nearest-neighbour graph by NN-descent.
 *
 * @param data - Points `[n, d]`.
 * @param k - Neighbours per point.
 * @param options - As {@link nnDescentAlgorithm}, plus the iteration cap (default 20) and trace options.
 * @returns Neighbour indices and distances `[n, k]`, sorted, and the per-iteration updates.
 * @example
 * ```ts
 * import { nnDescent } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * nnDescent([[0], [1], [2], [10], [11], [12]], 2, { key: key(0) }).indices.shape // => [6, 2]
 * ```
 */
export function nnDescent(
  data: TensorLike,
  k: number,
  options: Parameters<typeof nnDescentAlgorithm>[2] &
    Partial<Omit<RunOptions, 'maxSteps' | 'key'>> & { readonly maxIterations?: number },
): {
  readonly indices: Tensor
  readonly distances: Tensor
  readonly updates: readonly number[]
  readonly iterations: number
} {
  const { rho, delta, metric, key, truth, maxIterations, ...run } = options
  const updates: number[] = []
  const algorithm = nnDescentAlgorithm(data, k, {
    key,
    ...(rho === undefined ? {} : { rho }),
    ...(delta === undefined ? {} : { delta }),
    ...(metric === undefined ? {} : { metric }),
    ...(truth === undefined ? {} : { truth }),
  })
  const wrapped: Algorithm<NnDescentState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      updates.push(next.updates)
      return next
    },
  }
  const result = runAlgorithm(wrapped, { ...run, key, maxSteps: maxIterations ?? 20 })
  const n = result.state.indices.length / k
  return {
    indices: fromStorage(result.state.indices, [n, k], 'int32'),
    distances: fromStorage(result.state.distances, [n, k], 'float64'),
    updates,
    iterations: result.steps,
  }
}
