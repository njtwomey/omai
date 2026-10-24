// Weighted level graphs and the moves shared by Louvain and Leiden (not re-exported). A level graph is an undirected
// weighted graph whose nodes are communities of the level below; gains are written in the common form
// e(v, C) - gamma * w(v) * W(C), with node weights w the degrees (modularity, gamma = resolution / 2m) or the node
// counts (the constant Potts model, gamma = resolution).
import { kahanSum, maxOf } from '../base/index.ts'
import type { Graph } from './graph.ts'

/** An undirected weighted graph of communities. */
export interface LevelGraph {
  readonly n: number
  readonly nbr: readonly Int32Array[]
  readonly wt: readonly Float64Array[]
  /** Self-loop weight (each loop counted once). */
  readonly self: Float64Array
  /** Weighted degree, self-loops counted twice. */
  readonly k: Float64Array
  /** Original nodes inside. */
  readonly size: Float64Array
}

/** The level graph of a graph (directed graphs are read as undirected, reciprocal weights summed). */
export function levelGraph(g: Graph, weighted: boolean): LevelGraph {
  const maps: Map<number, number>[] = Array.from({ length: g.n }, () => new Map())
  const self = new Float64Array(g.n)
  for (const e of g.edges) {
    const w = weighted ? e.weight : 1
    if (e.source === e.target) {
      self[e.source] = (self[e.source] ?? 0) + w
      continue
    }
    const a = maps[e.source] as Map<number, number>
    const b = maps[e.target] as Map<number, number>
    a.set(e.target, (a.get(e.target) ?? 0) + w)
    b.set(e.source, (b.get(e.source) ?? 0) + w)
  }
  return fromMaps(maps, self, new Float64Array(g.n).fill(1))
}

function fromMaps(maps: readonly Map<number, number>[], self: Float64Array, size: Float64Array): LevelGraph {
  const nbr = maps.map((m) => Int32Array.from([...m.keys()].sort((p, q) => p - q)))
  const wt = maps.map((m, i) => Float64Array.from(nbr[i] ?? [], (j) => m.get(j) ?? 0))
  const k = Float64Array.from(self, (s, i) => {
    let d = 2 * s
    for (const w of wt[i] ?? []) d += w
    return d
  })
  return { n: maps.length, nbr, wt, self, k, size }
}

/** Total edge weight m (self-loops once). */
export function totalWeight(l: LevelGraph): number {
  let m = 0
  for (let i = 0; i < l.n; i++) m += (l.k[i] ?? 0) / 2
  return m
}

/** Contract each community to a node. Labels must be 0 .. c - 1. */
export function aggregate(l: LevelGraph, labels: Int32Array, count: number): LevelGraph {
  const maps: Map<number, number>[] = Array.from({ length: count }, () => new Map())
  const self = new Float64Array(count)
  const size = new Float64Array(count)
  for (let v = 0; v < l.n; v++) {
    const c = labels[v] ?? 0
    self[c] = (self[c] ?? 0) + (l.self[v] ?? 0)
    size[c] = (size[c] ?? 0) + (l.size[v] ?? 0)
    const nb = l.nbr[v] ?? new Int32Array(0)
    const wt = l.wt[v] ?? new Float64Array(0)
    for (let t = 0; t < nb.length; t++) {
      const d = labels[nb[t] ?? 0] ?? 0
      const w = wt[t] ?? 0
      if (d === c) self[c] = (self[c] ?? 0) + w / 2
      else {
        const m = maps[c] as Map<number, number>
        m.set(d, (m.get(d) ?? 0) + w)
      }
    }
  }
  return fromMaps(maps, self, size)
}

/** Relabel to 0 .. c - 1 in order of first appearance. */
export function compact(labels: Int32Array): { labels: Int32Array; count: number } {
  const map = new Map<number, number>()
  const out = Int32Array.from(labels, (c) => {
    let x = map.get(c)
    if (x === undefined) {
      x = map.size
      map.set(c, x)
    }
    return x
  })
  return { labels: out, count: map.size }
}

/** The quality function: node weights and the effective resolution of the gain e - gamma w W. */
export interface Quality {
  readonly weight: Float64Array
  readonly gamma: number
  /** Divide a gain by this to express it in quality units (m for modularity, 1 for CPM). */
  readonly unit: number
}

export function quality(l: LevelGraph, kind: 'modularity' | 'cpm', resolution: number, m: number): Quality {
  return kind === 'cpm'
    ? { weight: l.size, gamma: resolution, unit: 1 }
    : { weight: l.k, gamma: resolution / (2 * m || 1), unit: m || 1 }
}

/** Weights from v to each neighbouring community (excluding v's self-loop). */
export function linksTo(l: LevelGraph, labels: Int32Array, v: number): Map<number, number> {
  const out = new Map<number, number>()
  const nb = l.nbr[v] ?? new Int32Array(0)
  const wt = l.wt[v] ?? new Float64Array(0)
  for (let t = 0; t < nb.length; t++) {
    const c = labels[nb[t] ?? 0] ?? 0
    out.set(c, (out.get(c) ?? 0) + (wt[t] ?? 0))
  }
  return out
}

const EPS = 1e-12

/**
 * Louvain local moving (as NetworkX's _one_level): sweep the nodes in the given order, moving each to the neighbouring
 * community of largest positive gain, until a sweep moves nothing. Returns whether anything moved.
 */
export function louvainMoves(l: LevelGraph, labels: Int32Array, q: Quality, order: readonly number[]): boolean {
  const total = new Float64Array(l.n)
  for (let v = 0; v < l.n; v++) total[labels[v] ?? 0] = (total[labels[v] ?? 0] ?? 0) + (q.weight[v] ?? 0)
  let moved = false
  for (;;) {
    let moves = 0
    for (const v of order) {
      const own = labels[v] ?? 0
      const w = q.weight[v] ?? 0
      const links = linksTo(l, labels, v)
      total[own] = (total[own] ?? 0) - w
      let best = own
      let bestGain = (links.get(own) ?? 0) - q.gamma * w * (total[own] ?? 0)
      for (const [c, e] of links) {
        const gain = e - q.gamma * w * (total[c] ?? 0)
        if (gain > bestGain + EPS) {
          best = c
          bestGain = gain
        }
      }
      total[best] = (total[best] ?? 0) + w
      if (best !== own) {
        labels[v] = best
        moves++
      }
    }
    if (moves === 0) return moved
    moved = true
  }
}

/**
 * Leiden fast local moving: a queue of nodes in the given order; each is moved to the community (or a new empty one)
 * of largest gain, and when it moves its neighbours outside the new community rejoin the queue. Returns whether
 * anything moved. `fresh` supplies unused labels for empty communities.
 */
export function fastMoves(
  l: LevelGraph,
  labels: Int32Array,
  q: Quality,
  order: readonly number[],
  fresh: () => number,
): boolean {
  const total = new Map<number, number>()
  for (let v = 0; v < l.n; v++) total.set(labels[v] ?? 0, (total.get(labels[v] ?? 0) ?? 0) + (q.weight[v] ?? 0))
  const queue = [...order]
  const queued = new Uint8Array(l.n).fill(1)
  let moved = false
  for (let head = 0; head < queue.length; head++) {
    const v = queue[head] ?? 0
    queued[v] = 0
    const own = labels[v] ?? 0
    const w = q.weight[v] ?? 0
    const links = linksTo(l, labels, v)
    total.set(own, (total.get(own) ?? 0) - w)
    let best = own
    let bestGain = (links.get(own) ?? 0) - q.gamma * w * (total.get(own) ?? 0)
    for (const [c, e] of links) {
      const gain = e - q.gamma * w * (total.get(c) ?? 0)
      if (gain > bestGain + EPS) {
        best = c
        bestGain = gain
      }
    }
    if (bestGain < -EPS && (total.get(own) ?? 0) > 0) best = fresh()
    total.set(best, (total.get(best) ?? 0) + w)
    if (best === own) continue
    labels[v] = best
    moved = true
    const nb = l.nbr[v] ?? new Int32Array(0)
    for (const u of nb) {
      if (queued[u] === 0 && labels[u] !== best) {
        queued[u] = 1
        queue.push(u)
      }
    }
  }
  return moved
}

/**
 * Leiden refinement: inside each community, start from singletons and merge each well-connected singleton node into a
 * well-connected sub-community chosen at random with probability proportional to exp(gain / theta) over the
 * non-negative gains (staying alone has gain 0). Returns refined labels (unique across communities).
 */
export function refine(
  l: LevelGraph,
  labels: Int32Array,
  q: Quality,
  order: readonly number[],
  theta: number,
  random: () => number,
): Int32Array {
  const refined = Int32Array.from({ length: l.n }, (_, v) => v)
  const w = q.weight
  const communityWeight = new Map<number, number>()
  for (let v = 0; v < l.n; v++) {
    communityWeight.set(labels[v] ?? 0, (communityWeight.get(labels[v] ?? 0) ?? 0) + (w[v] ?? 0))
  }
  // Weight and external links (to the rest of the community) of each refined sub-community, and its member count.
  const subWeight = Float64Array.from(w)
  const subExternal = new Float64Array(l.n)
  const members = new Int32Array(l.n).fill(1)
  for (let v = 0; v < l.n; v++) {
    const nb = l.nbr[v] ?? new Int32Array(0)
    const wt = l.wt[v] ?? new Float64Array(0)
    for (let t = 0; t < nb.length; t++)
      if (labels[nb[t] ?? 0] === labels[v]) subExternal[v] = (subExternal[v] ?? 0) + (wt[t] ?? 0)
  }
  for (const v of order) {
    if ((members[refined[v] ?? 0] ?? 0) !== 1) continue
    const c = labels[v] ?? 0
    const W = communityWeight.get(c) ?? 0
    const wv = w[v] ?? 0
    // v itself must be well connected to its community.
    if ((subExternal[v] ?? 0) < q.gamma * wv * (W - wv) - EPS) continue
    const links = new Map<number, number>()
    const nb = l.nbr[v] ?? new Int32Array(0)
    const wt = l.wt[v] ?? new Float64Array(0)
    for (let t = 0; t < nb.length; t++) {
      const u = nb[t] ?? 0
      if (labels[u] !== c) continue
      const s = refined[u] ?? 0
      links.set(s, (links.get(s) ?? 0) + (wt[t] ?? 0))
    }
    const own = refined[v] ?? 0
    const candidates: [number, number][] = [[own, 0]]
    for (const [s, e] of links) {
      if (s === own) continue
      const ws = subWeight[s] ?? 0
      if ((subExternal[s] ?? 0) < q.gamma * ws * (W - ws) - EPS) continue
      const gain = e - q.gamma * wv * ws
      if (gain >= 0) candidates.push([s, gain / q.unit])
    }
    const top = maxOf(candidates.map(([, g]) => g))
    const scores = candidates.map(([, g]) => Math.exp((g - top) / theta))
    const total = kahanSum(scores)
    let u = random() * total
    let chosen = own
    for (const [i, [s]] of candidates.entries()) {
      u -= scores[i] ?? 0
      chosen = s
      if (u <= 0) break
    }
    if (chosen === own) continue
    const e = links.get(chosen) ?? 0
    subExternal[chosen] = (subExternal[chosen] ?? 0) + (subExternal[v] ?? 0) - 2 * e
    subWeight[chosen] = (subWeight[chosen] ?? 0) + wv
    members[chosen] = (members[chosen] ?? 0) + 1
    members[own] = 0
    refined[v] = chosen
  }
  return refined
}
