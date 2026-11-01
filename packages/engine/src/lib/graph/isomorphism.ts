/**
 * Graph isomorphism: deciding whether two graphs are the same up to relabelling nodes. Weisfeiler-Lehman colour
 * refinement (as a traced algorithm) and the WL graph hash, and VF2-style backtracking search for isomorphisms,
 * subgraph isomorphisms and monomorphisms, as in NetworkX's GraphMatcher.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { dict, type OptionTable, optionTable } from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  iterate,
  runAlgorithm,
  scalar,
  tensor,
} from '../trace/index.ts'
import { knobOption } from './_options.ts'
import { codec } from './_run.ts'
import { degree, type Graph, type GraphRunOptions } from './graph.ts'

/** A 64-bit FNV-1a hash of a string as 16 hex digits (two 32-bit lanes with different offsets). */
function hash(text: string): string {
  let a = 0x811c9dc5
  let b = 0x050c5d1f
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i)
    a = Math.imul(a ^ c, 0x01000193) >>> 0
    b = Math.imul(b ^ c, 0x01000193) >>> 0
    b = Math.imul(b ^ (a >>> 7), 0x5bd1e995) >>> 0
  }
  return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0')
}

/** The state of Weisfeiler-Lehman colour refinement. */
export interface WlState {
  /** The colour (a label) of each node. */
  readonly colours: readonly string[]
  /** The number of distinct colours. */
  readonly count: number
  /** The colours after each round, the initial ones first. */
  readonly history: readonly (readonly string[])[]
  /** Whether the last round split no colour class. */
  readonly stable: boolean
}

/** Options of the Weisfeiler-Lehman functions. */
export interface WlOptions {
  /** Initial node labels (default: the degrees, as NetworkX). */
  readonly nodeLabels?: ArrayLike<string | number>
  /** Edge labels in edge order, mixed into each neighbour's contribution. */
  readonly edgeLabels?: ArrayLike<string | number>
}

/**
 * Weisfeiler-Lehman colour refinement (1-WL) as a traced algorithm: each step replaces every node's colour by a hash
 * of its colour and the sorted multiset of its neighbours' colours (successors for a directed graph, as NetworkX),
 * until the colour classes stop splitting. Hashing (rather than per-graph relabelling) makes colours comparable across
 * graphs, as in NetworkX's weisfeiler_lehman_graph_hash.
 *
 * @param g - The graph.
 * @param options - Initial node labels and edge labels.
 * @returns The algorithm.
 * @concept weisfeiler-lehman
 * @reference B. Weisfeiler and A. Leman, The reduction of a graph to canonical form (1968); N. Shervashidze et al.,
 *   Weisfeiler-Lehman graph kernels, JMLR 12 (2011).
 * @example
 * ```ts
 * import { pathGraph, weisfeilerLehmanAlgorithm } from 'mdml-engine/graph'
 * weisfeilerLehmanAlgorithm(pathGraph(4)).name // => 'weisfeiler-lehman'
 * ```
 */
export function weisfeilerLehmanAlgorithm(g: Graph, options: WlOptions = {}): Algorithm<WlState> {
  const { initial, refine } = refiner(g, options)
  const distinct = (colours: readonly string[]): number => new Set(colours).size
  return {
    name: 'weisfeiler-lehman',
    config: { labelled: options.nodeLabels !== undefined, edgeLabelled: options.edgeLabels !== undefined } as Json,
    schema: channels({ metrics: { colours: { description: 'number of distinct colours', direction: 'higher' } } }),
    init: () => ({ colours: initial, count: distinct(initial), history: [initial], stable: false }),
    step: (s) => {
      const colours = refine(s.colours)
      const count = distinct(colours)
      return { colours, count, history: [...s.history, colours], stable: count === s.count }
    },
    done: (s) => (s.stable ? 'converged' : undefined),
    observe: (s) => ({ metrics: { colours: s.count } }),
  }
}

/** The initial colours and one refinement round. */
function refiner(
  g: Graph,
  options: WlOptions,
): { readonly initial: readonly string[]; readonly refine: (colours: readonly string[]) => string[] } {
  const a = g.out
  const degrees = degree(g)
  const initial = Array.from({ length: g.n }, (_, v) =>
    options.nodeLabels === undefined ? String(degrees[v] ?? 0) : String(options.nodeLabels[v] ?? ''),
  )
  const edgeLabel = (k: number): string =>
    options.edgeLabels === undefined ? '' : String(options.edgeLabels[a.edges[k] ?? 0] ?? '')
  // JSON keeps the encoding unambiguous for any labels: no colour, edge label or neighbour count can collide with
  // another (with plain concatenation, an isolated node and a node with one empty-labelled neighbour collided).
  const refine = (colours: readonly string[]): string[] =>
    Array.from({ length: g.n }, (_, v) => {
      const parts: string[] = []
      for (let k = a.indptr[v] ?? 0; k < (a.indptr[v + 1] ?? 0); k++) {
        parts.push(JSON.stringify([edgeLabel(k), colours[a.indices[k] ?? 0] ?? '']))
      }
      return hash(JSON.stringify([colours[v] ?? '', parts.sort()]))
    })
  return { initial, refine }
}

/**
 * The stable Weisfeiler-Lehman colouring: run refinement until the colour classes stop splitting (at most n rounds).
 *
 * @param g - The graph.
 * @param options - Initial node and edge labels, and tracing.
 * @returns The final colours, their count, the per-round history and the number of rounds.
 * @concept weisfeiler-lehman
 * @example
 * ```ts
 * import { pathGraph, weisfeilerLehman } from 'mdml-engine/graph'
 * const r = weisfeilerLehman(pathGraph(5))
 * [r.count, r.colours[0] === r.colours[4], r.colours[1] === r.colours[3]] // => [3, true, true]
 * ```
 */
export function weisfeilerLehman(
  g: Graph,
  options: WlOptions & GraphRunOptions = {},
): WlState & { readonly rounds: number } {
  const { nodeLabels, edgeLabels, ...run } = options
  const algorithm = weisfeilerLehmanAlgorithm(g, {
    ...(nodeLabels === undefined ? {} : { nodeLabels }),
    ...(edgeLabels === undefined ? {} : { edgeLabels }),
  })
  const result = runAlgorithm(algorithm, { ...run, maxSteps: g.n + 1 })
  return { ...result.state, rounds: result.steps }
}

/**
 * The Weisfeiler-Lehman graph hash (NetworkX weisfeiler_lehman_graph_hash): a hash of the colour histograms over a
 * fixed number of refinement rounds. Isomorphic graphs always hash equally; graphs that 1-WL cannot tell apart (such as
 * two disjoint triangles and a hexagon) do too. The digests differ from NetworkX's (which uses blake2b), but they are
 * equal exactly when NetworkX's are.
 *
 * @param g - The graph.
 * @param options - `iterations` (default 3), initial node labels and edge labels.
 * @returns A hexadecimal digest.
 * @concept weisfeiler-lehman
 * @example
 * ```ts
 * import { cycleGraph, graph, pathGraph, wlHash } from 'mdml-engine/graph'
 * wlHash(cycleGraph(4)) === wlHash(graph([[2, 3], [3, 0], [0, 1], [1, 2]])) // => true
 * wlHash(cycleGraph(4)) === wlHash(pathGraph(4)) // => false
 * ```
 */
export function wlHash(g: Graph, options: WlOptions & { readonly iterations?: number } = {}): string {
  const iterations = options.iterations ?? 3
  const { initial, refine } = refiner(g, options)
  let colours = initial
  const histogram = (colours: readonly string[]): string => {
    const counts = new Map<string, number>()
    for (const c of colours) counts.set(c, (counts.get(c) ?? 0) + 1)
    return [...counts.entries()]
      .sort((p, q) => (p[0] < q[0] ? -1 : p[0] > q[0] ? 1 : 0))
      .map(([c, k]) => `${c}*${k}`)
      .join(';')
  }
  const parts: string[] = []
  for (let i = 0; i < iterations; i++) {
    colours = refine(colours)
    parts.push(histogram(colours))
  }
  return hash(parts.join('|'))
}

/** Which correspondence {@link vf2} searches for. */
export type MatchMode = 'isomorphism' | 'subgraph' | 'monomorphism'

/** Options of the matching functions. */
export interface MatchOptions {
  /** Whether a node of the pattern may map to a node of the host (default: always). */
  readonly nodeMatch?: (patternNode: number, hostNode: number) => boolean
  /** The most mappings to produce (default all). */
  readonly limit?: number
}

/** The paths the VF2 search logs, as a type. */
export type Vf2Path = 'vf2/mapping' | 'vf2/depth' | 'vf2/backtrack' | 'vf2/matches' | 'vf2/tried'

/**
 * What the VF2 search logs, by path: the partial mapping after the step (host node per pattern node, -1 for unmapped),
 * the depth reached, whether the step backtracked (1) or extended the mapping (0), the matchings found so far and the
 * candidate pairs tried (accepted or rejected) so far.
 */
export const vf2Contract: Contract<unknown, Vf2Path> = defineContract({
  'vf2/mapping': tensor({
    role: 'state',
    shape: ['pattern'],
    description: 'the host node each pattern node is mapped to so far (-1: not yet)',
  }),
  'vf2/depth': scalar({ description: 'the number of pattern nodes mapped after the step' }),
  'vf2/backtrack': scalar({ description: '1 when the step undid a pairing, 0 when it extended the mapping' }),
  'vf2/matches': scalar({ description: 'complete matchings found so far' }),
  'vf2/tried': scalar({ description: 'candidate pairs tried so far, rejected ones included' }),
})

/** One level of the VF2 search: the pattern node being placed and the host nodes it may go to. */
export interface Vf2Frame {
  /** The pattern node placed at this level. */
  readonly p: number
  /** The host candidates, in order. */
  readonly candidates: readonly number[]
  /** The next candidate to try. */
  readonly index: number
  /** The host node currently paired with `p` (-1 when none). */
  readonly assigned: number
}

/** The state of {@link vf2Algorithm}. Steps update the arrays in place; keep a copy to look at later. */
export interface Vf2State {
  /** The levels of the search, outermost first. */
  readonly stack: Vf2Frame[]
  /** Host node to pattern node (-1 unmapped). */
  readonly core1: Int32Array
  /** Pattern node to host node (-1 unmapped). */
  readonly core2: Int32Array
  /** Mapped neighbours of each host node. */
  readonly term1: Int32Array
  /** Mapped neighbours of each pattern node. */
  readonly term2: Int32Array
  /** The complete matchings found: `mapping[p]` is the host node of pattern node p. */
  readonly mappings: number[][]
  /** Candidate pairs tried so far. */
  readonly tried: number
  /** Whether the last step backtracked. */
  readonly backtracked: boolean
}

/**
 * VF2-style matching as a steppable algorithm: each step either pairs the next pattern node (in a connectivity-first
 * order) with the first host candidate that is consistent with the mapping so far, or, when a level has no candidate
 * left (or a complete matching was just recorded), backtracks by undoing a pairing. Done when the search is exhausted
 * or `limit` matchings are found. See {@link vf2} for the modes.
 *
 * @param host - The host graph G1.
 * @param pattern - The pattern graph G2.
 * @param mode - `'isomorphism'` (default), `'subgraph'` (induced) or `'monomorphism'`; see {@link matchModes}.
 * @param options - A node compatibility test and a limit.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when one graph is directed and the other is not.
 * @concept graph-isomorphism
 * @example
 * ```ts
 * import { cycleGraph, vf2Algorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(vf2Algorithm(cycleGraph(4), cycleGraph(4)), { maxSteps: 1000 })
 * [run.status, run.state.mappings.length] // => ['converged', 8]
 * ```
 */
export function vf2Algorithm(
  host: Graph,
  pattern: Graph,
  mode: MatchMode = 'isomorphism',
  options: MatchOptions = {},
): Algorithm<Vf2State, Vf2Path> {
  if (host.directed !== pattern.directed) {
    throw new MdmlError('value', 'vf2: both graphs must be directed or both undirected')
  }
  const np = pattern.n
  const nh = host.n
  const impossible = (mode === 'isomorphism' && (np !== nh || pattern.edges.length !== host.edges.length)) || np > nh
  const limit = options.limit ?? Number.POSITIVE_INFINITY
  const arc = (g: Graph): ((u: number, v: number) => boolean) => {
    const sets = Array.from(
      { length: g.n },
      (_, u) => new Set(g.out.indices.subarray(g.out.indptr[u] ?? 0, g.out.indptr[u + 1] ?? 0)),
    )
    return (u, v) => sets[u]?.has(v) === true
  }
  const pArc = arc(pattern)
  const hArc = arc(host)
  // Underlying undirected neighbourhoods (without self-loops), for ordering and the look-ahead.
  const neighbourhood = (g: Graph): number[][] =>
    Array.from({ length: g.n }, (_, u) => {
      const s = new Set<number>()
      for (const a of g.directed ? [g.out, g.in] : [g.out]) {
        for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) if (a.indices[k] !== u) s.add(a.indices[k] ?? 0)
      }
      return [...s].sort((x, y) => x - y)
    })
  const pN = neighbourhood(pattern)
  const hN = neighbourhood(host)
  // Matching order: repeatedly take the unordered pattern node with most ordered neighbours, then highest degree.
  const order: number[] = []
  const placed = new Uint8Array(np)
  const links = new Int32Array(np)
  for (let i = 0; i < np; i++) {
    let best = -1
    for (let p = 0; p < np; p++) {
      if (placed[p] === 1) continue
      if (
        best < 0 ||
        (links[p] ?? 0) > (links[best] ?? 0) ||
        ((links[p] ?? 0) === (links[best] ?? 0) && (pN[p]?.length ?? 0) > (pN[best]?.length ?? 0))
      ) {
        best = p
      }
    }
    placed[best] = 1
    order.push(best)
    for (const q of pN[best] ?? []) links[q] = (links[q] ?? 0) + 1
  }
  const strict = mode !== 'monomorphism'
  const exact = mode === 'isomorphism'
  const compare = (a: number, b: number): boolean => (exact ? a === b : a <= b)
  const consistent = (s: Vf2State, p: number, h: number): boolean => {
    const { core1, core2, term1, term2 } = s
    if (options.nodeMatch !== undefined && !options.nodeMatch(p, h)) return false
    const selfP = pArc(p, p)
    const selfH = hArc(h, h)
    if (selfP && !selfH) return false
    if (strict && selfH && !selfP) return false
    if (compare(pN[p]?.length ?? 0, hN[h]?.length ?? 0) === false) return false
    for (const q of pN[p] ?? []) {
      const hq = core2[q] ?? -1
      if (hq < 0) continue
      if (pArc(p, q) && !hArc(h, hq)) return false
      if (pArc(q, p) && !hArc(hq, h)) return false
      if (strict && (hArc(h, hq) !== pArc(p, q) || hArc(hq, h) !== pArc(q, p))) return false
    }
    if (strict) {
      // A mapped host neighbour of h must come from a pattern neighbour of p.
      for (const hq of hN[h] ?? []) {
        const q = core1[hq] ?? -1
        if (q >= 0 && !(pArc(p, q) || pArc(q, p))) return false
      }
    }
    // Look-ahead: neighbours in the terminal sets and beyond them.
    let pT = 0
    let pRest = 0
    for (const q of pN[p] ?? []) {
      if ((core2[q] ?? -1) >= 0) continue
      if ((term2[q] ?? 0) > 0) pT++
      else pRest++
    }
    let hT = 0
    let hRest = 0
    for (const hq of hN[h] ?? []) {
      if ((core1[hq] ?? -1) >= 0) continue
      if ((term1[hq] ?? 0) > 0) hT++
      else hRest++
    }
    if (strict) return compare(pT, hT) && compare(pRest, hRest)
    return pT + pRest <= hT + hRest
  }
  const assign = (s: Vf2State, p: number, h: number, sign: 1 | -1): void => {
    s.core2[p] = sign === 1 ? h : -1
    s.core1[h] = sign === 1 ? p : -1
    for (const q of pN[p] ?? []) s.term2[q] = (s.term2[q] ?? 0) + sign
    for (const hq of hN[h] ?? []) s.term1[hq] = (s.term1[hq] ?? 0) + sign
  }
  /** The level that places the pattern node at `depth`: its candidates given the mapping so far. */
  const frame = (s: Vf2State, depth: number): Vf2Frame => {
    const p = order[depth] ?? 0
    // Candidates: neighbours of the host image of an already-mapped pattern neighbour, or every free host node.
    let anchor = -1
    for (const q of pN[p] ?? []) {
      if ((s.core2[q] ?? -1) >= 0) {
        anchor = s.core2[q] ?? -1
        break
      }
    }
    const candidates = anchor >= 0 ? [...(hN[anchor] ?? [])] : Array.from({ length: nh }, (_, h) => h)
    return { p, candidates, index: 0, assigned: -1 }
  }
  return {
    name: 'graph/vf2',
    config: { mode, pattern: np, host: nh, limit: Number.isFinite(limit) ? limit : null },
    init: () => {
      const s: Vf2State = {
        stack: [],
        core1: new Int32Array(nh).fill(-1),
        core2: new Int32Array(np).fill(-1),
        term1: new Int32Array(nh),
        term2: new Int32Array(np),
        mappings: [],
        tried: 0,
        backtracked: false,
      }
      if (impossible) return s
      if (np === 0) s.mappings.push([])
      else s.stack.push(frame(s, 0))
      return s
    },
    step: (s, ctx) => {
      const top = s.stack[s.stack.length - 1]
      if (top === undefined) return s
      let backtracked = false
      let tried = s.tried
      let level = top
      if (top.assigned >= 0) {
        // Undo the pairing of this level (after a complete matching, or to try the next candidate).
        assign(s, top.p, top.assigned, -1)
        level = { ...top, assigned: -1 }
        s.stack[s.stack.length - 1] = level
        backtracked = true
      } else {
        let found = -1
        let index = top.index
        while (index < top.candidates.length) {
          const h = top.candidates[index] ?? 0
          index++
          if ((s.core1[h] ?? -1) >= 0) continue
          tried++
          if (consistent(s, top.p, h)) {
            found = h
            break
          }
        }
        if (found < 0) {
          // No candidate left: leave this level and undo the pairing of the level below.
          s.stack.pop()
          const below = s.stack[s.stack.length - 1]
          if (below !== undefined) {
            assign(s, below.p, below.assigned, -1)
            s.stack[s.stack.length - 1] = { ...below, assigned: -1 }
          }
          backtracked = true
        } else {
          assign(s, top.p, found, 1)
          level = { ...top, index, assigned: found }
          s.stack[s.stack.length - 1] = level
          if (s.stack.length === np) {
            s.mappings.push(Array.from(s.core2))
            if (s.mappings.length >= limit) s.stack.length = 0
          } else s.stack.push(frame(s, s.stack.length))
        }
      }
      if (ctx.wants('vf2/mapping')) ctx.log('vf2/mapping', Int32Array.from(s.core2))
      if (ctx.wants('vf2/depth'))
        ctx.log(
          'vf2/depth',
          s.core2.reduce((c, h) => c + (h >= 0 ? 1 : 0), 0),
        )
      if (ctx.wants('vf2/backtrack')) ctx.log('vf2/backtrack', backtracked ? 1 : 0)
      if (ctx.wants('vf2/matches')) ctx.log('vf2/matches', s.mappings.length)
      if (ctx.wants('vf2/tried')) ctx.log('vf2/tried', tried)
      return { ...s, tried, backtracked }
    },
    done: (s) => (s.stack.length === 0 ? 'converged' : undefined),
    ...codec<Vf2State>(),
  }
}

/** What builds a matching search: the host, the pattern and the options. */
export type MatchBuilder = (host: Graph, pattern: Graph, options?: MatchOptions) => Algorithm<Vf2State, Vf2Path>

/** What a matching mode can do, for validity rules and selectors. */
export interface MatchModeProperties {
  /** Whether the pattern must cover the whole host. */
  readonly onto: boolean
  /** Whether non-edges of the pattern must be non-edges of the host (an induced subgraph). */
  readonly induced: boolean
}

/**
 * The matching modes of VF2 as one slot: `isomorphism` (the pattern onto the whole host), `subgraph` (onto a
 * node-induced subgraph) and `monomorphism` (onto any subgraph, extra host edges allowed).
 *
 * @concept graph-isomorphism
 * @example
 * ```ts
 * import { matchModes } from 'mdml-engine/graph'
 * matchModes.where((p) => p.induced) // => ['isomorphism', 'subgraph']
 * ```
 */
export const matchModes: OptionTable<MatchBuilder, MatchModeProperties> = optionTable(
  {
    isomorphism: knobOption<MatchBuilder, MatchModeProperties>(
      'map the pattern onto the whole host, preserving edges and non-edges',
      { onto: true, induced: true },
      dict({}),
      () => (host, pattern, options) => vf2Algorithm(host, pattern, 'isomorphism', options),
    ),
    subgraph: knobOption<MatchBuilder, MatchModeProperties>(
      'map the pattern onto a node-induced subgraph of the host',
      { onto: false, induced: true },
      dict({}),
      () => (host, pattern, options) => vf2Algorithm(host, pattern, 'subgraph', options),
    ),
    monomorphism: knobOption<MatchBuilder, MatchModeProperties>(
      'map the pattern onto any subgraph of the host: extra host edges are allowed',
      { onto: false, induced: false },
      dict({}),
      () => (host, pattern, options) => vf2Algorithm(host, pattern, 'monomorphism', options),
    ),
  },
  {
    default: 'isomorphism',
    label: 'matching mode',
    description: 'Which correspondence between pattern and host VF2 searches for.',
  },
)

/**
 * All matchings of a pattern graph into a host graph by VF2-style backtracking (Cordella et al., 2004; NetworkX
 * GraphMatcher and DiGraphMatcher): `'isomorphism'` maps the pattern onto the whole host; `'subgraph'` onto a
 * node-induced subgraph (NetworkX subgraph_isomorphisms_iter); `'monomorphism'` onto any subgraph, extra host edges
 * allowed (subgraph_monomorphisms_iter). Pattern nodes are matched in a connectivity-first order; each candidate pair
 * must be consistent with every mapped pair (edges and, except for monomorphisms, non-edges, with directions and
 * self-loops), and VF2's look-ahead on the terminal sets prunes branches that cannot complete. The search is the
 * steppable {@link vf2Algorithm}, run lazily.
 *
 * @param host - The host graph G1.
 * @param pattern - The pattern graph G2.
 * @param mode - The kind of matching (default `'isomorphism'`).
 * @param options - A node compatibility test and a limit.
 * @returns A generator of mappings: `mapping[p]` is the host node of pattern node p.
 * @throws MdmlError with code `value` when one graph is directed and the other is not.
 * @concept graph-isomorphism
 * @reference L. P. Cordella, P. Foggia, C. Sansone and M. Vento, A (sub)graph isomorphism algorithm for matching large
 *   graphs, IEEE TPAMI 26 (2004).
 * @example
 * ```ts
 * import { cycleGraph, vf2 } from 'mdml-engine/graph'
 * Array.from(vf2(cycleGraph(4), cycleGraph(4))).length // => 8
 * ```
 */
export function* vf2(
  host: Graph,
  pattern: Graph,
  mode: MatchMode = 'isomorphism',
  options: MatchOptions = {},
): Generator<number[]> {
  let yielded = 0
  for (const { state } of iterate(vf2Algorithm(host, pattern, mode, options), { maxSteps: Number.MAX_SAFE_INTEGER })) {
    while (yielded < state.mappings.length) yield state.mappings[yielded++] as number[]
  }
}

/**
 * Whether two graphs are isomorphic (NetworkX is_isomorphic), by {@link vf2}.
 *
 * @param g1 - A graph.
 * @param g2 - Another graph.
 * @param options - A node compatibility test (`nodeMatch(nodeOfG2, nodeOfG1)`).
 * @returns True when some bijection preserves edges and non-edges.
 * @concept graph-isomorphism
 * @example
 * ```ts
 * import { isIsomorphic, petersenGraph, graph } from 'mdml-engine/graph'
 * isIsomorphic(petersenGraph(), petersenGraph()) // => true
 * isIsomorphic(graph([[0, 1], [1, 2]]), graph([[0, 1], [0, 2]])) // => true
 * ```
 */
export function isIsomorphic(g1: Graph, g2: Graph, options: Omit<MatchOptions, 'limit'> = {}): boolean {
  return !vf2(g1, g2, 'isomorphism', { ...options, limit: 1 }).next().done
}

/**
 * All embeddings of a pattern in a host graph (NetworkX GraphMatcher(host, pattern).subgraph_isomorphisms_iter, or
 * subgraph_monomorphisms_iter with `induced: false`).
 *
 * @param host - The host graph.
 * @param pattern - The pattern.
 * @param options - `induced` (default true), a node compatibility test and a limit.
 * @returns The mappings, `mapping[p]` the host node of pattern node p.
 * @concept graph-isomorphism
 * @example
 * ```ts
 * import { completeGraph, pathGraph, subgraphIsomorphisms } from 'mdml-engine/graph'
 * subgraphIsomorphisms(completeGraph(4), pathGraph(3)).length // => 0
 * subgraphIsomorphisms(completeGraph(4), pathGraph(3), { induced: false }).length // => 24
 * ```
 */
export function subgraphIsomorphisms(
  host: Graph,
  pattern: Graph,
  options: MatchOptions & { readonly induced?: boolean } = {},
): number[][] {
  const { induced = true, ...rest } = options
  return [...vf2(host, pattern, induced ? 'subgraph' : 'monomorphism', rest)]
}

/**
 * Whether a pattern occurs in a host graph (NetworkX GraphMatcher.subgraph_is_isomorphic, or subgraph_is_monomorphic
 * with `induced: false`).
 *
 * @param host - The host graph.
 * @param pattern - The pattern.
 * @param options - `induced` (default true) and a node compatibility test.
 * @returns True when an embedding exists.
 * @concept graph-isomorphism
 * @example
 * ```ts
 * import { cycleGraph, isSubgraphIsomorphic, petersenGraph } from 'mdml-engine/graph'
 * [isSubgraphIsomorphic(petersenGraph(), cycleGraph(5)), isSubgraphIsomorphic(petersenGraph(), cycleGraph(4))] // => [true, false]
 * ```
 */
export function isSubgraphIsomorphic(
  host: Graph,
  pattern: Graph,
  options: Omit<MatchOptions, 'limit'> & { readonly induced?: boolean } = {},
): boolean {
  const { induced = true, ...rest } = options
  return !vf2(host, pattern, induced ? 'subgraph' : 'monomorphism', { ...rest, limit: 1 }).next().done
}
