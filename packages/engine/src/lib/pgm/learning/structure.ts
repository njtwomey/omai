import { type Json, WhiteboxError } from '../../base/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { type Dag, type DirectedModel, dag } from '../graphs.ts'
import { code, type DataRow, dataVariables } from './_data.ts'
import { meekRules, type Pdag, pdag } from './equivalence.ts'
import { type CiTest, type IndependenceOptions, independenceOracle } from './independence.ts'
import {
  type DiscreteScore,
  discreteScorer,
  type GaussianScore,
  gaussianScorer,
  type ScoreOptions,
  structurePrior,
} from './scores.ts'

/** Mutual information (nats) of two discrete columns, optionally within the rows where `mask` holds. */
function mutualInformation(x: Int32Array, y: Int32Array, kx: number, ky: number, rows: readonly number[]): number {
  const joint = new Float64Array(kx * ky)
  const px = new Float64Array(kx)
  const py = new Float64Array(ky)
  let n = 0
  for (const i of rows) {
    const a = x[i] ?? -1
    const b = y[i] ?? -1
    if (a < 0 || b < 0) continue
    joint[a * ky + b] = (joint[a * ky + b] ?? 0) + 1
    px[a] = (px[a] ?? 0) + 1
    py[b] = (py[b] ?? 0) + 1
    n++
  }
  let mi = 0
  for (let a = 0; a < kx; a++) {
    for (let b = 0; b < ky; b++) {
      const c = joint[a * ky + b] ?? 0
      if (c > 0) mi += (c / n) * Math.log((c * n) / ((px[a] ?? 1) * (py[b] ?? 1)))
    }
  }
  return mi
}

/** Options of {@link chowLiu}. */
export interface ChowLiuOptions {
  /** The root of the tree (default: the variable with the largest total mutual information, as pgmpy). */
  readonly root?: string
  /** Learn a tree-augmented naive Bayes model with this class variable instead (edges weighted by I(X; Y | class)). */
  readonly classVariable?: string
  readonly states?: Readonly<Record<string, readonly string[]>>
}

/**
 * The Chow-Liu tree (Chow and Liu 1968): the maximum-weight spanning tree of the pairwise mutual information,
 * oriented away from a root, which is the tree-structured distribution closest to the data in KL divergence (pgmpy's
 * TreeSearch). With a class variable it learns a tree-augmented naive Bayes model (Friedman et al. 1997).
 *
 * @param data - Discrete rows.
 * @param options - Root, class variable and states.
 * @returns The tree as a DAG.
 * @concept structure-learning
 * @example
 * ```ts
 * import { dagEdges } from 'whitebox-engine/pgm'
 * import { chowLiu } from 'whitebox-engine/pgm/learning'
 * const data = Array.from({ length: 60 }, (_, i) => ({ a: i % 2, b: i % 2, c: (i % 2) ^ (i % 3 === 0 ? 1 : 0) }))
 * dagEdges(chowLiu(data, { root: 'a' })) // => [['a', 'b'], ['a', 'c']]
 * ```
 */
export function chowLiu(data: readonly DataRow[], options: ChowLiuOptions = {}): Dag {
  const all = dataVariables(data)
  const cls = options.classVariable
  const names = all.filter((v) => v !== cls)
  const d = code(data, all, options.states)
  const rows = Array.from({ length: d.n }, (_, i) => i)
  const groups =
    cls === undefined
      ? [rows]
      : Array.from({ length: d.states[cls]?.length ?? 0 }, (_, c) => rows.filter((i) => d.codes[cls]?.[i] === c))
  const weight = (a: string, b: string): number =>
    groups.reduce(
      (s, g) =>
        s +
        (g.length / d.n) *
          mutualInformation(
            d.codes[a] ?? new Int32Array(0),
            d.codes[b] ?? new Int32Array(0),
            d.states[a]?.length ?? 1,
            d.states[b]?.length ?? 1,
            g,
          ),
      0,
    )
  const pairs: { a: string; b: string; w: number }[] = []
  for (let i = 0; i < names.length; i++)
    for (let j = i + 1; j < names.length; j++)
      pairs.push({ a: names[i] as string, b: names[j] as string, w: weight(names[i] as string, names[j] as string) })
  const total = new Map(names.map((v) => [v, pairs.reduce((s, p) => s + (p.a === v || p.b === v ? p.w : 0), 0)]))
  const root = options.root ?? [...names].sort((x, y) => (total.get(y) ?? 0) - (total.get(x) ?? 0))[0] ?? ''
  if (!names.includes(root)) throw new WhiteboxError('value', `chowLiu: unknown root ${root}`)
  // Kruskal on decreasing weight.
  const parentOf = new Map(names.map((v) => [v, v]))
  const find = (v: string): string => {
    let r = v
    while ((parentOf.get(r) ?? r) !== r) r = parentOf.get(r) ?? r
    return r
  }
  const tree = new Map(names.map((v) => [v, [] as string[]]))
  for (const p of [...pairs].sort((x, y) => y.w - x.w)) {
    const ra = find(p.a)
    const rb = find(p.b)
    if (ra === rb) continue
    parentOf.set(ra, rb)
    tree.get(p.a)?.push(p.b)
    tree.get(p.b)?.push(p.a)
  }
  const edges: [string, string][] = []
  const seen = new Set([root])
  const queue = [root]
  while (queue.length > 0) {
    const v = queue.shift() as string
    for (const w of tree.get(v) ?? []) {
      if (seen.has(w)) continue
      seen.add(w)
      edges.push([v, w])
      queue.push(w)
    }
  }
  if (cls !== undefined) for (const v of names) edges.push([cls, v])
  return dag(edges, all)
}

/** Options of {@link pcAlgorithm} and {@link pc}. */
export interface PcOptions extends IndependenceOptions {
  /** The conditional independence test (default `chiSquare`; use `fisherZ` for continuous data). */
  readonly test?: CiTest
  /** The largest conditioning set tried (default 5, as pgmpy). */
  readonly maxConditioning?: number
  /**
   * The separating set kept when several are found at the same level: the `first` found (PC-stable, Colombo and
   * Maathuis 2014; default) or the `union` of all of them (pgmpy's stable variant).
   */
  readonly sepsetRule?: 'first' | 'union'
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a PC run: the current skeleton, the separating sets, and the conditioning-set size reached. */
export interface PcState {
  readonly adjacency: ReadonlyMap<string, ReadonlySet<string>>
  readonly sepsets: ReadonlyMap<string, readonly string[]>
  readonly level: number
  readonly removed: readonly (readonly [string, string])[]
  readonly finished: boolean
}

const pairKey = (a: string, b: string): string => (a < b ? `${a}|${b}` : `${b}|${a}`)

function* subsets(items: readonly string[], k: number, start = 0, acc: string[] = []): Generator<string[]> {
  if (acc.length === k) {
    yield [...acc]
    return
  }
  for (let i = start; i < items.length; i++) {
    acc.push(items[i] as string)
    yield* subsets(items, k, i + 1, acc)
    acc.pop()
  }
}

/**
 * The skeleton phase of the PC algorithm (Spirtes, Glymour and Scheines 2000) as an {@link Algorithm}, in its
 * order-independent "stable" form: step l tests, for every remaining edge x - y, conditioning sets of size l drawn
 * from the neighbours of x or of y (as they stood at the start of the step), removes the edges found independent and
 * records their separating sets. Removals are recorded as `removeEdge` messages.
 *
 * @param data - The rows.
 * @param options - Test, significance, largest conditioning set and separating-set rule.
 * @returns The algorithm.
 * @concept constraint-based-learning
 * @example
 * ```ts
 * import { pcAlgorithm } from 'whitebox-engine/pgm/learning'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const data = Array.from({ length: 200 }, (_, i) => ({ a: i % 2, b: i % 2, c: Math.floor(i / 2) % 2 }))
 * runAlgorithm(pcAlgorithm(data), { maxSteps: 10 }).state.removed.length // => 2
 * ```
 */
export function pcAlgorithm(data: readonly DataRow[], options: PcOptions = {}): Algorithm<PcState> {
  const names = dataVariables(data)
  const oracle = independenceOracle(data, options.test ?? 'chiSquare', options)
  const maxCond = options.maxConditioning ?? 5
  const union = options.sepsetRule === 'union'
  return {
    name: 'pc',
    config: {
      test: typeof options.test === 'string' ? options.test : 'custom',
      maxConditioning: maxCond,
      significance: options.significance ?? 0.01,
    } as Json,
    schema: channels({
      metrics: { edges: { description: 'edges remaining in the skeleton' } },
      messages: { removeEdge: { description: 'an edge removed with its separating set' } },
    }),
    init: () => ({
      adjacency: new Map(names.map((v) => [v, new Set(names.filter((w) => w !== v))])),
      sepsets: new Map(),
      level: 0,
      removed: [],
      finished: false,
    }),
    step: (s, ctx) => {
      const l = s.level
      const neighbours = new Map([...s.adjacency].map(([v, ns]) => [v, [...ns].sort()]))
      const adjacency = new Map([...s.adjacency].map(([v, ns]) => [v, new Set(ns)]))
      const sepsets = new Map(s.sepsets)
      const removed: [string, string][] = []
      for (const x of names) {
        for (const y of neighbours.get(x) ?? []) {
          if (y <= x) continue
          const candidates = new Map<string, string[]>()
          for (const side of [x, y]) {
            const other = side === x ? y : x
            const pool = (neighbours.get(side) ?? []).filter((v) => v !== other)
            for (const sub of subsets(pool, l)) candidates.set(sub.join(','), sub)
          }
          const found: string[][] = []
          for (const key of [...candidates.keys()].sort()) {
            const z = candidates.get(key) ?? []
            if (oracle(x, y, z)) {
              found.push(z)
              if (!union) break
            }
          }
          if (found.length > 0) {
            const sep = union ? [...new Set(found.flat())].sort() : (found[0] ?? [])
            sepsets.set(pairKey(x, y), sep)
            adjacency.get(x)?.delete(y)
            adjacency.get(y)?.delete(x)
            removed.push([x, y])
            ctx.run.message(l, 'removeEdge', { x, y, separatingSet: sep }, { from: x, to: y })
          }
        }
      }
      const finished = l >= maxCond || names.every((v) => (adjacency.get(v)?.size ?? 0) - 1 < l + 1)
      return { adjacency, sepsets, level: l + 1, removed: [...s.removed, ...removed], finished }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    observe: (s) => ({ metrics: { edges: [...s.adjacency.values()].reduce((a, ns) => a + ns.size, 0) / 2 } }),
  }
}

/** Result of {@link pc}. */
export interface PcResult {
  /** The learned CPDAG (colliders oriented, then Meek's rules). */
  readonly cpdag: Pdag
  /** The separating set of each removed edge, keyed `"a|b"` (sorted). */
  readonly separatingSets: Readonly<Record<string, readonly string[]>>
  readonly status: RunStatus
}

/**
 * The PC algorithm (pgmpy's PC): learn the skeleton by conditional independence tests ({@link pcAlgorithm}), orient
 * every unshielded triple x - z - y with z outside the separating set of x and y as a collider x -> z <- y (unless that
 * contradicts an orientation already made), and complete the orientation with Meek's rules.
 *
 * @param data - The rows (discrete for chi-square and G tests, continuous for Fisher z and partial correlation).
 * @param options - Test, significance, largest conditioning set, separating-set rule and tracer.
 * @returns The CPDAG and the separating sets.
 * @concept constraint-based-learning
 * @example
 * ```ts
 * import { pc } from 'whitebox-engine/pgm/learning'
 * // a -> c <- b: c is the "or" of two independent coins.
 * const data = Array.from({ length: 400 }, (_, i) => ({ a: i % 2, b: Math.floor(i / 2) % 2, c: (i % 2) | (Math.floor(i / 2) % 2) }))
 * pc(data).cpdag.directed // => [['a', 'c'], ['b', 'c']]
 * ```
 */
export function pc(data: readonly DataRow[], options: PcOptions = {}): PcResult {
  const result = runAlgorithm(pcAlgorithm(data, options), {
    maxSteps: (options.maxConditioning ?? 5) + 2,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const { adjacency, sepsets } = result.state
  const names = dataVariables(data)
  const directed = new Set<string>()
  const undirected = new Set<string>()
  for (const [v, ns] of adjacency) for (const w of ns) if (v < w) undirected.add(`${v}|${w}`)
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const x = names[i] as string
      const y = names[j] as string
      if (adjacency.get(x)?.has(y)) continue
      const sep = sepsets.get(pairKey(x, y)) ?? []
      for (const z of [...(adjacency.get(x) ?? [])].filter((w) => adjacency.get(y)?.has(w)).sort()) {
        if (sep.includes(z)) continue
        // Orient only when both edges are still undirected (no conflicting earlier orientation).
        if (undirected.has(pairKey(x, z)) && undirected.has(pairKey(y, z))) {
          undirected.delete(pairKey(x, z))
          undirected.delete(pairKey(y, z))
          directed.add(`${x}|${z}`)
          directed.add(`${y}|${z}`)
        }
      }
    }
  }
  const split = (k: string): [string, string] => k.split('|') as [string, string]
  const cp = meekRules(pdag(names, [...directed].map(split), [...undirected].map(split)), { r4: true })
  return { cpdag: cp, separatingSets: Object.fromEntries(sepsets), status: result.status }
}

/** Options of {@link hillClimbAlgorithm} and {@link hillClimb}. */
export interface HillClimbOptions extends ScoreOptions {
  /** The score (default `bic` for discrete data; `bicGauss` and friends for continuous data). */
  readonly score?: DiscreteScore | GaussianScore
  /** The starting DAG (default: no edges). */
  readonly start?: DirectedModel
  /** The largest number of parents allowed (default unlimited). */
  readonly maxIndegree?: number
  /** Recently applied operations that may not be undone (default 100, as pgmpy). */
  readonly tabuLength?: number
  /** Stop when the best improvement is below this (default 1e-4, as pgmpy). */
  readonly epsilon?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a hill-climbing run. */
export interface HillClimbState {
  /** Parents of each variable. */
  readonly parents: ReadonlyMap<string, readonly string[]>
  readonly score: number
  readonly tabu: readonly string[]
  readonly last?: {
    readonly operation: '+' | '-' | 'flip'
    readonly from: string
    readonly to: string
    readonly delta: number
  }
  readonly stuck: boolean
}

function reaches(parents: ReadonlyMap<string, readonly string[]>, from: string, to: string): boolean {
  // Whether a directed path from -> ... -> to exists (following child links).
  const children = new Map<string, string[]>()
  for (const [v, ps] of parents) for (const p of ps) children.set(p, [...(children.get(p) ?? []), v])
  const stack = [from]
  const seen = new Set<string>()
  while (stack.length > 0) {
    const v = stack.pop() as string
    if (v === to) return true
    if (seen.has(v)) continue
    seen.add(v)
    stack.push(...(children.get(v) ?? []))
  }
  return false
}

/**
 * Greedy hill climbing over DAGs as an {@link Algorithm} (Koller and Friedman, section 18.4.3; pgmpy's
 * HillClimbSearch): each step applies the single edge addition, removal or reversal that most improves a decomposable
 * score (BIC, AIC, K2, BDeu, BDs, log-likelihood, or their Gaussian forms), subject to acyclicity, a parent limit and a
 * tabu list of recent operations, and stops when no move improves the score by epsilon. Each move is recorded as a
 * `move` message, the score as a metric.
 *
 * @param data - The rows.
 * @param options - Score, start, parent limit, tabu length, epsilon and score options.
 * @returns The algorithm.
 * @concept score-based-learning
 * @example
 * ```ts
 * import { hillClimbAlgorithm } from 'whitebox-engine/pgm/learning'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const data = Array.from({ length: 100 }, (_, i) => ({ a: i % 2, b: i % 2 }))
 * // One improving move, then a step that finds none.
 * runAlgorithm(hillClimbAlgorithm(data), { maxSteps: 10 }).steps // => 2
 * ```
 */
export function hillClimbAlgorithm(
  data: readonly DataRow[],
  options: HillClimbOptions = {},
): Algorithm<HillClimbState> {
  const names = dataVariables(data)
  const score = options.score ?? 'bic'
  const local =
    score === 'bicGauss' || score === 'aicGauss' || score === 'logLikelihoodGauss'
      ? gaussianScorer(data, names, score)
      : discreteScorer(data, names, score, options)
  const maxIndegree = options.maxIndegree ?? Number.POSITIVE_INFINITY
  const tabuLength = options.tabuLength ?? 100
  const epsilon = options.epsilon ?? 1e-4
  const total = (parents: ReadonlyMap<string, readonly string[]>): number =>
    names.reduce((s, v) => s + local(v, parents.get(v) ?? []), 0) +
    structurePrior(
      score,
      [...parents.values()].reduce((k, ps) => k + ps.length, 0),
      names.length,
    )
  // The change of the log structure prior when an edge is added or removed (nonzero only for BDs).
  const priorStep = structurePrior(score, 1, 0) - structurePrior(score, 0, 0)
  return {
    name: 'hillClimb',
    config: { score, maxIndegree: Number.isFinite(maxIndegree) ? maxIndegree : -1, tabuLength, epsilon } as Json,
    schema: channels({
      metrics: { score: { description: 'structure score', direction: 'higher' } },
      messages: { move: { description: 'the edge operation applied' } },
    }),
    init: () => {
      const parents = new Map(names.map((v) => [v, [] as string[]]))
      for (const n of options.start?.nodes ?? []) parents.set(n.name, [...n.parents])
      return { parents, score: total(parents), tabu: [], stuck: false }
    },
    step: (s, ctx) => {
      let best: { operation: '+' | '-' | 'flip'; from: string; to: string; delta: number } | undefined
      const consider = (operation: '+' | '-' | 'flip', from: string, to: string, delta: number): void => {
        if (best === undefined || delta > best.delta) best = { operation, from, to, delta }
      }
      const tabu = new Set(s.tabu)
      for (const x of names) {
        for (const y of names) {
          if (x === y) continue
          const py = s.parents.get(y) ?? []
          const px = s.parents.get(x) ?? []
          if (py.includes(x)) {
            if (!tabu.has(`-:${x}:${y}`))
              consider(
                '-',
                x,
                y,
                local(
                  y,
                  py.filter((v) => v !== x),
                ) -
                  local(y, py) -
                  priorStep,
              )
            // Reverse x -> y unless another path x -> ... -> y exists.
            const without = new Map(s.parents)
            without.set(
              y,
              py.filter((v) => v !== x),
            )
            if (
              !tabu.has(`flip:${x}:${y}`) &&
              !tabu.has(`flip:${y}:${x}`) &&
              px.length + 1 <= maxIndegree &&
              !reaches(without, x, y)
            ) {
              consider(
                'flip',
                x,
                y,
                local(x, [...px, y]) +
                  local(
                    y,
                    py.filter((v) => v !== x),
                  ) -
                  local(x, px) -
                  local(y, py),
              )
            }
          } else if (!px.includes(y)) {
            if (!tabu.has(`+:${x}:${y}`) && py.length + 1 <= maxIndegree && !reaches(s.parents, y, x)) {
              consider('+', x, y, local(y, [...py, x]) - local(y, py) + priorStep)
            }
          }
        }
      }
      const chosen = best as { operation: '+' | '-' | 'flip'; from: string; to: string; delta: number } | undefined
      if (chosen === undefined || chosen.delta < epsilon) return { ...s, stuck: true }
      const parents = new Map(s.parents)
      const { operation, from, to } = chosen
      if (operation === '+') parents.set(to, [...(parents.get(to) ?? []), from])
      else if (operation === '-')
        parents.set(
          to,
          (parents.get(to) ?? []).filter((v) => v !== from),
        )
      else {
        parents.set(
          to,
          (parents.get(to) ?? []).filter((v) => v !== from),
        )
        parents.set(from, [...(parents.get(from) ?? []), to])
      }
      // The inverse operation becomes tabu.
      const inverse =
        operation === '+' ? `-:${from}:${to}` : operation === '-' ? `+:${from}:${to}` : `flip:${to}:${from}`
      const nextTabu = [...s.tabu, inverse].slice(-tabuLength)
      ctx.run.message(ctx.t, 'move', { operation, from, to, delta: chosen.delta }, { from, to })
      return { parents, score: s.score + chosen.delta, tabu: nextTabu, last: chosen, stuck: false }
    },
    done: (s) => (s.stuck ? 'converged' : undefined),
    observe: (s) => ({ metrics: { score: s.score } }),
  }
}

/**
 * Learn a DAG by greedy hill climbing ({@link hillClimbAlgorithm}).
 *
 * @param data - The rows.
 * @param options - Score, start, parent limit, tabu length, epsilon, score options and tracer.
 * @returns The learned DAG and its score.
 * @concept score-based-learning
 * @example
 * ```ts
 * import { dagEdges } from 'whitebox-engine/pgm'
 * import { hillClimb } from 'whitebox-engine/pgm/learning'
 * const data = Array.from({ length: 100 }, (_, i) => ({ a: i % 2, b: i % 2, c: Math.floor(i / 2) % 2 }))
 * dagEdges(hillClimb(data).dag).length // => 1
 * ```
 */
export function hillClimb(
  data: readonly DataRow[],
  options: HillClimbOptions & { readonly maxIter?: number } = {},
): { readonly dag: Dag; readonly score: number; readonly status: RunStatus } {
  const result = runAlgorithm(hillClimbAlgorithm(data, options), {
    maxSteps: options.maxIter ?? 1000000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const names = dataVariables(data)
  const edges = names.flatMap((v) => (result.state.parents.get(v) ?? []).map((p): [string, string] => [p, v]))
  return { dag: dag(edges, names), score: result.state.score, status: result.status }
}

/**
 * The empirical mutual information (nats) between two discrete variables (scikit-learn's `mutual_info_score`), the
 * edge weight of the Chow-Liu tree.
 *
 * @param data - Discrete rows.
 * @param x - First variable.
 * @param y - Second variable.
 * @returns I(X; Y).
 * @concept structure-learning
 * @example
 * ```ts
 * import { mutualInformationOf } from 'whitebox-engine/pgm/learning'
 * Math.abs(mutualInformationOf([{ a: 0, b: 0 }, { a: 1, b: 1 }], 'a', 'b') - Math.LN2) < 1e-15 // => true
 * ```
 */
export function mutualInformationOf(data: readonly DataRow[], x: string, y: string): number {
  const d = code(data, [x, y])
  return mutualInformation(
    d.codes[x] ?? new Int32Array(0),
    d.codes[y] ?? new Int32Array(0),
    d.states[x]?.length ?? 1,
    d.states[y]?.length ?? 1,
    Array.from({ length: d.n }, (_, i) => i),
  )
}
