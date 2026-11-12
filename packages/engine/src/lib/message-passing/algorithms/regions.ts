/**
 * Region graphs and generalised belief propagation (Yedidia, Freeman and Weiss 2005) for discrete models: the family that
 * contains belief propagation, the Kikuchi cluster variation method and the junction tree as choices of the regions. A
 * region is a set of variables with a counting number; the approximation keeps a belief on every region and asks that
 * beliefs agree where regions overlap. The Bethe region graph (one region per factor, one per variable) gives loopy belief
 * propagation and its free energy, exact on trees. Larger regions that contain the short loops (a Kikuchi graph of
 * overlapping blocks, closed under intersection) give a better answer on a grid at the cost of tables of the region's size.
 * The regions of a junction tree (its cliques and separators) give the exact answer on any model in one pass. The
 * messages run from a region to the regions it contains (parent to child), each set so that the child's belief equals the
 * marginal of the parent's; on the Bethe graph this is exactly the sum-product update.
 *
 * @index module Region graphs and generalised belief propagation: Bethe, Kikuchi and junction regions on discrete models.
 */

import type { Json } from '../../base/index.ts'
import {
  type DiscreteModel,
  type DiscreteVariable,
  type EliminationHeuristic,
  eliminationOrder,
  type Factor,
  factorEntropy,
  factorMarginal,
  factorProductAll,
  interactionGraph,
  type JunctionTree,
  junctionTree,
  logFactor,
  modelFactors,
  modelVariables,
  normaliseFactor,
  permuteFactor,
  unitFactor,
} from '../../pgm/index.ts'
import { choice, dict, int, real } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  channels,
  defineContract,
  defineMethod,
  frame,
  type Method,
  message,
  type RunStatus,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { assertDamping } from '../core/_messages.ts'
import { ParameterError } from '../core/errors.ts'
import type { MpSnapshot } from '../engine/state.ts'
import { shiftedSum } from '../numerics/_moments.ts'

/** A region: its variables, the factors inside it, its counting number and its place in the region graph. */
export interface Region {
  /** The variables, in model order. */
  readonly variables: readonly string[]
  /** The factors (indices into `RegionGraph.factors`) whose scope lies inside this region: all of them enter its belief. */
  readonly factors: readonly number[]
  /** The counting number: 1 minus the counting numbers of the regions that strictly contain it (the Moebius function). */
  readonly count: number
  /** The regions that directly contain this one (no region between). */
  readonly parents: readonly number[]
  /** The regions this one directly contains. */
  readonly children: readonly number[]
}

/** A region graph over a discrete model. */
export interface RegionGraph {
  /** How it was built: `bethe`, `clusters` or `junction`. */
  readonly kind: string
  readonly variables: readonly DiscreteVariable[]
  readonly factors: readonly Factor[]
  readonly regions: readonly Region[]
  /** The parent-to-child pairs: the edges that carry messages. */
  readonly edges: readonly { readonly parent: number; readonly child: number }[]
}

/** How to build a region graph. */
export type RegionOptions =
  | { readonly kind: 'bethe' }
  | { readonly kind: 'clusters'; readonly clusters: readonly (readonly string[])[] }
  | { readonly kind: 'junction'; readonly order?: readonly string[]; readonly heuristic?: EliminationHeuristic }

const sorted = (xs: Iterable<string>, rank: ReadonlyMap<string, number>): string[] =>
  [...xs].sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0))
const key = (xs: readonly string[]): string => xs.join('\u0000')
const subsetOf = (a: readonly string[], b: readonly string[]): boolean => a.every((v) => b.includes(v))

/**
 * Overlapping square blocks of a grid, for the Kikuchi approximation: every `size` by `size` block of the grid of
 * variables named by `name(row, column)` (stride one), so every pairwise factor between neighbours lies in some block.
 *
 * @param rows - The rows of the grid.
 * @param columns - The columns.
 * @param name - The variable's name at a position.
 * @param size - The side of a block (default 2).
 * @returns The blocks, as lists of variable names.
 * @throws OmaiError with code `value` for a size that does not fit the grid.
 * @concept region-graphs
 * @example
 * ```ts
 * import { blockClusters } from 'omai-engine/message-passing'
 * blockClusters(2, 3, (r, c) => `x${r}${c}`) // => [['x00', 'x01', 'x10', 'x11'], ['x01', 'x02', 'x11', 'x12']]
 * ```
 */
export function blockClusters(
  rows: number,
  columns: number,
  name: (row: number, column: number) => string,
  size = 2,
): string[][] {
  if (!(Number.isInteger(size) && size >= 1 && size <= rows && size <= columns))
    throw new ParameterError(`blockClusters: a block of ${size} does not fit a ${rows} by ${columns} grid`)
  const out: string[][] = []
  for (let r = 0; r + size <= rows; r++) {
    for (let c = 0; c + size <= columns; c++) {
      const block: string[] = []
      for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) block.push(name(r + i, c + j))
      out.push(block)
    }
  }
  return out
}

/**
 * Build the region graph of a discrete model. `bethe` takes the largest factor scopes as outer regions and the single
 * variables as inner ones (loopy BP, exact on a tree); `clusters` takes the given clusters as outer regions and closes them
 * under intersection (the Kikuchi cluster variation method: the intersections are the overlaps, with counting numbers that
 * correct for them); `junction` takes the cliques of a triangulation as outer regions (their separators are the
 * intersections), which is exact on any model; its counting numbers are 1 for a clique and -1 for a separator. For the other kinds
 * the counting numbers are the Moebius numbers `c_R = 1 - sum of c_S over the regions S that strictly contain R`. Every factor
 * enters the belief of every region that contains its scope, and the result is checked to be a valid approximation: the
 * counting numbers of the regions that contain a variable, and of those that contain a factor's scope, sum to 1.
 *
 * @param model - The discrete model (a Bayes network, Markov network or factor graph).
 * @param options - The kind and its parameters.
 * @returns The region graph.
 * @throws OmaiError with code `value` for a factor that no region contains, a cluster with an unknown variable, or counting numbers that do not give a valid approximation.
 * @concept region-graphs
 * @index function Build a region graph (Bethe, Kikuchi clusters, junction tree) with counting numbers.
 * @example
 * ```ts
 * import { blockClusters, regionGraph } from 'omai-engine/message-passing'
 * import { factor, factorGraph } from 'omai-engine/pgm'
 * const names = ['a', 'b', 'c', 'd']
 * const fg = factorGraph(
 *   names.map((name) => ({ name, states: ['0', '1'] })),
 *   [factor(['a', 'b'], [2, 2], [2, 1, 1, 2]), factor(['b', 'd'], [2, 2], [2, 1, 1, 2]), factor(['c', 'd'], [2, 2], [2, 1, 1, 2]), factor(['a', 'c'], [2, 2], [2, 1, 1, 2])],
 * )
 * regionGraph(fg, { kind: 'bethe' }).regions.map((r) => r.count) // => [1, 1, 1, 1, -1, -1, -1, -1]
 * ```
 */
export function regionGraph(model: DiscreteModel, options: RegionOptions): RegionGraph {
  const variables = modelVariables(model)
  const factors = modelFactors(model)
  const names = variables.map((v) => v.name)
  const rank = new Map(names.map((n, i) => [n, i]))
  const known = new Set(names)
  for (const f of factors)
    for (const v of f.variables) if (!known.has(v)) throw new ParameterError(`regionGraph: unknown variable ${v}`)
  const scopes = factors.map((f) => sorted(f.variables, rank))
  let outer: string[][]
  let tree: JunctionTree = { cliques: [], edges: [] }
  if (options.kind === 'bethe') {
    const maximal = scopes.filter(
      (s, i) => s.length > 0 && !scopes.some((t, j) => j !== i && t.length > s.length && subsetOf(s, t)),
    )
    outer = [...new Map(maximal.map((s) => [key(s), s])).values()]
    for (const v of names) if (!outer.some((s) => s.includes(v))) outer.push([v])
  } else if (options.kind === 'clusters') {
    for (const c of options.clusters)
      for (const v of c)
        if (!known.has(v)) throw new ParameterError(`regionGraph: the cluster has the unknown variable ${v}`)
    outer = [...new Map(options.clusters.map((c) => sorted(c, rank)).map((c) => [key(c), c])).values()]
  } else {
    const graph = interactionGraph(factors, names)
    const card = Object.fromEntries(variables.map((v) => [v.name, v.states.length]))
    const order = options.order ?? eliminationOrder(graph, names, options.heuristic ?? 'minFill', card)
    tree = junctionTree(graph, order)
    outer = tree.cliques.map((c) => sorted(c, rank))
  }
  // Regions: the outer ones, the closure under intersection of the clusters, the single variables of Bethe.
  let regionVars: string[][]
  let count: number[]
  let parents: number[][]
  let explicit: { parent: number; child: number }[] | undefined
  if (options.kind === 'junction') {
    // The regions of a junction tree are its cliques (counting number 1) and one separator per tree edge (-1), and the
    // only parent-child pairs are an edge's two cliques and its separator: the tree itself, not the inclusion order.
    regionVars = outer.map((c) => c.slice())
    count = outer.map(() => 1)
    explicit = []
    for (const e of tree.edges) {
      if (e.separator.length === 0) continue
      const sep = regionVars.length
      regionVars.push(sorted(e.separator, rank))
      count.push(-1)
      explicit.push({ parent: e.from, child: sep }, { parent: e.to, child: sep })
    }
    const parentsOf = regionVars.map(() => [] as number[])
    for (const e of explicit) (parentsOf[e.child] as number[]).push(e.parent)
    parents = parentsOf
  } else {
    const sets = new Map<string, string[]>(outer.map((s) => [key(s), s]))
    if (options.kind === 'clusters') {
      let grew = true
      while (grew) {
        grew = false
        const now = [...sets.values()]
        for (let i = 0; i < now.length; i++) {
          for (let j = i + 1; j < now.length; j++) {
            const inter = (now[i] as string[]).filter((v) => (now[j] as string[]).includes(v))
            if (inter.length > 0 && !sets.has(key(inter))) {
              sets.set(key(inter), inter)
              grew = true
            }
          }
        }
      }
    } else {
      for (const v of names) if (!sets.has(key([v]))) sets.set(key([v]), [v])
    }
    regionVars = [...sets.values()].sort((a, b) => b.length - a.length || key(a).localeCompare(key(b)))
    // Counting numbers by the Moebius function, from the largest regions down.
    count = new Array<number>(regionVars.length).fill(0)
    for (let r = 0; r < regionVars.length; r++) {
      let above = 0
      for (let t = 0; t < r; t++) {
        const vs = regionVars[t] as string[]
        const vr = regionVars[r] as string[]
        if (vs.length > vr.length && subsetOf(vr, vs)) above += count[t] as number
      }
      count[r] = 1 - above
    }
    // The Hasse diagram: a region's parents are the minimal regions that strictly contain it.
    parents = regionVars.map((vr) => {
      const above = regionVars.flatMap((vs, t) => (vs.length > vr.length && subsetOf(vr, vs) ? [t] : []))
      return above.filter(
        (t) =>
          !above.some(
            (u) =>
              u !== t &&
              (regionVars[u] as string[]).length < (regionVars[t] as string[]).length &&
              subsetOf(regionVars[u] as string[], regionVars[t] as string[]),
          ),
      )
    })
  }
  // A valid approximation counts every variable and every factor exactly once.
  for (const v of names) {
    const total = regionVars.reduce((a, vars, r) => a + (vars.includes(v) ? (count[r] as number) : 0), 0)
    if (total !== 1)
      throw new ParameterError(
        `regionGraph: the counting numbers of the regions containing ${v} sum to ${total}, not 1`,
      )
  }
  const contained = regionVars.map((vars) => scopes.flatMap((scope, fi) => (subsetOf(scope, vars) ? [fi] : [])))
  for (const [fi, scope] of scopes.entries()) {
    const total = regionVars.reduce((a, vars, r) => a + (subsetOf(scope, vars) ? (count[r] as number) : 0), 0)
    if (total !== 1)
      throw new ParameterError(
        `regionGraph: the regions containing the factor ${fi} over [${scope.join(', ')}] have counting numbers that sum to ${total}, not 1 (add the intersection of its regions as a cluster)`,
      )
  }
  const children = regionVars.map((_, r) => parents.flatMap((ps, c) => (ps.includes(r) ? [c] : [])))
  const regions: Region[] = regionVars.map((variablesOf, r) => ({
    variables: variablesOf,
    factors: contained[r] as number[],
    count: count[r] as number,
    parents: parents[r] as number[],
    children: children[r] as number[],
  }))
  const edges = explicit ?? regions.flatMap((reg, parent) => reg.children.map((child) => ({ parent, child })))
  return { kind: options.kind, variables, factors, regions, edges }
}

/** What the algorithm computes once from a region graph: the descendants, the messages each belief takes, the factors in each. */
interface Plan {
  readonly card: Readonly<Record<string, number>>
  /** For each region, the edges whose messages enter its belief. */
  readonly incoming: readonly (readonly number[])[]
  /** For each region, the product of the factors inside it. */
  readonly potential: readonly Factor[]
  /** The home region of every variable's marginal: the smallest region containing it. */
  readonly home: ReadonlyMap<string, number>
}

function planOf(g: RegionGraph): Plan {
  const card = Object.fromEntries(g.variables.map((v) => [v.name, v.states.length]))
  const n = g.regions.length
  const descendants = g.regions.map((_, r) => {
    const seen = new Set<number>()
    const walk = (x: number): void => {
      for (const c of g.regions[x]?.children ?? []) {
        if (!seen.has(c)) {
          seen.add(c)
          walk(c)
        }
      }
    }
    walk(r)
    return seen
  })
  const edgeOf = new Map(g.edges.map((e, i) => [`${e.parent}>${e.child}`, i]))
  const incoming: number[][] = []
  const potential: Factor[] = []
  for (let r = 0; r < n; r++) {
    const closure = new Set<number>([r, ...(descendants[r] as Set<number>)])
    const edges: number[] = []
    for (const d of closure) {
      for (const p of g.regions[d]?.parents ?? []) if (!closure.has(p)) edges.push(edgeOf.get(`${p}>${d}`) as number)
    }
    incoming.push(edges)
    potential.push(factorProductAll((g.regions[r] as Region).factors.map((f) => g.factors[f] as Factor)))
  }
  const home = new Map<string, number>()
  for (const v of g.variables) {
    let best = -1
    for (const [r, reg] of g.regions.entries())
      if (
        reg.variables.includes(v.name) &&
        (best < 0 || reg.variables.length < (g.regions[best] as Region).variables.length)
      )
        best = r
    home.set(v.name, best)
  }
  return { card, incoming, potential, home }
}

/** A log table over a region's variables, in the region's order. */
const tableOf = (g: RegionGraph, plan: Plan, r: number, values: ArrayLike<number>): Factor => {
  const vars = (g.regions[r] as Region).variables
  return logFactor(
    vars,
    vars.map((v) => plan.card[v] ?? 1),
    values,
  )
}

/** The state of a run: a log message on every parent-to-child edge, in the child's variables, normalised. */
export interface RegionState {
  readonly messages: readonly (readonly number[])[]
  /** The marginals of every variable (probabilities, flattened in variable order) at the start of the sweep. */
  readonly start: readonly number[]
  readonly sweeps: number
  /** The edge to update next within a serial sweep; 0 between sweeps. */
  readonly cursor: number
  readonly updates: number
  /** The largest change of a marginal probability over the last completed sweep (Infinity before the first). */
  readonly change: number
  /** The edges sent by the last step. */
  readonly sent: readonly number[]
}

/** The knobs of region-graph belief propagation. */
export interface RegionBpOptions {
  /** `parallel` computes every message from the state at the start of the sweep, `serial` updates edge by edge with the newest messages. */
  readonly schedule?: 'parallel' | 'serial'
  /** For a serial schedule: a step is a whole `sweep` or one `edge` (the anytime resolution). */
  readonly granularity?: 'sweep' | 'edge'
  /** Damping in [0, 1): the new log message is `old + (1 - damping) (new - old)`. */
  readonly damping?: number
  /** Stop when no marginal moves by more than this over a sweep. */
  readonly tol?: number
  /** Stop, uncompleted, after this many sweeps. */
  readonly maxSweeps?: number
}

/** The defaults of region-graph belief propagation: serial sweeps, damping one half (parent-to-child GBP on Kikuchi regions diverges undamped). */
export const regionBpDefaults: Required<RegionBpOptions> = {
  schedule: 'serial',
  granularity: 'sweep',
  damping: 0.5,
  tol: 1e-10,
  maxSweeps: 1000,
}

/** The paths of a region-graph run. */
export type RegionPath =
  | 'rg/change'
  | 'rg/sweeps'
  | 'rg/updates'
  | 'rg/evidence'
  | 'rg/message'
  | 'rg/marginals'
  | 'rg/frame'
  | 'rg/snapshot'

/**
 * What a region-graph belief propagation run logs, by path.
 *
 * @concept region-graphs
 */
export const regionContract: Contract<RegionState, RegionPath> = defineContract({
  'rg/change': scalar({
    from: 'change',
    description: 'the largest change of a marginal over the last completed sweep',
  }),
  'rg/sweeps': scalar({ from: 'sweeps', description: 'completed sweeps' }),
  'rg/updates': scalar({ from: 'updates', description: 'messages sent so far' }),
  'rg/evidence': scalar({
    from: 'evidence',
    description: 'the region-based (Bethe, Kikuchi or exact) estimate of the log partition function',
    direction: 'higher',
  }),
  'rg/message': message({
    from: 'message',
    description: 'a message from a region to one it contains (ids: region:<index>)',
    payload: { parent: 'the parent region', child: 'the child region', edge: 'the edge index' },
  }),
  'rg/marginals': message({ from: 'marginals', description: 'the current marginal of every variable, by name' }),
  'rg/frame': frame({ description: 'the marginals by variable name and the messages of the last step' }),
  'rg/snapshot': message({
    description: 'an anytime snapshot (see runAnytimeWith): where the run is and every marginal',
  }),
})

/** The unnormalised belief of a region under the messages: its potential times the messages that enter it. */
function beliefOf(g: RegionGraph, plan: Plan, messages: readonly (readonly number[])[], r: number): Factor {
  const reg = g.regions[r] as Region
  const parts: Factor[] = [
    unitFactor(
      reg.variables,
      reg.variables.map((v) => plan.card[v] ?? 1),
    ),
    plan.potential[r] as Factor,
  ]
  for (const e of plan.incoming[r] ?? [])
    parts.push(tableOf(g, plan, (g.edges[e] as { child: number }).child, messages[e] as number[]))
  return permuteFactor(factorProductAll(parts), reg.variables)
}

/** A factor normalised, or all NaN when it has no mass (a contradiction): the run then ends as `diverged` instead of throwing. */
function normalisedOrNaN(f: Factor): Factor {
  try {
    return normaliseFactor(f)
  } catch {
    return { ...f, logValues: new Float64Array(f.logValues.length).fill(Number.NaN) }
  }
}

/** The probabilities of a normalised log table. */
const probabilities = (f: Factor): number[] => Array.from(f.logValues, Math.exp)

function marginalsFlat(g: RegionGraph, plan: Plan, messages: readonly (readonly number[])[]): number[] {
  const cache = new Map<number, Factor>()
  const out: number[] = []
  for (const v of g.variables) {
    const r = plan.home.get(v.name) as number
    let b = cache.get(r)
    if (b === undefined) {
      b = beliefOf(g, plan, messages, r)
      cache.set(r, b)
    }
    out.push(...probabilities(normalisedOrNaN(factorMarginal(b, [v.name]))))
  }
  return out
}

/**
 * The region-based estimate of the log partition function of a state: `sum_R c_R (sum_{a inside R} E_{b_R}[log f_a] + H(b_R))` over
 * the beliefs of the regions (the negative of the Kikuchi free energy; the Bethe approximation on the Bethe region graph,
 * the exact value on a junction tree's regions at its fixed point).
 *
 * @param g - The region graph.
 * @param state - A state of a run.
 * @returns The estimate, or NaN when a belief has no mass.
 * @concept region-graphs
 * @index function The Bethe/Kikuchi estimate of log Z from the beliefs of a region graph run.
 * @example
 * ```ts
 * import { regionBpAlgorithm, regionEvidence, regionGraph } from 'omai-engine/message-passing'
 * import { factor, factorGraph } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }, { name: 'b', states: ['0', '1'] }], [factor(['a', 'b'], [2, 2], [3, 1, 1, 3])])
 * const g = regionGraph(fg, { kind: 'bethe' })
 * const r = runAlgorithm(regionBpAlgorithm(g), { maxSteps: 50 })
 * Math.exp(regionEvidence(g, r.state)) // => 8
 * ```
 */
export function regionEvidence(g: RegionGraph, state: RegionState): number {
  const plan = planOf(g)
  return evidenceFor(g, plan, state.messages)
}

function evidenceFor(g: RegionGraph, plan: Plan, messages: readonly (readonly number[])[]): number {
  let total = 0
  const beliefs = g.regions.map((_, r) => {
    try {
      return normaliseFactor(beliefOf(g, plan, messages, r))
    } catch {
      return undefined
    }
  })
  for (const [r, reg] of g.regions.entries()) {
    const b = beliefs[r]
    if (b === undefined) return Number.NaN
    if (reg.count === 0) continue
    total += reg.count * factorEntropy(b)
    for (const fi of reg.factors) {
      const f = g.factors[fi] as Factor
      const p = probabilities(factorMarginal(b, f.variables))
      const lf = f.logValues
      for (const [i, pi] of p.entries()) if (pi > 0) total += reg.count * pi * (lf[i] ?? 0)
    }
  }
  return total
}

/**
 * An anytime snapshot of a region-graph state: where it is and the marginals by variable name (`{ family: 'discrete',
 * probs }`), in the shape of the engine's `MpSnapshot`. Pass it to `runAnytimeWith`.
 *
 * @param g - The region graph.
 * @returns A function of a state alone.
 * @concept region-graphs
 * @example
 * ```ts
 * import { regionBpAlgorithm, regionGraph, regionSnapshot } from 'omai-engine/message-passing'
 * import { factor, factorGraph } from 'omai-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }], [factor(['a'], [2], [1, 3])])
 * const g = regionGraph(fg, { kind: 'bethe' })
 * regionSnapshot(g)(regionBpAlgorithm(g).init({ kind: 'key', words: [0, 0] })).sweeps // => 0
 * ```
 */
export function regionSnapshot(g: RegionGraph): (state: RegionState) => MpSnapshot {
  const plan = planOf(g)
  return (state) => ({
    sweeps: state.sweeps,
    updates: state.updates,
    change: state.change,
    improper: { events: 0, skipped: 0, damped: 0, reset: 0 },
    marginals: marginalsOf(g, plan, state),
  })
}

/**
 * The normalised belief of every region under a state, as a log table over the region's variables (a clique's table on
 * a junction tree's regions at the fixed point is its exact marginal).
 *
 * @param g - The region graph.
 * @param state - A state of a run.
 * @returns One factor per region, in the region graph's order.
 * @throws OmaiError with code `value` when a belief has no mass (a contradiction in the model).
 * @concept region-graphs
 * @example
 * ```ts
 * import { regionBeliefs, regionBpAlgorithm, regionGraph } from 'omai-engine/message-passing'
 * import { factor, factorGraph, factorValues } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }, { name: 'b', states: ['0', '1'] }], [factor(['a', 'b'], [2, 2], [3, 1, 1, 3])])
 * const g = regionGraph(fg, { kind: 'junction' })
 * factorValues(regionBeliefs(g, runAlgorithm(regionBpAlgorithm(g), { maxSteps: 10 }).state)[0] as never) // => [0.375, 0.125, 0.125, 0.375]
 * ```
 */
export function regionBeliefs(g: RegionGraph, state: RegionState): Factor[] {
  const plan = planOf(g)
  return g.regions.map((_, r) => normaliseFactor(beliefOf(g, plan, state.messages, r)))
}

function marginalsOf(g: RegionGraph, plan: Plan, state: RegionState): Record<string, Record<string, Json>> {
  const out: Record<string, Record<string, Json>> = {}
  let offset = 0
  const flat = marginalsFlat(g, plan, state.messages)
  for (const v of g.variables) {
    out[v.name] = { family: 'discrete', probs: flat.slice(offset, offset + v.states.length) }
    offset += v.states.length
  }
  return out
}

/**
 * Generalised belief propagation (Yedidia, Freeman and Weiss 2005) on a region graph of a discrete model: the
 * parent-to-child algorithm, in which every region keeps a belief, the product of its factors and the messages that enter it
 * (its parents' messages to it, and the messages into the regions it contains from regions outside it), and the message from
 * a parent P to a child R is set so that the marginal of P's belief on R equals R's belief. On the Bethe region graph this is
 * the sum-product update of loopy belief propagation (it reproduces `bpAlgorithm`'s fixed points); on the regions of a
 * junction tree it is exact; on Kikuchi clusters it is the cluster variation method, which on a grid beats the Bethe
 * approximation. The tape carries the change, the sweeps and updates, the region-based log partition function estimate,
 * the messages of the last step (from `region:<i>` to `region:<j>`) and the marginals. Messages are normalised log tables, so
 * they never become improper; a model whose beliefs lose all their mass (a contradiction) ends the run as `diverged`. The state
 * is plain numbers: a run serialises, resumes to the same state, and has anytime snapshots through `regionSnapshot`.
 *
 * @param g - The region graph.
 * @param options - Schedule, granularity, damping, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link regionContract}.
 * @throws OmaiError with code `value` for a damping outside [0, 1).
 * @concept region-graphs
 * @reference Yedidia, J. S., Freeman, W. T. and Weiss, Y. (2005). Constructing free-energy approximations and generalized belief propagation algorithms. IEEE Transactions on Information Theory 51(7).
 * @reference Kikuchi, R. (1951). A theory of cooperative phenomena. Physical Review 81(6).
 * @index algorithm Generalised belief propagation on a region graph: Bethe, Kikuchi or junction regions.
 * @example
 * ```ts
 * import { regionBpAlgorithm, regionGraph, regionSnapshot } from 'omai-engine/message-passing'
 * import { factor, factorGraph } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const states = ['0', '1']
 * const fg = factorGraph(
 *   [{ name: 'a', states }, { name: 'b', states }, { name: 'c', states }],
 *   [factor(['a'], [2], [0.3, 0.7]), factor(['a', 'b'], [2, 2], [0.9, 0.1, 0.2, 0.8]), factor(['b', 'c'], [2, 2], [0.7, 0.3, 0.4, 0.6])],
 * )
 * const g = regionGraph(fg, { kind: 'bethe' })
 * const r = runAlgorithm(regionBpAlgorithm(g), { maxSteps: 100 })
 * regionSnapshot(g)(r.state).marginals.c?.probs.map((p) => p.toFixed(4)) // => ['0.5230', '0.4770']
 * ```
 */
export function regionBpAlgorithm(g: RegionGraph, options: RegionBpOptions = {}): Algorithm<RegionState, RegionPath> {
  const o = { ...regionBpDefaults, ...options }
  assertDamping(o.damping, 'region BP')
  const plan = planOf(g)
  const normalised = (values: ArrayLike<number>): number[] => {
    const lse = shiftedSum(values).logTotal
    return Array.from({ length: values.length }, (_, i) => (values[i] ?? 0) - lse)
  }
  /** The new message of an edge from the beliefs of its two regions under `source`. */
  const proposal = (source: readonly (readonly number[])[], e: number): number[] => {
    const edge = g.edges[e] as { parent: number; child: number }
    const child = g.regions[edge.child] as Region
    const bp = normalisedOrNaN(beliefOf(g, plan, source, edge.parent))
    const bc = normalisedOrNaN(beliefOf(g, plan, source, edge.child))
    const marg = factorMarginal(bp, child.variables)
    const old = source[e] as number[]
    const next = old.map((m, i) => {
      const num = marg.logValues[i] ?? Number.NEGATIVE_INFINITY
      const den = bc.logValues[i] ?? Number.NEGATIVE_INFINITY
      const ratio = den === Number.NEGATIVE_INFINITY || num === Number.NEGATIVE_INFINITY ? 0 : num - den
      return m + (1 - o.damping) * ratio
    })
    return normalised(next)
  }
  const edgeCount = g.edges.length
  const group = o.schedule === 'serial' && o.granularity === 'edge' ? 1 : edgeCount
  return {
    name: 'region-bp',
    config: {
      schedule: o.schedule,
      granularity: o.granularity,
      damping: o.damping,
      tol: o.tol,
      maxSweeps: o.maxSweeps,
      regions: g.kind,
    } as Json,
    schema: channels({
      metrics: {
        change: { description: 'the largest change of a marginal over the last completed sweep' },
        sweeps: { description: 'completed sweeps' },
        updates: { description: 'messages sent so far' },
        evidence: { description: 'the region-based estimate of the log partition function', direction: 'higher' },
      },
      messages: {
        message: { description: 'a message from a region to a region it contains' },
        marginals: { description: 'the current marginal of every variable, by name' },
      },
      frames: true,
    }),
    init: (): RegionState => {
      const messages = g.edges.map((e) => {
        const size = (g.regions[e.child] as Region).variables.reduce((a, v) => a * (plan.card[v] ?? 1), 1)
        return new Array<number>(size).fill(-Math.log(size))
      })
      return {
        messages,
        start: marginalsFlat(g, plan, messages),
        sweeps: 0,
        cursor: 0,
        updates: 0,
        change: Number.POSITIVE_INFINITY,
        sent: [],
      }
    },
    step: (state): RegionState => {
      const messages = state.messages.slice()
      const from = state.cursor
      const to = Math.min(edgeCount, from + group)
      const frozen = o.schedule === 'parallel' ? state.messages : undefined
      for (let e = from; e < to; e++) messages[e] = proposal(frozen ?? messages, e)
      const updates = state.updates + (to - from)
      const sent = Array.from({ length: to - from }, (_, k) => from + k)
      if (to < edgeCount) return { ...state, messages, cursor: to, updates, sent }
      const now = marginalsFlat(g, plan, messages)
      let change = 0
      for (let i = 0; i < now.length; i++) change = Math.max(change, Math.abs((now[i] ?? 0) - (state.start[i] ?? 0)))
      return { messages, start: now, sweeps: state.sweeps + 1, cursor: 0, updates, change, sent }
    },
    done: (state): RunStatus | undefined => {
      if (state.messages.some((m) => m.some((v) => Number.isNaN(v)))) return 'diverged'
      if (state.sweeps >= 1 && state.cursor === 0 && state.change < o.tol) return 'converged'
      return state.sweeps >= o.maxSweeps ? 'completed' : undefined
    },
    observe: (state) => {
      const marginals = marginalsOf(g, plan, state)
      const evidence = evidenceFor(g, plan, state.messages)
      return {
        metrics: {
          sweeps: state.sweeps,
          updates: state.updates,
          ...(Number.isFinite(state.change) ? { change: state.change } : {}),
          ...(Number.isFinite(evidence) ? { evidence } : {}),
        },
        messages: [
          ...state.sent.map((e) => {
            const edge = g.edges[e] as { parent: number; child: number }
            return {
              kind: 'message',
              from: `region:${edge.parent}`,
              to: `region:${edge.child}`,
              payload: { parent: edge.parent, child: edge.child, edge: e } as Json,
            }
          }),
          { kind: 'marginals', payload: marginals as Json },
        ],
        frame: { marginals: marginals as Json },
      }
    },
    serialise: (state) => stateToJson(state),
    deserialise: (json) => stateFromJson(json) as RegionState,
  }
}

/**
 * Region-graph belief propagation on a model as a method: the choice of regions (`bethe`, `kikuchi` on the clusters given
 * here, or `junction`) and the knobs (schedule, granularity, damping, tolerance, sweep cap) as a space, the tape as the
 * typed contract {@link regionContract}.
 *
 * @param model - The discrete model.
 * @param clusters - The clusters of the `kikuchi` choice (for example `blockClusters` of a grid); without them `kikuchi` is refused.
 * @returns The method, named `region-bp`.
 * @concept region-graphs
 * @index function The generalised BP method of a model: regions (Bethe, Kikuchi, junction) as a knob, typed tape.
 * @example
 * ```ts
 * import { regionBpMethod } from 'omai-engine/message-passing'
 * import { factor, factorGraph } from 'omai-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }, { name: 'b', states: ['0', '1'] }], [factor(['a', 'b'], [2, 2], [3, 1, 1, 3])])
 * regionBpMethod(fg).name // => 'region-bp'
 * ```
 */
export function regionBpMethod(
  model: DiscreteModel,
  clusters?: readonly (readonly string[])[],
): Method<RegionState, Config, RegionPath> {
  return defineMethod<RegionState, Config, RegionPath>({
    name: 'region-bp',
    config: dict({
      regions: choice(['bethe', 'kikuchi', 'junction'] as const, {
        default: 'bethe',
        description: 'the regions: Bethe (BP), Kikuchi clusters, or a junction tree (exact)',
      }),
      schedule: choice(['parallel', 'serial'] as const, {
        default: regionBpDefaults.schedule,
        description: 'every message from the state at the start of the sweep, or edge by edge',
      }),
      granularity: choice(['sweep', 'edge'] as const, {
        default: regionBpDefaults.granularity,
        description: 'for a serial schedule: how much one step does',
      }),
      damping: real(0, 0.95, {
        default: regionBpDefaults.damping,
        description: 'weight of the old message in the new (in the log)',
      }),
      tol: real(1e-12, 1e-2, {
        log: true,
        default: regionBpDefaults.tol,
        description: 'stop when no marginal moves more than this over a sweep',
      }),
      maxSweeps: int(1, 10_000, { default: regionBpDefaults.maxSweeps, description: 'the sweep cap' }),
    }),
    make: (c) => {
      const kind = typeof c['regions'] === 'string' ? c['regions'] : 'bethe'
      let graph: RegionGraph
      if (kind === 'kikuchi') {
        if (clusters === undefined)
          throw new ParameterError('region BP: the kikuchi regions need clusters (see blockClusters)')
        graph = regionGraph(model, { kind: 'clusters', clusters })
      } else graph = regionGraph(model, { kind: kind === 'junction' ? 'junction' : 'bethe' })
      const schedule = c['schedule'] === 'serial' ? 'serial' : 'parallel'
      return regionBpAlgorithm(graph, {
        schedule,
        granularity: c['granularity'] === 'edge' ? 'edge' : 'sweep',
        ...(typeof c['damping'] === 'number' ? { damping: c['damping'] } : {}),
        ...(typeof c['tol'] === 'number' ? { tol: c['tol'] } : {}),
        ...(typeof c['maxSweeps'] === 'number' ? { maxSweeps: c['maxSweeps'] } : {}),
      })
    },
    contract: regionContract,
  })
}
