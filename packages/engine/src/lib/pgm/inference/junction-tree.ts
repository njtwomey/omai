/**
 * Junction-tree propagation, exact inference by passing messages between cliques of a triangulated graph: collect
 * towards a root then distribute back, so clique and separator beliefs agree. A traced algorithm, calibration, queries
 * over any set of variables, every marginal at once, and the most probable explanation by max-product propagation.
 */

import type { Json } from '../../base/index.ts'
import { MdmlError } from '../../base/index.ts'
import { type Algorithm, channels, runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
import {
  type Factor,
  factorArgmax,
  factorDivide,
  factorMarginal,
  factorProductAll,
  factorValues,
  logFactor,
  logTotal,
  normaliseFactor,
  reduceFactor,
  unitFactor,
} from '../factor.ts'
import { interactionGraph } from '../graphs.ts'
import {
  type DiscreteModel,
  evidenceIndices,
  modelFactors,
  modelVariables,
  stateNames,
} from '../models/factor-graph.ts'
import { type EliminationHeuristic, eliminationOrder, type JunctionTree, junctionTree } from '../triangulation.ts'

/** Options of {@link junctionTreeAlgorithm} and {@link calibrate}. */
export interface JunctionTreeOptions {
  /** Observed states by variable (names or indices). */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The triangulation order (default: greedy by `heuristic` on the interaction graph of the reduced factors). */
  readonly order?: readonly string[]
  /** The greedy ordering criterion (default `minFill`). */
  readonly heuristic?: EliminationHeuristic
  /** `shaferShenoy` (messages kept on edges, default) or `hugin` (clique and separator potentials updated by division). */
  readonly method?: 'shaferShenoy' | 'hugin'
  /** Sum-product (marginals) or max-product (max-marginals, for MAP). */
  readonly mode?: 'sum' | 'max'
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** A calibrated junction tree: clique and separator beliefs that agree on their shared variables. */
export interface CalibratedTree {
  readonly tree: JunctionTree
  /** The unnormalised belief of each clique (proportional to the posterior over the clique given the evidence). */
  readonly cliqueBeliefs: readonly Factor[]
  /** The belief of each tree edge's separator. */
  readonly separatorBeliefs: readonly Factor[]
  /** log Z of the reduced model (log P(evidence) for a Bayes network); under max-product, the log of the MAP value. */
  readonly logPartition: number
  /** The evidence, as state indices. */
  readonly evidence: Readonly<Record<string, number>>
  readonly mode: 'sum' | 'max'
}

/** The state of a junction-tree run: the schedule position and the current potentials and messages. */
export interface JunctionTreeState {
  /** Messages sent so far. */
  readonly sent: number
  /** Shafer-Shenoy: the message along each directed edge `i->j`. */
  readonly messages: ReadonlyMap<string, Factor>
  /** HUGIN: the current clique potentials. */
  readonly cliques: readonly Factor[]
  /** HUGIN: the current separator potentials. */
  readonly separators: readonly Factor[]
  /** The last message sent, for the trace. */
  readonly last?: { readonly from: number; readonly to: number; readonly phase: 'collect' | 'distribute' }
}

interface Setup {
  readonly tree: JunctionTree
  readonly potentials: Factor[]
  readonly constant: number
  readonly evidence: Record<string, number>
  /** Directed edges in schedule order: collect (towards clique 0) then distribute. */
  readonly schedule: { from: number; to: number; edge: number; phase: 'collect' | 'distribute' }[]
  readonly neighbours: number[][]
  readonly card: Readonly<Record<string, number>>
}

/** Build the junction tree of the reduced factors, assign each factor to a clique, and fix the message schedule. */
function setup(model: DiscreteModel, options: JunctionTreeOptions): Setup {
  const evidence = evidenceIndices(model, options.evidence ?? {})
  const variables = modelVariables(model).filter((v) => !(v.name in evidence))
  const card = Object.fromEntries(variables.map((v) => [v.name, v.states.length]))
  const reduced = modelFactors(model).map((f) => reduceFactor(f, evidence))
  const constant = reduced.filter((f) => f.variables.length === 0).reduce((s, f) => s + (f.logValues[0] ?? 0), 0)
  const scoped = reduced.filter((f) => f.variables.length > 0)
  const names = variables.map((v) => v.name)
  const graph = interactionGraph(scoped, names)
  const order =
    options.order?.filter((v) => !(v in evidence)) ??
    eliminationOrder(graph, names, options.heuristic ?? 'minFill', card)
  const tree = junctionTree(graph, order)
  const potentials = tree.cliques.map((c) =>
    unitFactor(
      c,
      c.map((v) => card[v] ?? 1),
    ),
  )
  for (const f of scoped) {
    const home = tree.cliques.findIndex((c) => f.variables.every((v) => c.includes(v)))
    if (home < 0) throw new MdmlError('value', `junction tree: no clique holds [${f.variables.join(', ')}]`)
    potentials[home] = factorProductAll([potentials[home] as Factor, f])
  }
  const neighbours = tree.cliques.map(() => [] as number[])
  for (const e of tree.edges) {
    neighbours[e.from]?.push(e.to)
    neighbours[e.to]?.push(e.from)
  }
  const edgeOf = (a: number, b: number): number =>
    tree.edges.findIndex((e) => (e.from === a && e.to === b) || (e.from === b && e.to === a))
  // Depth-first from clique 0: collect in post-order (children before parents), distribute in pre-order.
  const collect: Setup['schedule'] = []
  const distribute: Setup['schedule'] = []
  const visit = (i: number, parent: number): void => {
    for (const j of neighbours[i] ?? []) {
      if (j === parent) continue
      distribute.push({ from: i, to: j, edge: edgeOf(i, j), phase: 'distribute' })
      visit(j, i)
      collect.push({ from: j, to: i, edge: edgeOf(i, j), phase: 'collect' })
    }
  }
  if (tree.cliques.length > 0) visit(0, -1)
  return { tree, potentials, constant, evidence, schedule: [...collect, ...distribute], neighbours, card }
}

const project = (f: Factor, keep: readonly string[], mode: 'sum' | 'max'): Factor => factorMarginal(f, keep, mode)

/**
 * Junction-tree propagation as an {@link Algorithm}: one step sends one message along the tree, first towards the root
 * (collect) and then back out (distribute), so after 2(n - 1) steps the tree is calibrated. Shafer-Shenoy keeps the
 * messages on the edges; HUGIN keeps clique and separator potentials and updates them by division. Each step records a
 * `collect` or `distribute` message (the cliques, the separator and its values).
 *
 * @param model - The model.
 * @param options - Evidence, ordering, method and mode.
 * @returns The algorithm.
 * @concept junction-trees
 * @example
 * ```ts
 * import { bayesNet, junctionTreeAlgorithm } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const bn = bayesNet([
 *   { name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
 *   { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] },
 *   { name: 'c', states: ['0', '1'], parents: ['b'], cpt: [0.7, 0.3, 0.4, 0.6] },
 * ])
 * runAlgorithm(junctionTreeAlgorithm(bn), { maxSteps: 100 }).steps // => 2
 * ```
 */
export function junctionTreeAlgorithm(
  model: DiscreteModel,
  options: JunctionTreeOptions = {},
): Algorithm<JunctionTreeState> {
  const s = setup(model, options)
  const mode = options.mode ?? 'sum'
  const hugin = options.method === 'hugin'
  return {
    name: 'junctionTree',
    config: { method: hugin ? 'hugin' : 'shaferShenoy', mode, cliques: s.tree.cliques.map((c) => [...c]) } as Json,
    schema: channels({
      messages: {
        collect: { description: 'a message towards the root clique' },
        distribute: { description: 'a message away from the root clique' },
      },
    }),
    init: () => ({
      sent: 0,
      messages: new Map(),
      cliques: s.potentials,
      separators: s.tree.edges.map((e) =>
        unitFactor(
          e.separator,
          e.separator.map((v) => s.card[v] ?? 1),
        ),
      ),
    }),
    step: (state) => {
      const msg = s.schedule[state.sent]
      if (msg === undefined) return state
      const separator = s.tree.edges[msg.edge]?.separator ?? []
      if (hugin) {
        const cliques = [...state.cliques]
        const separators = [...state.separators]
        const projected = project(cliques[msg.from] as Factor, separator, mode)
        cliques[msg.to] = factorProductAll([
          cliques[msg.to] as Factor,
          factorDivide(projected, separators[msg.edge] as Factor),
        ])
        separators[msg.edge] = projected
        return { ...state, sent: state.sent + 1, cliques, separators, last: msg }
      }
      const incoming = (s.neighbours[msg.from] ?? [])
        .filter((k) => k !== msg.to)
        .map((k) => state.messages.get(`${k}->${msg.from}`))
        .filter((m): m is Factor => m !== undefined)
      const message = project(factorProductAll([s.potentials[msg.from] as Factor, ...incoming]), separator, mode)
      const messages = new Map(state.messages)
      messages.set(`${msg.from}->${msg.to}`, message)
      return { ...state, sent: state.sent + 1, messages, last: msg }
    },
    done: (state) => (state.sent >= s.schedule.length ? 'converged' : undefined),
    observe: (state) => {
      const m = state.last
      if (m === undefined) return {}
      const sep = s.tree.edges.find((_, i) => i === s.schedule[state.sent - 1]?.edge)?.separator ?? []
      const values = hugin
        ? (state.separators[s.schedule[state.sent - 1]?.edge ?? 0] ?? unitFactor([], []))
        : (state.messages.get(`${m.from}->${m.to}`) ?? unitFactor([], []))
      return {
        messages: [
          {
            kind: m.phase,
            payload: { from: m.from, to: m.to, separator: [...sep], values: factorValues(values) },
            from: `C${m.from}`,
            to: `C${m.to}`,
          },
        ],
      }
    },
  }
}

/** The calibrated beliefs from a finished junction-tree state. */
function beliefs(s: Setup, state: JunctionTreeState, hugin: boolean, mode: 'sum' | 'max'): CalibratedTree {
  const cliqueBeliefs = hugin
    ? [...state.cliques]
    : s.tree.cliques.map((_, i) =>
        factorProductAll([
          s.potentials[i] as Factor,
          ...(s.neighbours[i] ?? [])
            .map((k) => state.messages.get(`${k}->${i}`))
            .filter((m): m is Factor => m !== undefined),
        ]),
      )
  const separatorBeliefs = hugin
    ? [...state.separators]
    : s.tree.edges.map((e) => project(cliqueBeliefs[e.from] as Factor, e.separator, mode))
  const root = cliqueBeliefs[0]
  const rootTotal =
    root === undefined ? 0 : mode === 'sum' ? logTotal(root) : (factorMarginal(root, [], 'max').logValues[0] ?? 0)
  return {
    tree: s.tree,
    cliqueBeliefs,
    separatorBeliefs,
    logPartition: s.constant + rootTotal,
    evidence: s.evidence,
    mode,
  }
}

/**
 * Calibrate a junction tree (run {@link junctionTreeAlgorithm} to the end) and return the clique and separator
 * beliefs, from which every marginal follows.
 *
 * @param model - The model.
 * @param options - Evidence, ordering, method, mode and tracer.
 * @returns The calibrated tree.
 * @concept junction-trees
 * @example
 * ```ts
 * import { bayesNet, calibrate } from 'mdml-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
 *   { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * Math.exp(calibrate(bn, { evidence: { b: '1' } }).logPartition) // => 0.45
 * ```
 */
export function calibrate(model: DiscreteModel, options: JunctionTreeOptions = {}): CalibratedTree {
  const s = setup(model, options)
  const mode = options.mode ?? 'sum'
  const result = runAlgorithm(junctionTreeAlgorithm(model, options), {
    maxSteps: s.schedule.length + 1,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return beliefs(s, result.state, options.method === 'hugin', mode)
}

/** The tree path between two cliques (as edge indices). */
function pathEdges(tree: JunctionTree, a: number, b: number): number[] {
  const prev = new Map<number, [number, number]>([[a, [-1, -1]]])
  const queue = [a]
  while (queue.length > 0) {
    const i = queue.shift() as number
    if (i === b) break
    for (const [k, e] of tree.edges.entries()) {
      const j = e.from === i ? e.to : e.to === i ? e.from : -1
      if (j >= 0 && !prev.has(j)) {
        prev.set(j, [i, k])
        queue.push(j)
      }
    }
  }
  const out: number[] = []
  let cur = b
  while (cur !== a) {
    const [p, k] = prev.get(cur) ?? [a, -1]
    if (k < 0) break
    out.push(k)
    cur = p
  }
  return out
}

/**
 * The posterior over some variables from a calibrated tree: from one clique when one holds them all, otherwise from
 * the subtree joining cliques that cover them (the product of its clique beliefs over the product of its separator
 * beliefs). Observed variables cannot be queried.
 *
 * @param calibrated - A sum-calibrated tree.
 * @param variables - The query variables.
 * @returns The normalised posterior over them, in the order given.
 * @throws MdmlError with code `value` for observed or unknown variables, or a max-calibrated tree.
 * @concept junction-trees
 * @example
 * ```ts
 * import { bayesNet, calibrate, junctionTreeQuery, factorValues } from 'mdml-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
 *   { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * factorValues(junctionTreeQuery(calibrate(bn, { evidence: { b: '1' } }), ['a'])).map((p) => Math.round(p * 1e4) / 1e4) // => [0.1111, 0.8889]
 * ```
 */
export function junctionTreeQuery(calibrated: CalibratedTree, variables: readonly string[]): Factor {
  if (calibrated.mode !== 'sum') throw new MdmlError('value', 'junctionTreeQuery: needs a sum-calibrated tree')
  for (const v of variables) {
    if (v in calibrated.evidence) throw new MdmlError('value', `junctionTreeQuery: ${v} is observed`)
  }
  const { tree, cliqueBeliefs, separatorBeliefs } = calibrated
  const single = tree.cliques.findIndex((c) => variables.every((v) => c.includes(v)))
  if (single >= 0) return normaliseFactor(factorMarginal(cliqueBeliefs[single] as Factor, variables))
  const homes = variables.map((v) => {
    const i = tree.cliques.findIndex((c) => c.includes(v))
    if (i < 0) throw new MdmlError('value', `junctionTreeQuery: unknown variable ${v}`)
    return i
  })
  const edges = new Set<number>()
  for (const h of homes.slice(1)) for (const k of pathEdges(tree, homes[0] ?? 0, h)) edges.add(k)
  const cliques = new Set<number>(homes)
  for (const k of edges) {
    cliques.add(tree.edges[k]?.from ?? 0)
    cliques.add(tree.edges[k]?.to ?? 0)
  }
  let joint = factorProductAll([...cliques].map((i) => cliqueBeliefs[i] as Factor))
  for (const k of edges) joint = factorDivide(joint, separatorBeliefs[k] as Factor)
  return normaliseFactor(factorMarginal(joint, variables))
}

/**
 * The posterior marginal of every unobserved variable, by sum-product junction-tree propagation.
 *
 * @param model - The model.
 * @param options - Evidence, ordering and method.
 * @returns A normalised single-variable factor for each unobserved variable, in model order.
 * @concept junction-trees
 * @example
 * ```ts
 * import { bayesNet, junctionTreeMarginals, factorValues } from 'mdml-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.25, 0.75] }])
 * Object.values(junctionTreeMarginals(bn)).map(factorValues) // => [[0.25, 0.75]]
 * ```
 */
export function junctionTreeMarginals(
  model: DiscreteModel,
  options: Omit<JunctionTreeOptions, 'mode'> = {},
): Record<string, Factor> {
  const calibrated = calibrate(model, { ...options, mode: 'sum' })
  const out: Record<string, Factor> = {}
  for (const v of modelVariables(model)) {
    if (v.name in calibrated.evidence) continue
    const home = calibrated.tree.cliques.findIndex((c) => c.includes(v.name))
    out[v.name] =
      home < 0
        ? logFactor([v.name], [v.states.length], new Float64Array(v.states.length).fill(-Math.log(v.states.length)))
        : normaliseFactor(factorMarginal(calibrated.cliqueBeliefs[home] as Factor, [v.name]))
  }
  return out
}

/**
 * The most probable explanation by max-product junction-tree propagation: max-calibrate, take the best assignment of
 * the root clique, then extend it clique by clique outwards, each time maximising over the new variables with the
 * separator fixed (which yields a jointly consistent maximiser even with ties).
 *
 * @param model - The model.
 * @param options - Evidence, ordering and method.
 * @returns The assignment (state names) of every unobserved variable and the log of its unnormalised probability.
 * @concept map-inference
 * @example
 * ```ts
 * import { bayesNet, junctionTreeMap } from 'mdml-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'a', states: ['0', '1'], parents: [], cpt: [0.6, 0.4] },
 *   { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.5, 0.5, 0.1, 0.9] },
 * ])
 * junctionTreeMap(bn).assignment // => { a: '1', b: '1' }
 * ```
 */
export function junctionTreeMap(
  model: DiscreteModel,
  options: Omit<JunctionTreeOptions, 'mode'> = {},
): { readonly assignment: Record<string, string>; readonly logValue: number } {
  const calibrated = calibrate(model, { ...options, mode: 'max' })
  const { tree, cliqueBeliefs } = calibrated
  const indices: Record<string, number> = {}
  const done = new Set<number>()
  const visit = (i: number): void => {
    done.add(i)
    const belief = cliqueBeliefs[i] as Factor
    const fixed = Object.fromEntries(belief.variables.filter((v) => v in indices).map((v) => [v, indices[v] ?? 0]))
    Object.assign(indices, factorArgmax(reduceFactor(belief, fixed)))
    for (const e of tree.edges) {
      const j = e.from === i ? e.to : e.to === i ? e.from : -1
      if (j >= 0 && !done.has(j)) visit(j)
    }
  }
  for (let i = 0; i < tree.cliques.length; i++) if (!done.has(i)) visit(i)
  const names = modelVariables(model)
    .map((v) => v.name)
    .filter((v) => !(v in calibrated.evidence))
  for (const v of names) indices[v] ??= 0
  const ordered = Object.fromEntries(names.map((v) => [v, indices[v] ?? 0]))
  return { assignment: stateNames(model, ordered), logValue: calibrated.logPartition }
}
