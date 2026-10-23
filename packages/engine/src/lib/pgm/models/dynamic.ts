import { WhiteboxError } from '../../base/index.ts'
import { type Factor, factorProduct, logFactor, permuteFactor } from '../factor.ts'
import { topologicalOrder } from '../graphs.ts'
import { calibrate, junctionTreeQuery } from '../inference/junction-tree.ts'
import { type BayesNet, type BayesNode, bayesNet } from './bayes-net.ts'
import { evidenceIndices, factorGraph, modelFactors } from './factor-graph.ts'

/** A node of the transition template of a dynamic Bayes network. */
export interface TemplateNode {
  readonly name: string
  readonly states: readonly string[]
  /** Parents in the same slice. */
  readonly parents: readonly string[]
  /** Parents in the previous slice. */
  readonly previous: readonly string[]
  /** P(node | previous, parents) in row-major order over [...previous, ...parents, node]. */
  readonly cpt: readonly number[]
}

/**
 * A dynamic Bayes network as a two-slice template (2TBN): a Bayes network for the first slice and a transition model
 * in which each node may depend on nodes of its own slice and of the previous one (pgmpy's DynamicBayesianNetwork).
 *
 * @concept dynamic-bayes-networks
 */
export interface DynamicBayesNet {
  readonly kind: 'dynamicBayesNet'
  readonly initial: BayesNet
  readonly transition: readonly TemplateNode[]
}

/** The slice-t name of a template variable. */
const at = (name: string, t: number): string => `${name}_${t}`
/** The name of a previous-slice variable inside one transition step. */
const prev = (name: string): string => `${name}@prev`

/**
 * Build and validate a dynamic Bayes network: both slices must have the same variables and states, intra-slice parents
 * must be acyclic, and every CPT row must be a distribution.
 *
 * @param initial - The nodes of the first slice.
 * @param transition - The transition template.
 * @returns The network.
 * @throws WhiteboxError with code `value` for mismatched slices, unknown parents, cycles or invalid CPTs.
 * @concept dynamic-bayes-networks
 * @example
 * ```ts
 * import { dynamicBayesNet } from 'whitebox-engine/pgm'
 * const dbn = dynamicBayesNet(
 *   [{ name: 'z', states: ['a', 'b'], parents: [], cpt: [0.5, 0.5] }],
 *   [{ name: 'z', states: ['a', 'b'], parents: [], previous: ['z'], cpt: [0.9, 0.1, 0.2, 0.8] }],
 * )
 * dbn.transition.length // => 1
 * ```
 */
export function dynamicBayesNet(initial: readonly BayesNode[], transition: readonly TemplateNode[]): DynamicBayesNet {
  const first = bayesNet(initial)
  const states = new Map(first.nodes.map((n) => [n.name, n.states]))
  if (transition.length !== first.nodes.length || new Set(transition.map((n) => n.name)).size !== transition.length)
    throw new WhiteboxError('value', 'dynamicBayesNet: both slices need the same variables')
  for (const n of transition) {
    const s = states.get(n.name)
    if (s === undefined || s.length !== n.states.length || s.some((x, i) => x !== n.states[i]))
      throw new WhiteboxError('value', `dynamicBayesNet: ${n.name} differs between the slices`)
    for (const p of [...n.parents, ...n.previous])
      if (!states.has(p)) throw new WhiteboxError('value', `dynamicBayesNet: ${n.name} has unknown parent ${p}`)
  }
  let ordered: TemplateNode[]
  try {
    ordered = topologicalOrder(transition)
  } catch (e) {
    throw new WhiteboxError('value', `dynamicBayesNet: ${(e as Error).message}`)
  }
  const dbn: DynamicBayesNet = { kind: 'dynamicBayesNet', initial: first, transition: ordered }
  unroll(dbn, 2) // validates the transition CPTs
  return dbn
}

/**
 * Unroll a dynamic Bayes network over T slices into a Bayes network whose variables are named `<name>_<t>`.
 *
 * @param dbn - The network.
 * @param T - Number of slices (at least 1).
 * @returns The unrolled Bayes network.
 * @throws WhiteboxError with code `value` for T < 1.
 * @concept dynamic-bayes-networks
 * @example
 * ```ts
 * import { dynamicBayesNet, unroll } from 'whitebox-engine/pgm'
 * const dbn = dynamicBayesNet(
 *   [{ name: 'z', states: ['a', 'b'], parents: [], cpt: [0.5, 0.5] }],
 *   [{ name: 'z', states: ['a', 'b'], parents: [], previous: ['z'], cpt: [0.9, 0.1, 0.2, 0.8] }],
 * )
 * unroll(dbn, 3).nodes.map((n) => n.name) // => ['z_0', 'z_1', 'z_2']
 * ```
 */
export function unroll(dbn: DynamicBayesNet, T: number): BayesNet {
  if (!(T >= 1) || !Number.isInteger(T)) throw new WhiteboxError('value', 'unroll: T must be a positive integer')
  const nodes: BayesNode[] = dbn.initial.nodes.map((n) => ({
    name: at(n.name, 0),
    states: n.states,
    parents: n.parents.map((p) => at(p, 0)),
    cpt: n.cpt,
  }))
  for (let t = 1; t < T; t++) {
    for (const n of dbn.transition) {
      nodes.push({
        name: at(n.name, t),
        states: n.states,
        parents: [...n.previous.map((p) => at(p, t - 1)), ...n.parents.map((p) => at(p, t))],
        cpt: n.cpt,
      })
    }
  }
  return bayesNet(nodes)
}

/**
 * The interface of a dynamic Bayes network: the variables of a slice with a child in the next slice, which separate the
 * past from the future (Murphy 2002).
 *
 * @param dbn - The network.
 * @returns The interface variables (template names, in template order).
 * @concept dynamic-bayes-networks
 * @example
 * ```ts
 * import { dynamicBayesNet, interfaceVariables } from 'whitebox-engine/pgm'
 * const dbn = dynamicBayesNet(
 *   [{ name: 'z', states: ['a', 'b'], parents: [], cpt: [0.5, 0.5] }, { name: 'x', states: ['u', 'v'], parents: ['z'], cpt: [0.7, 0.3, 0.1, 0.9] }],
 *   [{ name: 'z', states: ['a', 'b'], parents: [], previous: ['z'], cpt: [0.9, 0.1, 0.2, 0.8] }, { name: 'x', states: ['u', 'v'], parents: ['z'], previous: [], cpt: [0.7, 0.3, 0.1, 0.9] }],
 * )
 * interfaceVariables(dbn) // => ['z']
 * ```
 */
export function interfaceVariables(dbn: DynamicBayesNet): string[] {
  const used = new Set(dbn.transition.flatMap((n) => n.previous))
  return dbn.transition.map((n) => n.name).filter((v) => used.has(v))
}

/** Result of {@link dbnFilter}. */
export interface DbnFilterResult {
  /** The filtered marginal p(v_t | evidence of slices 0..t) of every variable of every slice, by template name. */
  readonly marginals: Record<string, Factor>[]
  /** log p(evidence of all slices). */
  readonly logLikelihood: number
}

/** A point-mass factor. */
function pointMass(name: string, states: readonly string[], index: number): Factor {
  return logFactor(
    [name],
    [states.length],
    states.map((_, i) => (i === index ? 0 : Number.NEGATIVE_INFINITY)),
  )
}

/**
 * Exact filtering in a dynamic Bayes network by the interface algorithm (Murphy 2002): carry the joint belief over the
 * interface variables from slice to slice, each step calibrating a junction tree of the one-and-a-half-slice network
 * (the belief over the previous interface times the transition CPTs, reduced by the slice's evidence).
 *
 * @param dbn - The network.
 * @param evidence - Observed states (names or indices) for each slice, by template name.
 * @returns The filtered marginals of every slice and the log-likelihood of the evidence.
 * @throws WhiteboxError with code `value` for impossible evidence.
 * @concept dynamic-bayes-networks
 * @example
 * ```ts
 * import { dynamicBayesNet, dbnFilter, factorValues } from 'whitebox-engine/pgm'
 * const dbn = dynamicBayesNet(
 *   [{ name: 'z', states: ['a', 'b'], parents: [], cpt: [0.5, 0.5] }, { name: 'x', states: ['u', 'v'], parents: ['z'], cpt: [0.8, 0.2, 0.2, 0.8] }],
 *   [{ name: 'z', states: ['a', 'b'], parents: [], previous: ['z'], cpt: [1, 0, 0, 1] }, { name: 'x', states: ['u', 'v'], parents: ['z'], previous: [], cpt: [0.8, 0.2, 0.2, 0.8] }],
 * )
 * dbnFilter(dbn, [{ x: 'u' }, { x: 'u' }]).marginals.map((m) => Object.entries(m).filter(([v]) => v === 'z').map(([, f]) => Math.round((factorValues(f)[0] ?? 0) * 1000) / 1000)) // => [[0.8], [0.941]]
 * ```
 */
export function dbnFilter(
  dbn: DynamicBayesNet,
  evidence: readonly Readonly<Record<string, string | number>>[],
): DbnFilterResult {
  const iface = interfaceVariables(dbn)
  const states = new Map(dbn.initial.nodes.map((n) => [n.name, n.states]))
  const names = dbn.transition.map((n) => n.name)
  const sliceVariables = names.map((name) => ({ name, states: states.get(name) ?? [] }))
  const transitionFactors = modelFactors(
    bayesNet([
      ...[...new Set(dbn.transition.flatMap((n) => n.previous))].map((v) => {
        const k = states.get(v)?.length ?? 1
        return { name: prev(v), states: states.get(v) ?? [], parents: [], cpt: new Array<number>(k).fill(1 / k) }
      }),
      ...dbn.transition.map((n) => ({
        name: n.name,
        states: n.states,
        parents: [...n.previous.map(prev), ...n.parents],
        cpt: n.cpt,
      })),
    ]),
  ).filter((f) => !(f.variables.length === 1 && (f.variables[0] ?? '').endsWith('@prev')))
  const marginals: Record<string, Factor>[] = []
  let logLikelihood = 0
  let belief: Factor | null = null
  for (const [t, ev] of evidence.entries()) {
    const variables =
      t === 0
        ? sliceVariables
        : [...sliceVariables, ...iface.map((v) => ({ name: prev(v), states: states.get(v) ?? [] }))]
    const factors = t === 0 ? modelFactors(dbn.initial) : [...transitionFactors, ...(belief === null ? [] : [belief])]
    const fg = factorGraph(variables, factors)
    const observed = evidenceIndices(fg, ev)
    const tree = calibrate(fg, { evidence: observed })
    logLikelihood += tree.logPartition
    const slice: Record<string, Factor> = {}
    for (const v of names) {
      const s = observed[v]
      slice[v] = s === undefined ? junctionTreeQuery(tree, [v]) : pointMass(v, states.get(v) ?? [], s)
    }
    marginals.push(slice)
    const free = iface.filter((v) => !(v in observed))
    let next: Factor | null = free.length > 0 ? junctionTreeQuery(tree, free) : null
    for (const v of iface) {
      const s = observed[v]
      if (s === undefined) continue
      const point = pointMass(v, states.get(v) ?? [], s)
      next = next === null ? point : factorProduct(next, point)
    }
    belief = next === null ? null : renamed(permuteFactor(next, iface), prev)
  }
  if (!Number.isFinite(logLikelihood)) throw new WhiteboxError('value', 'dbnFilter: the evidence is impossible')
  return { marginals, logLikelihood }
}

function renamed(f: Factor, rename: (v: string) => string): Factor {
  return { ...f, variables: f.variables.map(rename) }
}
