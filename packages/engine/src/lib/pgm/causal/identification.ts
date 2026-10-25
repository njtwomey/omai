/**
 * Identifying causal effects from a causal graph, pgmpy style: the back-door and front-door criteria with every (or a
 * minimal) adjustment set, the adjustment formulas evaluated on a network's observational distribution,
 * `identifyEffect` for the simple cases of the ID algorithm, and checks for instrumental variables.
 */

import { MdmlError } from '../../base/index.ts'
import { type Factor, factor, factorValues } from '../factor.ts'
import { type DagNode, type DirectedModel, descendants } from '../graphs.ts'
import { variableElimination } from '../inference/elimination.ts'
import { type BayesNet, bayesNode, dSeparated, minimalDSeparator } from '../models/bayes-net.ts'

/** Options naming the unobserved variables of a causal graph (latent confounders cannot be adjusted for). */
export interface CausalGraphOptions {
  readonly latent?: readonly string[]
}

function nodeNames(g: DirectedModel): string[] {
  return g.nodes.map((n) => n.name)
}

function check(g: DirectedModel, ...names: string[]): void {
  const known = new Set(nodeNames(g))
  for (const v of names) if (!known.has(v)) throw new MdmlError('value', `unknown variable ${v}`)
}

/** The graph with every edge out of x removed (the back-door graph). */
function withoutOutgoing(g: DirectedModel, x: string): DirectedModel {
  return { nodes: g.nodes.map((n) => ({ name: n.name, parents: n.parents.filter((p) => p !== x) })) }
}

/** Whether a directed path from a to b exists. */
function directedPath(g: DirectedModel, a: string, b: string): boolean {
  return a !== b && descendants(g, a).includes(b)
}

/**
 * Whether a set satisfies Pearl's back-door criterion for the effect of x on y: no member is a descendant of x, and it
 * d-separates x from y once the edges out of x are removed (it blocks every path into x).
 *
 * @param g - The causal DAG.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param z - The candidate adjustment set.
 * @returns `true` when valid.
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { isBackdoorAdjustmentSet } from 'mdml-engine/pgm/causal'
 * const g = dag([['z', 'x'], ['z', 'y'], ['x', 'y']])
 * [isBackdoorAdjustmentSet(g, 'x', 'y', []), isBackdoorAdjustmentSet(g, 'x', 'y', ['z'])] // => [false, true]
 * ```
 */
export function isBackdoorAdjustmentSet(g: DirectedModel, x: string, y: string, z: readonly string[]): boolean {
  check(g, x, y, ...z)
  const desc = new Set(descendants(g, x))
  if (z.some((v) => desc.has(v) || v === y)) return false
  return dSeparated(withoutOutgoing(g, x), [x], [y], z)
}

function* subsetsOf(items: readonly string[]): Generator<string[]> {
  for (let k = 0; k <= items.length; k++) {
    const rec = function* (start: number, acc: string[]): Generator<string[]> {
      if (acc.length === k) {
        yield [...acc]
        return
      }
      for (let i = start; i < items.length; i++) yield* rec(i + 1, [...acc, items[i] as string])
    }
    yield* rec(0, [])
  }
}

/**
 * Every minimal back-door adjustment set of observed variables (pgmpy's `get_all_backdoor_adjustment_sets`): subsets of
 * the observed non-descendants of x by increasing size, skipping supersets of sets already found. `[[]]` when no
 * adjustment is needed, `[]` when none exists.
 *
 * @param g - The causal DAG.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param options - Latent variables.
 * @returns The minimal sets (each sorted).
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { backdoorAdjustmentSets } from 'mdml-engine/pgm/causal'
 * const g = dag([['a', 'x'], ['a', 'b'], ['b', 'y'], ['x', 'y']])
 * backdoorAdjustmentSets(g, 'x', 'y') // => [['a'], ['b']]
 * ```
 */
export function backdoorAdjustmentSets(
  g: DirectedModel,
  x: string,
  y: string,
  options: CausalGraphOptions = {},
): string[][] {
  check(g, x, y)
  const latent = new Set(options.latent ?? [])
  const desc = new Set(descendants(g, x))
  const pool = nodeNames(g)
    .filter((v) => v !== x && v !== y && !desc.has(v) && !latent.has(v))
    .sort()
  const found: string[][] = []
  for (const s of subsetsOf(pool)) {
    if (found.some((f) => f.every((v) => s.includes(v)))) continue
    if (isBackdoorAdjustmentSet(g, x, y, s)) found.push(s)
  }
  return found
}

/**
 * A minimal adjustment set (pgmpy's `get_minimal_adjustment_set`): a minimal d-separator of x and y in the proper
 * back-door graph, from which the first edge of every directed path from x to y has been removed. Latent parents in
 * the starting separator are replaced by their parents; when the observed variables cannot separate, there is none.
 *
 * @param g - The causal DAG.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param options - The latent variables.
 * @returns The set (sorted), or `null` when no observed set works (or x and y stay adjacent).
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { minimalAdjustmentSet } from 'mdml-engine/pgm/causal'
 * minimalAdjustmentSet(dag([['z', 'x'], ['z', 'y'], ['x', 'm'], ['m', 'y']]), 'x', 'y') // => ['z']
 * ```
 */
export function minimalAdjustmentSet(
  g: DirectedModel,
  x: string,
  y: string,
  options: CausalGraphOptions = {},
): string[] | null {
  check(g, x, y)
  // First edges of directed paths x -> c -> ... -> y: the edges x -> c with c an ancestor of y (or y itself).
  const toRemove = new Set(
    g.nodes.filter((n) => n.parents.includes(x) && (n.name === y || directedPath(g, n.name, y))).map((n) => n.name),
  )
  const proper: DirectedModel = {
    nodes: g.nodes.map(
      (n): DagNode => ({ name: n.name, parents: toRemove.has(n.name) ? n.parents.filter((p) => p !== x) : n.parents }),
    ),
  }
  const latent = new Set(options.latent ?? [])
  if (latent.size === 0) return minimalDSeparator(proper, x, y)
  const parentsOf = (v: string): readonly string[] => proper.nodes.find((n) => n.name === v)?.parents ?? []
  if (parentsOf(x).includes(y) || parentsOf(y).includes(x)) return null
  let sep = new Set([...parentsOf(x), ...parentsOf(y)])
  while ([...sep].some((v) => latent.has(v))) {
    const next = new Set<string>()
    for (const v of sep) {
      if (latent.has(v)) for (const p of parentsOf(v)) next.add(p)
      else next.add(v)
    }
    sep = next
  }
  sep.delete(x)
  sep.delete(y)
  let current = [...sep].sort()
  if (!dSeparated(proper, [x], [y], current)) return null
  for (const u of [...current]) {
    const without = current.filter((v) => v !== u)
    if (dSeparated(proper, [x], [y], without)) current = without
  }
  return current
}

/**
 * Whether a set satisfies the front-door criterion (Pearl 1995) as pgmpy states it: it intercepts every directed path
 * from x to y, there is no unblocked back-door path from x to any member, and x blocks every back-door path from each
 * member to y.
 *
 * @param g - The causal DAG (latent confounders included as nodes).
 * @param x - The treatment.
 * @param y - The outcome.
 * @param z - The candidate mediator set.
 * @returns `true` when valid.
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { isFrontdoorAdjustmentSet } from 'mdml-engine/pgm/causal'
 * const g = dag([['u', 'x'], ['u', 'y'], ['x', 'm'], ['m', 'y']])
 * isFrontdoorAdjustmentSet(g, 'x', 'y', ['m']) // => true
 * ```
 */
export function isFrontdoorAdjustmentSet(g: DirectedModel, x: string, y: string, z: readonly string[]): boolean {
  check(g, x, y, ...z)
  if (!directedPath(g, x, y)) return false
  // Every directed path from x to y must pass through z: none remains once z is removed.
  const cut: DirectedModel = {
    nodes: g.nodes
      .filter((n) => !z.includes(n.name))
      .map((n) => ({ name: n.name, parents: n.parents.filter((p) => !z.includes(p)) })),
  }
  if (directedPath(cut, x, y)) return false
  for (const m of z) if (!isBackdoorAdjustmentSet(g, x, m, [])) return false
  for (const m of z) if (!isBackdoorAdjustmentSet(g, m, y, [x])) return false
  return true
}

/**
 * Every front-door adjustment set among the observed variables (pgmpy's `get_all_frontdoor_adjustment_sets`).
 *
 * @param g - The causal DAG.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param options - Latent variables (never part of a set).
 * @returns The valid sets (each sorted), in increasing size.
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { frontdoorAdjustmentSets } from 'mdml-engine/pgm/causal'
 * frontdoorAdjustmentSets(dag([['u', 'x'], ['u', 'y'], ['x', 'm'], ['m', 'y']]), 'x', 'y', { latent: ['u'] }) // => [['m']]
 * ```
 */
export function frontdoorAdjustmentSets(
  g: DirectedModel,
  x: string,
  y: string,
  options: CausalGraphOptions = {},
): string[][] {
  check(g, x, y)
  const latent = new Set(options.latent ?? [])
  const pool = nodeNames(g)
    .filter((v) => v !== x && v !== y && !latent.has(v))
    .sort()
  return [...subsetsOf(pool)].filter((s) => s.length > 0 && isFrontdoorAdjustmentSet(g, x, y, s))
}

function stateOf(bn: BayesNet, v: string, s: string | number): number {
  const states = bayesNode(bn, v).states
  const i = typeof s === 'number' ? s : states.indexOf(s)
  if (!(i >= 0 && i < states.length)) throw new MdmlError('value', `${v} has no state ${String(s)}`)
  return i
}

/**
 * The back-door adjustment formula P(y | do(x)) = sum_z P(y | x, z) P(z), evaluated from a network's observational
 * distribution (equal to the interventional distribution when z satisfies the back-door criterion).
 *
 * @param bn - The network (the observational distribution).
 * @param x - The treatment.
 * @param y - The outcome.
 * @param z - The adjustment set.
 * @param treatment - The treatment state.
 * @returns The distribution over y.
 * @concept identification
 * @example
 * ```ts
 * import { bayesNet, factorValues } from 'mdml-engine/pgm'
 * import { backdoorAdjustment } from 'mdml-engine/pgm/causal'
 * const bn = bayesNet([
 *   { name: 'z', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] },
 *   { name: 'x', states: ['0', '1'], parents: ['z'], cpt: [0.8, 0.2, 0.2, 0.8] },
 *   { name: 'y', states: ['0', '1'], parents: ['x', 'z'], cpt: [0.9, 0.1, 0.5, 0.5, 0.6, 0.4, 0.2, 0.8] },
 * ])
 * factorValues(backdoorAdjustment(bn, 'x', 'y', ['z'], '1')).map((p) => Math.round(p * 1e12) / 1e12) // => [0.4, 0.6]
 * ```
 */
export function backdoorAdjustment(
  bn: BayesNet,
  x: string,
  y: string,
  z: readonly string[],
  treatment: string | number,
): Factor {
  const xs = stateOf(bn, x, treatment)
  const k = bayesNode(bn, y).states.length
  const out = new Array<number>(k).fill(0)
  const pz = z.length === 0 ? null : variableElimination(bn, z).posterior
  const configs = pz === null ? [{}] : allConfigs(bn, z)
  for (const [i, c] of configs.entries()) {
    const weight = pz === null ? 1 : Math.exp(pz.logValues[i] ?? Number.NEGATIVE_INFINITY)
    if (weight === 0) continue
    const cond = factorValues(variableElimination(bn, [y], { evidence: { ...c, [x]: xs } }).posterior)
    for (let s = 0; s < k; s++) out[s] = (out[s] ?? 0) + weight * (cond[s] ?? 0)
  }
  return factor([y], [k], out)
}

function allConfigs(bn: BayesNet, vars: readonly string[]): Record<string, number>[] {
  let out: Record<string, number>[] = [{}]
  for (const v of vars) out = out.flatMap((c) => bayesNode(bn, v).states.map((_, s) => ({ ...c, [v]: s })))
  return out
}

/**
 * The front-door adjustment formula P(y | do(x)) = sum_m P(m | x) sum_x' P(y | x', m) P(x'), evaluated from a
 * network's observational distribution over the observed variables.
 *
 * @param bn - The network.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param m - The mediator set.
 * @param treatment - The treatment state.
 * @returns The distribution over y.
 * @concept identification
 * @example
 * ```ts
 * import { bayesNet, factorValues } from 'mdml-engine/pgm'
 * import { frontdoorAdjustment, interventionalDistribution } from 'mdml-engine/pgm/causal'
 * const bn = bayesNet([
 *   { name: 'u', states: ['0', '1'], parents: [], cpt: [0.6, 0.4] },
 *   { name: 'x', states: ['0', '1'], parents: ['u'], cpt: [0.7, 0.3, 0.2, 0.8] },
 *   { name: 'm', states: ['0', '1'], parents: ['x'], cpt: [0.9, 0.1, 0.3, 0.7] },
 *   { name: 'y', states: ['0', '1'], parents: ['m', 'u'], cpt: [0.8, 0.2, 0.4, 0.6, 0.5, 0.5, 0.1, 0.9] },
 * ])
 * const a = factorValues(frontdoorAdjustment(bn, 'x', 'y', ['m'], '1'))[1] ?? 0
 * const b = factorValues(interventionalDistribution(bn, ['y'], { x: '1' }))[1] ?? 0
 * Math.abs(a - b) < 1e-12 // => true
 * ```
 */
export function frontdoorAdjustment(
  bn: BayesNet,
  x: string,
  y: string,
  m: readonly string[],
  treatment: string | number,
): Factor {
  const xs = stateOf(bn, x, treatment)
  const k = bayesNode(bn, y).states.length
  const out = new Array<number>(k).fill(0)
  const px = factorValues(variableElimination(bn, [x]).posterior)
  const pmGivenX = variableElimination(bn, [...m], { evidence: { [x]: xs } }).posterior
  for (const [i, c] of allConfigs(bn, m).entries()) {
    const pm = Math.exp(pmGivenX.logValues[i] ?? Number.NEGATIVE_INFINITY)
    if (pm === 0) continue
    for (const [xp, w] of px.entries()) {
      if (w === 0) continue
      const cond = factorValues(variableElimination(bn, [y], { evidence: { ...c, [x]: xp } }).posterior)
      for (let s = 0; s < k; s++) out[s] = (out[s] ?? 0) + pm * w * (cond[s] ?? 0)
    }
  }
  return factor([y], [k], out)
}

/** The outcome of {@link identifyEffect}. */
export interface Identification {
  readonly identifiable: boolean
  readonly method: 'backdoor' | 'frontdoor' | 'none'
  readonly set: readonly string[]
  /** The estimand, written out. */
  readonly formula: string
}

/**
 * Identify the effect of x on y from observational data in the simple cases of the ID algorithm (Tian and Pearl
 * 2002; Shpitser and Pearl 2006): by back-door adjustment when the observed variables admit an adjustment set (the
 * smallest, empty when there is no confounding), otherwise by front-door adjustment; otherwise report that neither
 * criterion applies.
 *
 * @param g - The causal DAG with latent confounders as nodes.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param options - The latent variables.
 * @returns Whether and how the effect is identified, with the estimand.
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { identifyEffect } from 'mdml-engine/pgm/causal'
 * identifyEffect(dag([['u', 'x'], ['u', 'y'], ['x', 'm'], ['m', 'y']]), 'x', 'y', { latent: ['u'] }).method // => 'frontdoor'
 * ```
 */
export function identifyEffect(
  g: DirectedModel,
  x: string,
  y: string,
  options: CausalGraphOptions = {},
): Identification {
  const back = backdoorAdjustmentSets(g, x, y, options)
  const smallest = [...back].sort((a, b) => a.length - b.length)[0]
  if (smallest !== undefined) {
    const z = smallest.join(', ')
    return {
      identifiable: true,
      method: 'backdoor',
      set: smallest,
      formula:
        smallest.length === 0
          ? `P(${y} | do(${x})) = P(${y} | ${x})`
          : `P(${y} | do(${x})) = sum_{${z}} P(${y} | ${x}, ${z}) P(${z})`,
    }
  }
  const front = frontdoorAdjustmentSets(g, x, y, options)[0]
  if (front !== undefined) {
    const m = front.join(', ')
    return {
      identifiable: true,
      method: 'frontdoor',
      set: front,
      formula: `P(${y} | do(${x})) = sum_{${m}} P(${m} | ${x}) sum_{${x}'} P(${y} | ${x}', ${m}) P(${x}')`,
    }
  }
  return {
    identifiable: false,
    method: 'none',
    set: [],
    formula: `no back-door or front-door estimand for P(${y} | do(${x}))`,
  }
}

/**
 * Whether z is an instrumental variable for the effect of x on y (given a conditioning set w): relevance (z and x
 * d-connected given w), and exclusion with exogeneity (z and y d-separated given w once the edges out of x are
 * removed), with no member of w a descendant of x.
 *
 * @param g - The causal DAG with latent confounders as nodes.
 * @param z - The candidate instrument.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param given - The conditioning set w (default empty).
 * @returns `true` when z is a (conditional) instrument.
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { isInstrument } from 'mdml-engine/pgm/causal'
 * const g = dag([['z', 'x'], ['u', 'x'], ['u', 'y'], ['x', 'y']])
 * [isInstrument(g, 'z', 'x', 'y'), isInstrument(g, 'u', 'x', 'y')] // => [true, false]
 * ```
 */
export function isInstrument(
  g: DirectedModel,
  z: string,
  x: string,
  y: string,
  given: readonly string[] = [],
): boolean {
  check(g, z, x, y, ...given)
  if (z === x || z === y) return false
  const desc = new Set(descendants(g, x))
  if (given.some((w) => desc.has(w))) return false
  if (dSeparated(g, [z], [x], given)) return false
  return dSeparated(withoutOutgoing(g, x), [z], [y], given)
}

/**
 * The observed instrumental variables for the effect of x on y, unconditionally or given a set w.
 *
 * @param g - The causal DAG with latent confounders as nodes.
 * @param x - The treatment.
 * @param y - The outcome.
 * @param options - Latent variables and the conditioning set.
 * @returns The instruments, sorted.
 * @concept identification
 * @example
 * ```ts
 * import { dag } from 'mdml-engine/pgm'
 * import { instrumentalVariables } from 'mdml-engine/pgm/causal'
 * instrumentalVariables(dag([['z', 'x'], ['u', 'x'], ['u', 'y'], ['x', 'y']]), 'x', 'y', { latent: ['u'] }) // => ['z']
 * ```
 */
export function instrumentalVariables(
  g: DirectedModel,
  x: string,
  y: string,
  options: CausalGraphOptions & { readonly given?: readonly string[] } = {},
): string[] {
  const latent = new Set(options.latent ?? [])
  const given = options.given ?? []
  return nodeNames(g)
    .filter((v) => !latent.has(v) && !given.includes(v) && isInstrument(g, v, x, y, given))
    .sort()
}
