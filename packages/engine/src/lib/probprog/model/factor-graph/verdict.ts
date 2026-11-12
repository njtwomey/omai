/**
 * The verdict of a compilation for a message-passing method: `factorGraphVerdict` says whether a bound model has a factor
 * graph that `ep`, `vmp` or `bp` can run and exactly why not (the issues, the operators that lack the rule), with
 * `operatorsWithout` and `MessageRule`, and `compileFactorGraph` indexes the compiled graph for the engine.
 */

import { OmaiError } from '../../../base/index.ts'
import {
  type CompiledGraph,
  compileGraph,
  type FactorOperator,
  defaultOperators as factorOperators,
} from '../../../message-passing/index.ts'
import type { FactorGraphCompilation } from './types.ts'

/** The message rule a method asks of every operator of a graph. */
export type MessageRule = 'ep' | 'vmp' | 'bp'

/**
 * The factors of a compiled graph whose operator has no message rule for a method (an operator with only an EP rule, such as a hard comparison, cannot
 * run variational message passing; one with no `tilted` and no `ep` cannot run EP), with the operator's name: what a
 * method's capability check reports.
 *
 * @param g - The compiled graph.
 * @param rule - `ep`, `vmp` or `bp`.
 * @returns The names of the operators that lack the rule, without repeats.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators } from 'omai-engine/message-passing'
 * import { operatorsWithout } from 'omai-engine/probprog/model'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'b', family: { kind: 'bernoulli' }, observed: 1 }], factors: [{ operator: 'greater-than', args: ['b', 'x'], params: { threshold: 0 } }, { operator: 'prior', args: ['x'], params: { eta: [0, 1] } }] }, defaultOperators())
 * [operatorsWithout(g, 'ep'), operatorsWithout(g, 'vmp')] // => [[], ['greater-than']]
 * ```
 */
export function operatorsWithout(g: CompiledGraph, rule: MessageRule): string[] {
  const out = new Set<string>()
  for (const f of g.factors) {
    const op = f.operator
    const has = rule === 'ep' ? op.ep !== undefined || op.tilted !== undefined : op[rule] !== undefined
    if (!has || op.unsupported?.(rule, f.params) !== undefined) out.add(op.name)
  }
  return [...out]
}

/** The number of distinct factors of one operator that read a variable as an input (any port but the first). */
function readersOf(g: CompiledGraph, variable: number, operator: string): number {
  const seen = new Set<number>()
  for (const e of g.variables[variable]?.edges ?? []) {
    const edge = g.edges[e]
    if (edge !== undefined && edge.port > 0 && g.factors[edge.factor]?.operator.name === operator) seen.add(edge.factor)
  }
  return seen.size
}

/** A unit of a network layer: its `apply` factor, the output weights it is multiplied by and the input weights its affine map reads. */
interface LayerUnit {
  readonly factor: number
  readonly outputWeights: ReadonlySet<number>
  readonly inputWeights: ReadonlySet<number>
}

/** The units of a layer: the `apply` factors with a copy (another unit that shares an input weight and an output weight) in the graph. */
function layerUnits(g: CompiledGraph): LayerUnit[] {
  // Per unit (an `apply` factor): the output weights it is multiplied by and the input weights its affine map reads.
  const units: LayerUnit[] = []
  g.factors.forEach((f, factor) => {
    if (f.operator.name !== 'apply') return
    const out = f.ports[0]?.variable
    const input = f.ports[1]?.variable
    if (out === undefined || input === undefined) return
    // The output is multiplied by a random value that other rows multiply as well (the output weight of the unit).
    const outputWeights = new Set<number>()
    for (const e of g.variables[out]?.edges ?? []) {
      const edge = g.edges[e]
      const p = edge === undefined ? undefined : g.factors[edge.factor]
      if (edge === undefined || p === undefined || p.operator.name !== 'product' || edge.port === 0) continue
      p.ports.forEach((q, k) => {
        if (k > 0 && q.variable !== undefined && q.variable !== out && readersOf(g, q.variable, 'product') >= 2)
          outputWeights.add(q.variable)
      })
    }
    // The input is an affine map that reads a random weight other rows read as well (the input weight of the unit).
    const definer = g.variables[input]?.definer
    const d = definer === undefined ? undefined : g.edges[definer]
    const affine = d === undefined ? undefined : g.factors[d.factor]
    const inputWeights = new Set<number>()
    affine?.ports.forEach((q, k) => {
      if (
        k > 0 &&
        q.variable !== undefined &&
        q.value === undefined &&
        readersOf(g, q.variable, affine.operator.name) >= 2
      )
        inputWeights.add(q.variable)
    })
    if (outputWeights.size > 0 && inputWeights.size > 0) units.push({ factor, outputWeights, inputWeights })
  })
  // The weights of a hidden unit are copies of one another across rows only when the units that read an input weight also
  // multiply the same output weight: units that read one input value (the latent coordinates of a row, read by a feature each)
  // and multiply their own output weights are different features, told apart by their known constants, not copies of one unit.
  const readers = new Map<number, number[]>()
  units.forEach((u, i) => {
    for (const q of u.inputWeights) readers.set(q, [...(readers.get(q) ?? []), i])
  })
  return units.filter((u, i) =>
    [...u.inputWeights].some((q) =>
      (readers.get(q) ?? []).some(
        (j) => j !== i && [...(units[j]?.outputWeights ?? [])].some((w) => u.outputWeights.has(w)),
      ),
    ),
  )
}

/**
 * The units of a network layer in a compiled graph: the `apply` factors (`tanh`, `relu`, ...) whose input is an affine map of
 * random weights that other rows share (the first layer) and whose output is multiplied by a random weight that other rows share
 * (the next layer), and where the units that read one input weight multiply the same output weight (copies of one hidden
 * unit, which can be permuted). A hidden unit of a Bayesian neural network is one per unit and row; a hierarchy's `exp` of a scale
 * read by many products is not one (its other factor is per row), nor is a nonlinear link of a single regression, nor are the
 * random Fourier features of a latent-variable model (the features that read one latent coordinate multiply weights of their
 * own, and their known frequencies tell them apart). One hidden unit with any activation is counted too (a dose-response curve
 * `Emax sigmoid(a + b x)`): whether EP can hold its weights is {@link networkLayerShape}'s question.
 *
 * @param g - The compiled graph.
 * @returns The number of such units (0 when the graph has no network layer).
 * @example
 * ```ts
 * import { compileGraph, defaultOperators } from 'omai-engine/message-passing'
 * import { networkLayerUnits } from 'omai-engine/probprog/model'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'gaussian' } }], factors: [{ operator: 'apply', args: ['y', 'x'], params: { fn: 'tanh' } }] }, defaultOperators())
 * networkLayerUnits(g) // => 0
 * ```
 */
export function networkLayerUnits(g: CompiledGraph): number {
  return layerUnits(g).length
}

/**
 * What the units of a network layer are, which decides whether EP can hold their weights: the number of distinct hidden units
 * (units with different weights, which can be permuted) and the activations they use (`sigmoid`, `tanh`, ...; `chain` for a chain
 * of steps). Two or more hidden units have permutation symmetry. One hidden unit has none, and EP runs it when its activation is the
 * sigmoid (a dose-response curve `Emax sigmoid(a + b x)`: measured to converge to the posterior of a NUTS run); not otherwise: an
 * odd or even activation (`tanh`, `cos`) has two modes that flip the signs of the weights (EP converged to one of them, or did not
 * converge), and the rectifier and `exp` have a ridge, since the output weight can be traded for the input weights (EP did not
 * converge).
 *
 * @param g - The compiled graph.
 * @returns The number of distinct hidden units (0 without a layer) and the distinct activations of the units.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators } from 'omai-engine/message-passing'
 * import { networkLayerShape } from 'omai-engine/probprog/model'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'gaussian' } }], factors: [{ operator: 'apply', args: ['y', 'x'], params: { fn: 'tanh' } }] }, defaultOperators())
 * networkLayerShape(g).hidden // => 0
 * ```
 */
export function networkLayerShape(g: CompiledGraph): {
  readonly hidden: number
  readonly activations: readonly string[]
} {
  const units = layerUnits(g)
  const signature = (u: LayerUnit): string =>
    `${[...u.inputWeights].sort().join(',')}|${[...u.outputWeights].sort().join(',')}`
  const activation = (u: LayerUnit): string => {
    const params = g.factors[u.factor]?.params ?? {}
    const steps = params['steps']
    if (Array.isArray(steps) && steps.length > 0) return 'chain'
    return typeof params['fn'] === 'string' ? params['fn'] : 'unknown'
  }
  return { hidden: new Set(units.map(signature)).size, activations: [...new Set(units.map(activation))].sort() }
}

/**
 * Whether a bound model has a factor graph a message-passing method can run, and exactly why not: the static check behind
 * `ep`, `vmp` and `bp` as inference methods. It reads the compilation (the issues, with node and reason) and, for a rule, the
 * operators of the compiled graph that lack it; for `ep` it also refuses a graph with the units of a network layer
 * (`networkLayerUnits`, `networkLayerShape`: two or more distinct hidden units, or one whose activation is not the sigmoid), whose weights
 * have symmetric modes or a ridge that one Gaussian per weight does not hold.
 *
 * @param c - The compilation.
 * @param g - The compiled graph, when the compilation has a graph.
 * @param rule - The method's rule.
 * @returns `ok` with a reason, or the reasons it does not apply and the missing features.
 * @example
 * ```ts
 * import { defineModel, factorGraphVerdict, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), b.latent('s', normal(1, 1))), b.data('y', { dims: ['row'] })) })
 * factorGraphVerdict(toFactorGraph(m.bind({ y: [1, 2] })), undefined, 'ep').ok // => false
 * ```
 */
export function factorGraphVerdict(
  c: FactorGraphCompilation,
  g: CompiledGraph | undefined,
  rule: MessageRule,
): { readonly ok: boolean; readonly reason: string; readonly missing: readonly string[] } {
  if (c.graph === undefined || g === undefined) {
    const reasons = c.issues.map((i) => `${i.node}: ${i.reason}`)
    const missing = [
      ...new Set(
        c.issues.map((i) =>
          i.kind === 'opaque' || i.kind === 'family'
            ? 'exponential-family factors'
            : `a factor graph without ${i.kind} issues`,
        ),
      ),
    ]
    return { ok: false, reason: reasons.join('; '), missing }
  }
  const lacking = operatorsWithout(g, rule)
  if (lacking.length > 0) {
    const why = lacking.flatMap((name) => {
      const factors = g.factors.filter((f) => f.operator.name === name)
      const note =
        factors[0]?.operator.noRule?.[rule] ??
        factors.map((f) => f.operator.unsupported?.(rule, f.params)).find((v) => v !== undefined)
      return note === undefined ? [] : [`${name}: ${note}`]
    })
    return {
      ok: false,
      reason: `the operator${lacking.length > 1 ? 's' : ''} ${lacking.join(', ')} ${lacking.length > 1 ? 'have' : 'has'} no ${rule === 'ep' ? 'expectation propagation' : rule === 'vmp' ? 'variational' : 'sum-product'} message rule${why.length > 0 ? ` (${why.join('; ')})` : ''}`,
      missing: [`${rule} message rules`],
    }
  }
  if (rule === 'ep' && networkLayerUnits(g) > 0) {
    const { hidden, activations } = networkLayerShape(g)
    if (hidden >= 2)
      return {
        ok: false,
        reason: `the graph has the units of a neural-network layer (an \`apply\` of an affine map of weights that rows share, multiplied by a weight that rows share): the weights are not identified (sign and permutation symmetries), so a Gaussian per weight averages the modes or keeps one and drops the other units, and the run does not converge to a posterior`,
        missing: ['beliefs that hold the symmetric modes of a hidden layer'],
      }
    // One hidden unit has no permutation: the sigmoid has no sign symmetry or ridge either, and runs (a dose-response curve).
    const other = activations.filter((a) => a !== 'sigmoid')
    if (other.length > 0)
      return {
        ok: false,
        reason: `the graph has one hidden unit with the activation ${other.join(', ')} (an \`apply\` of an affine map of weights that rows share, multiplied by a weight that rows share): its weights are not identified (an odd or even activation has two modes that flip the signs of the weights, the rectifier and exp have a ridge between the output weight and the input weights), so a Gaussian per weight holds one mode or the run does not converge. A saturating sigmoid unit, as in a dose-response curve, runs`,
        missing: ['beliefs that hold the symmetric modes of a hidden unit'],
      }
  }
  return {
    ok: true,
    reason: `a factor graph of ${g.variables.length} variables and ${g.factors.length} factors`,
    missing: [],
  }
}

/**
 * The compiled graph (indexed, ready for the engine) of a compilation, against the engine's operators or a registry of
 * your own.
 *
 * @param c - The compilation.
 * @param operators - The factor operators by name (default the engine's `defaultOperators()`).
 * @returns The compiled graph.
 * @throws OmaiError with code `value` when the compilation has issues (no graph), naming each node and reason.
 * @concept model-spec
 * @example
 * ```ts
 * import { compileFactorGraph, defineModel, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), 1), b.data('y', { dims: ['row'] })) })
 * const g = compileFactorGraph(toFactorGraph(m.bind({ y: [1, 2, 3] })))
 * [g.variables.length, g.factors.length, g.edges.length] // => [4, 4, 4]
 * ```
 */
export function compileFactorGraph(
  c: FactorGraphCompilation,
  operators: Readonly<Record<string, FactorOperator>> = factorOperators(),
): CompiledGraph {
  if (c.graph === undefined)
    throw new OmaiError(
      'value',
      `'${c.name}' does not compile to a factor graph: ${c.issues.map((i) => `${i.node}: ${i.reason}`).join('; ')}`,
    )
  return compileGraph(c.graph, operators)
}
