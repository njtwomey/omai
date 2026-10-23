import { WhiteboxError } from '../../base/index.ts'
import { nullTracer, startRun, type TracePolicy, type Tracer } from '../../trace/index.ts'
import {
  type Factor,
  factorArgmax,
  factorProduct,
  factorValues,
  logTotal,
  marginalise,
  normaliseFactor,
  reduceFactor,
} from '../factor.ts'
import { type BayesNet, bayesFactors, bayesNode } from '../models/bayes-net.ts'
import { eliminationOrder, moralGraph } from '../structure.ts'

/** Options of {@link variableElimination}. */
export interface EliminationOptions {
  /** Observed states by variable (state names or indices). */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The elimination order of the hidden variables (default: greedy min-fill on the moral graph). */
  readonly order?: readonly string[]
  /** Where to send one `message` event per eliminated variable. */
  readonly tracer?: Tracer
  /** What to record (messages are recorded when the policy selects them). */
  readonly policy?: TracePolicy
}

/** Result of {@link variableElimination}. */
export interface EliminationResult {
  /** The posterior over the query variables (normalised). */
  readonly posterior: Factor
  /** log P(evidence). */
  readonly logEvidence: number
  /** The order in which hidden variables were eliminated. */
  readonly order: readonly string[]
  /** The size (number of entries) of the largest intermediate factor. */
  readonly largestFactor: number
}

function evidenceIndices(bn: BayesNet, evidence: Readonly<Record<string, string | number>>): Record<string, number> {
  return Object.fromEntries(
    Object.entries(evidence).map(([k, v]) => {
      const states = bayesNode(bn, k).states
      const i = typeof v === 'number' ? v : states.indexOf(v)
      if (i < 0 || i >= states.length) throw new WhiteboxError('value', `evidence: ${k} has no state ${String(v)}`)
      return [k, i]
    }),
  )
}

/**
 * Exact inference by variable elimination (sum-product): reduce the CPT factors by the evidence, then eliminate the
 * hidden variables one at a time, multiplying the factors that mention each and summing it out. Each elimination is
 * recorded as a `message` event (the variable, the scope of the new factor and its size), so a drawing of the network
 * can replay the computation.
 *
 * @param bn - The network.
 * @param query - The query variables.
 * @param options - Evidence, elimination order and tracer.
 * @returns The posterior over the query, log P(evidence), the order and the largest factor size.
 * @concept variable-elimination
 * @example
 * ```ts
 * import { bayesNet, variableElimination, factorValues } from 'whitebox-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * factorValues(variableElimination(bn, ['rain'], { evidence: { wet: 'yes' } }).posterior).map((p) => Math.round(p * 1000) / 1000) // => [0.333, 0.667]
 * ```
 */
export function variableElimination(
  bn: BayesNet,
  query: readonly string[],
  options: EliminationOptions = {},
): EliminationResult {
  const evidence = evidenceIndices(bn, options.evidence ?? {})
  for (const q of query) {
    bayesNode(bn, q)
    if (q in evidence) throw new WhiteboxError('value', `variableElimination: ${q} is both queried and observed`)
  }
  let factors = bayesFactors(bn).map((f) => reduceFactor(f, evidence))
  const hidden = bn.nodes.map((n) => n.name).filter((v) => !query.includes(v) && !(v in evidence))
  const order = options.order ?? eliminationOrder(moralGraph(bn), hidden)
  const run = startRun(options.tracer ?? nullTracer, {
    name: 'variableElimination',
    seed: null,
    config: { query: [...query], order: [...order] },
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  let largest = 0
  order.forEach((v, step) => {
    const involved = factors.filter((f) => f.variables.includes(v))
    if (involved.length === 0) return
    const product = involved.reduce((a, b) => factorProduct(a, b))
    largest = Math.max(largest, product.logValues.length)
    const message = marginalise(product, [v])
    factors = [...factors.filter((f) => !f.variables.includes(v)), message]
    run.message(
      step,
      'eliminate',
      { variable: v, scope: [...message.variables], size: message.logValues.length, factors: involved.length },
      { from: v },
    )
  })
  const joint = factors.reduce((a, b) => factorProduct(a, b))
  // Put the query variables in the requested order and drop any constant (empty-scope) parts.
  const restricted = marginalise(
    joint,
    joint.variables.filter((v) => !query.includes(v)),
  )
  const ordered = reorder(restricted, query)
  run.end('completed', order.length, { largestFactor: largest })
  return { posterior: normaliseFactor(ordered), logEvidence: logTotal(ordered), order, largestFactor: largest }
}

/** The same factor over the variables in a given order. */
function reorder(f: Factor, variables: readonly string[]): Factor {
  if (f.variables.every((v, i) => v === variables[i])) return f
  const card = variables.map((v) => f.cardinality[f.variables.indexOf(v)] ?? 1)
  const values: number[] = []
  const total = card.reduce((a, b) => a * b, 1)
  for (let i = 0; i < total; i++) {
    let rest = i
    const assignment: Record<string, number> = {}
    for (let k = variables.length - 1; k >= 0; k--) {
      assignment[variables[k] as string] = rest % (card[k] ?? 1)
      rest = Math.floor(rest / (card[k] ?? 1))
    }
    let idx = 0
    for (let k = 0; k < f.variables.length; k++)
      idx = idx * (f.cardinality[k] ?? 1) + (assignment[f.variables[k] as string] ?? 0)
    values.push(f.logValues[idx] ?? Number.NEGATIVE_INFINITY)
  }
  return { variables: [...variables], cardinality: card, logValues: Float64Array.from(values) }
}

/**
 * The most probable joint assignment of the query variables given evidence: the mode of their posterior, with every
 * other hidden variable summed out (the marginal MAP of the query, as pgmpy's map_query).
 *
 * @param bn - The network.
 * @param query - The variables to assign.
 * @param options - Evidence, order and tracer.
 * @returns The state names of the most probable assignment, and its posterior probability.
 * @example
 * ```ts
 * import { bayesNet, mapQuery } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] }])
 * mapQuery(bn, ['rain']).assignment // => { rain: 'no' }
 * ```
 */
export function mapQuery(
  bn: BayesNet,
  query: readonly string[],
  options: EliminationOptions = {},
): { readonly assignment: Record<string, string>; readonly probability: number } {
  const { posterior } = variableElimination(bn, query, options)
  const best = factorArgmax(posterior)
  const probability = Math.max(...factorValues(posterior))
  return {
    assignment: Object.fromEntries(Object.entries(best).map(([k, i]) => [k, bayesNode(bn, k).states[i] ?? String(i)])),
    probability,
  }
}
