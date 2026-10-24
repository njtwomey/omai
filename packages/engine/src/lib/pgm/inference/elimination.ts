import { WhiteboxError } from '../../base/index.ts'
import { nullTracer, type RunHandle, startRun, type TracePolicy, type Tracer } from '../../trace/index.ts'
import {
  type Factor,
  factorArgmax,
  factorMarginal,
  factorProductAll,
  logFactorAt,
  logTotal,
  marginalise,
  maxMarginalise,
  normaliseFactor,
  reduceFactor,
} from '../factor.ts'
import { interactionGraph } from '../graphs.ts'
import {
  type DiscreteModel,
  evidenceIndices,
  modelFactors,
  modelVariable,
  modelVariables,
  stateNames,
} from '../models/factor-graph.ts'
import { type EliminationHeuristic, eliminationOrder } from '../triangulation.ts'

/** Options of {@link variableElimination} and the other elimination routines. */
export interface EliminationOptions {
  /** Observed states by variable (state names or indices). */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The elimination order of the hidden variables (default: greedy by `heuristic` on the interaction graph). */
  readonly order?: readonly string[]
  /** The greedy ordering criterion when no order is given (default `minFill`). */
  readonly heuristic?: EliminationHeuristic
  /** Where to send one `message` event per eliminated variable. */
  readonly tracer?: Tracer
  /** What to record (messages are recorded when the policy selects them). */
  readonly policy?: TracePolicy
}

/** Result of {@link variableElimination}. */
export interface EliminationResult {
  /** The posterior over the query variables (normalised), in query order. */
  readonly posterior: Factor
  /** log P(evidence) for a Bayes network; log of the reduced partition function for other models. */
  readonly logEvidence: number
  /** The order in which hidden variables were eliminated. */
  readonly order: readonly string[]
  /** The size (number of entries) of the largest intermediate factor. */
  readonly largestFactor: number
}

interface Prepared {
  readonly factors: Factor[]
  readonly evidence: Record<string, number>
  readonly order: string[]
  readonly run: RunHandle
}

/** Reduce the factors by the evidence, choose the order of the hidden variables and start the traced run. */
function prepare(model: DiscreteModel, keep: readonly string[], options: EliminationOptions, name: string): Prepared {
  const evidence = evidenceIndices(model, options.evidence ?? {})
  for (const q of keep) {
    modelVariable(model, q)
    if (q in evidence) throw new WhiteboxError('value', `${name}: ${q} is both queried and observed`)
  }
  const names = modelVariables(model).map((v) => v.name)
  const hidden = names.filter((v) => !keep.includes(v) && !(v in evidence))
  const all = modelFactors(model)
  const card = Object.fromEntries(modelVariables(model).map((v) => [v.name, v.states.length]))
  const order =
    options.order === undefined
      ? eliminationOrder(interactionGraph(all, names), hidden, options.heuristic ?? 'minFill', card)
      : [...options.order]
  const missing = hidden.filter((v) => !order.includes(v))
  if (missing.length > 0) throw new WhiteboxError('value', `${name}: the order misses ${missing.join(', ')}`)
  const run = startRun(options.tracer ?? nullTracer, {
    name,
    seed: null,
    config: { query: [...keep], order: [...order] },
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return { factors: all.map((f) => reduceFactor(f, evidence)), evidence, order, run }
}

/**
 * Eliminate variables one at a time, recording each step; returns the remaining factors and the product factor formed
 * at each step (for max-product traceback).
 */
function eliminateAll(
  p: Prepared,
  mode: 'sum' | 'max',
): { factors: Factor[]; products: Map<string, Factor>; largest: number } {
  let factors = p.factors
  const products = new Map<string, Factor>()
  let largest = 0
  for (const [step, v] of p.order.entries()) {
    const involved = factors.filter((f) => f.variables.includes(v))
    if (involved.length === 0) continue
    const product = factorProductAll(involved)
    products.set(v, product)
    largest = Math.max(largest, product.logValues.length)
    const message = mode === 'sum' ? marginalise(product, [v]) : maxMarginalise(product, [v])
    factors = [...factors.filter((f) => !f.variables.includes(v)), message]
    p.run.message(
      step,
      mode === 'sum' ? 'eliminate' : 'maxEliminate',
      { variable: v, scope: [...message.variables], size: message.logValues.length, factors: involved.length },
      { from: v },
    )
  }
  return { factors, products, largest }
}

/**
 * Exact inference by variable elimination (sum-product): reduce the factors by the evidence, then eliminate the hidden
 * variables one at a time, multiplying the factors that mention each and summing it out. Works for Bayes networks,
 * Markov networks and factor graphs. Each elimination is recorded as a `message` event (the variable, the scope of the
 * new factor and its size), so a drawing of the network can replay the computation.
 *
 * @param model - The model.
 * @param query - The query variables.
 * @param options - Evidence, elimination order or heuristic, and tracer.
 * @returns The posterior over the query, log P(evidence), the order and the largest factor size.
 * @throws WhiteboxError with code `value` for unknown or observed query variables, or an incomplete order.
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
  model: DiscreteModel,
  query: readonly string[],
  options: EliminationOptions = {},
): EliminationResult {
  const p = prepare(model, query, options, 'variableElimination')
  const { factors, largest } = eliminateAll(p, 'sum')
  const joint = factorProductAll(factors)
  const restricted = factorMarginal(joint, [...query])
  p.run.end('completed', p.order.length, { largestFactor: largest })
  return {
    posterior: normaliseFactor(restricted),
    logEvidence: logTotal(restricted),
    order: p.order,
    largestFactor: largest,
  }
}

/**
 * The log partition function of a model, log Z = log sum_x prod_c psi_c(x_c), by variable elimination; with evidence,
 * the log of the partition function of the reduced model (log P(evidence) for a Bayes network).
 *
 * @param model - The model.
 * @param options - Evidence, order and tracer.
 * @returns log Z.
 * @concept partition-function
 * @example
 * ```ts
 * import { isingModel, logPartition } from 'whitebox-engine/pgm'
 * logPartition(isingModel({ rows: 1, cols: 2, coupling: 0 })) === Math.log(4) // => true
 * ```
 */
export function logPartition(model: DiscreteModel, options: EliminationOptions = {}): number {
  const p = prepare(model, [], options, 'logPartition')
  const { factors, largest } = eliminateAll(p, 'sum')
  p.run.end('completed', p.order.length, { largestFactor: largest })
  return logTotal(factorProductAll(factors))
}

/** Result of {@link maxProductElimination}. */
export interface MaxProductResult {
  /** The most probable joint assignment of every unobserved variable (state names). */
  readonly assignment: Record<string, string>
  /** The same assignment as state indices. */
  readonly indices: Record<string, number>
  /** log of the unnormalised probability of the assignment with the evidence (log P(x, e) for a Bayes network). */
  readonly logValue: number
  /** The order of elimination. */
  readonly order: readonly string[]
}

/**
 * The most probable explanation (MPE) by max-product variable elimination: maximise the hidden variables out one at a
 * time, then recover the maximising assignment by traceback in reverse elimination order (Koller and Friedman,
 * algorithm 13.1). Ties go to the first state.
 *
 * @param model - The model.
 * @param options - Evidence, order and tracer (`maxEliminate` messages).
 * @returns The assignment and its log value.
 * @concept map-inference
 * @example
 * ```ts
 * import { bayesNet, maxProductElimination } from 'whitebox-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * maxProductElimination(bn, { evidence: { wet: 'yes' } }).assignment // => { rain: 'yes' }
 * ```
 */
export function maxProductElimination(model: DiscreteModel, options: EliminationOptions = {}): MaxProductResult {
  const p = prepare(model, [], options, 'maxProductElimination')
  const { factors, products, largest } = eliminateAll(p, 'max')
  const logValue = factorProductAll(factors).logValues[0] ?? Number.NEGATIVE_INFINITY
  const indices: Record<string, number> = {}
  for (const v of [...p.order].reverse()) {
    const product = products.get(v)
    if (product === undefined) {
      indices[v] = 0
      continue
    }
    const fixed = Object.fromEntries(product.variables.filter((w) => w !== v).map((w) => [w, indices[w] ?? 0]))
    indices[v] = factorArgmax(reduceFactor(product, fixed))[v] ?? 0
  }
  p.run.end('completed', p.order.length, { largestFactor: largest, logValue })
  const ordered = Object.fromEntries(
    modelVariables(model)
      .map((v) => v.name)
      .filter((v) => v in indices)
      .map((v) => [v, indices[v] ?? 0]),
  )
  return { assignment: stateNames(model, ordered), indices: ordered, logValue, order: p.order }
}

/**
 * The most probable joint assignment of the query variables given evidence. When the query is every unobserved
 * variable this is the MPE, found by max-product elimination; otherwise it is the marginal MAP of the query (every
 * other hidden variable summed out), the mode of the query's posterior, as pgmpy's `map_query`.
 *
 * @param model - The model.
 * @param query - The variables to assign.
 * @param options - Evidence, order and tracer.
 * @returns The state names of the most probable assignment (in query order), and its posterior probability.
 * @example
 * ```ts
 * import { bayesNet, mapQuery } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] }])
 * mapQuery(bn, ['rain']).assignment // => { rain: 'no' }
 * ```
 */
export function mapQuery(
  model: DiscreteModel,
  query: readonly string[],
  options: EliminationOptions = {},
): { readonly assignment: Record<string, string>; readonly probability: number } {
  const evidence = evidenceIndices(model, options.evidence ?? {})
  const unobserved = modelVariables(model).filter((v) => !(v.name in evidence))
  const ordered = (a: Readonly<Record<string, string>>): Record<string, string> =>
    Object.fromEntries(query.map((q) => [q, a[q] ?? '']))
  if (unobserved.length === query.length && unobserved.every((v) => query.includes(v.name))) {
    // Max-product eliminates every unobserved variable: a given order is used when it lists them all.
    const { order, ...rest } = options
    const complete = order !== undefined && unobserved.every((v) => order.includes(v.name))
    const best = maxProductElimination(model, complete ? options : rest)
    const logEvidence = logPartition(model, options.evidence === undefined ? {} : { evidence: options.evidence })
    return { assignment: ordered(best.assignment), probability: Math.exp(best.logValue - logEvidence) }
  }
  const { posterior } = variableElimination(model, query, options)
  const best = factorArgmax(posterior)
  return { assignment: ordered(stateNames(model, best)), probability: Math.exp(logFactorAt(posterior, best)) }
}
