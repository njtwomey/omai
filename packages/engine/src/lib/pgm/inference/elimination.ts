/**
 * Variable elimination, exact inference that sums (or maximises) hidden variables out one at a time in a chosen order:
 * posterior queries given evidence, the log partition function or evidence, the most probable explanation by
 * max-product elimination, and MAP queries over a subset of variables.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  message,
  runAlgorithm,
  scalar,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import {
  type Factor,
  factorArgmax,
  factorFromJson,
  factorMarginal,
  factorProductAll,
  factorToJson,
  logFactorAt,
  logTotal,
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
import { type EliminationCostFn, type EliminationHeuristic, eliminationOrderAlgorithm } from '../triangulation.ts'
import { type EliminationSemiring, eliminationSemirings } from './elimination-options.ts'

/** Options of {@link variableElimination} and the other elimination routines. */
export interface EliminationOptions {
  /** Observed states by variable (state names or indices). */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The elimination order of the hidden variables (default: greedy by `heuristic` on the interaction graph). */
  readonly order?: readonly string[]
  /** The greedy ordering criterion when no order is given (default `minFill`): a name or a cost function. */
  readonly heuristic?: EliminationHeuristic | EliminationCostFn
  /** Where to send one `message` event per eliminated variable. */
  readonly tracer?: Tracer
  /** What to record (messages are recorded when the policy selects them). */
  readonly policy?: TracePolicy
}

/** Options of {@link variableEliminationAlgorithm}: those of the routines, and the semiring. */
export interface EliminationAlgorithmOptions extends Omit<EliminationOptions, 'tracer' | 'policy'> {
  /** How a variable is removed: `sum` (default, marginals) or `max` (most probable explanation); see the table. */
  readonly semiring?: 'sum' | 'max' | EliminationSemiring
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

/**
 * The state of {@link variableEliminationAlgorithm}: the factors as they stand, the variables eliminated so far, and
 * what the eliminations produced. The factors are tables of log values, so the state serialises (zero entries, which
 * are minus infinity, as strings) and a run resumes.
 */
export interface EliminationRunState {
  /** The order the hidden variables are eliminated in. */
  readonly order: readonly string[]
  /** The factors now: those not involving an eliminated variable, and the new factors formed. */
  readonly factors: readonly Factor[]
  /** The variables eliminated so far (also those in no factor, which cost nothing). */
  readonly eliminated: readonly string[]
  /** The product formed at each elimination, for the traceback of a maximisation (empty when summing). */
  readonly products: readonly { readonly variable: string; readonly factor: Factor }[]
  /** The size (entries) of the largest product formed so far. */
  readonly largest: number
  /** The induced width so far: the most variables in a product, minus one. */
  readonly width: number
}

/** The paths {@link variableEliminationAlgorithm} logs, as a type. */
export type EliminationRunPath = 've/eliminated' | 've/factorSize' | 've/width' | 've/factors'

/**
 * What {@link variableEliminationAlgorithm} logs, by path: the variable eliminated (a message with the scope and size
 * of the new factor and the number of factors multiplied), the size of the new factor, the induced width so far and the
 * number of factors now.
 */
export const eliminationRunContract: Contract<EliminationRunState, EliminationRunPath> = defineContract({
  've/eliminated': message({
    description: 'the variable eliminated, the scope and size of the new factor and the factors multiplied',
    payload: {
      variable: 'the variable',
      scope: 'the variables of the new factor',
      size: 'entries of the new factor',
      factors: 'factors multiplied',
    },
  }),
  've/factorSize': scalar({ description: 'entries of the factor formed by this elimination' }),
  've/width': scalar({ description: 'the induced width so far (the most variables in a product, minus one)' }),
  've/factors': scalar({ description: 'factors alive after this elimination' }),
})

/** Reduce the factors by the evidence and choose the order of the hidden variables. */
function prepare(
  model: DiscreteModel,
  keep: readonly string[],
  options: EliminationAlgorithmOptions,
  name: string,
): { readonly factors: Factor[]; readonly order: string[] } {
  const evidence = evidenceIndices(model, options.evidence ?? {})
  for (const q of keep) {
    modelVariable(model, q)
    if (q in evidence) throw new MdmlError('value', `${name}: ${q} is both queried and observed`)
  }
  const names = modelVariables(model).map((v) => v.name)
  const hidden = names.filter((v) => !keep.includes(v) && !(v in evidence))
  const all = modelFactors(model)
  const card = Object.fromEntries(modelVariables(model).map((v) => [v.name, v.states.length]))
  const order =
    options.order === undefined
      ? [
          ...runAlgorithm(
            eliminationOrderAlgorithm(interactionGraph(all, names), hidden, {
              heuristic: options.heuristic ?? 'minFill',
              cardinality: card,
            }),
            { maxSteps: hidden.length },
          ).state.order,
        ]
      : [...options.order]
  const missing = hidden.filter((v) => !order.includes(v))
  if (missing.length > 0) throw new MdmlError('value', `${name}: the order misses ${missing.join(', ')}`)
  return { factors: all.map((f) => reduceFactor(f, evidence)), order }
}

/**
 * Variable elimination as an {@link Algorithm}: reduce the factors by the evidence, then each step takes the next
 * hidden variable of the order, multiplies the factors that mention it and removes it from the product by the
 * semiring (sum, or max), leaving a new factor. The state holds the factors as they stand, so a page can show the
 * network shrink one variable at a time with the size of every intermediate factor. The routines
 * {@link variableElimination}, {@link logPartition} and {@link maxProductElimination} are this run to the end.
 *
 * @param model - The model.
 * @param query - The query variables (kept, not eliminated).
 * @param options - Evidence, elimination order or heuristic (a slot: see `eliminationHeuristics`), and the semiring.
 * @returns The algorithm; it is done when every hidden variable is eliminated.
 * @throws MdmlError with code `value` for unknown or observed query variables, or an incomplete order.
 * @concept variable-elimination
 * @example
 * ```ts
 * import { bayesNet, variableEliminationAlgorithm } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * const chain = bayesNet([
 *   ...bn.nodes,
 *   { name: 'late', states: ['no', 'yes'], parents: ['wet'], cpt: [0.7, 0.3, 0.1, 0.9] },
 * ])
 * const alg = variableEliminationAlgorithm(chain, ['rain'], { evidence: { late: 'yes' } })
 * const end = runAlgorithm(alg, { maxSteps: 5 }).state
 * [end.eliminated, end.factors.length] // => [['wet'], 2]
 * ```
 */
export function variableEliminationAlgorithm(
  model: DiscreteModel,
  query: readonly string[],
  options: EliminationAlgorithmOptions = {},
): Algorithm<EliminationRunState, EliminationRunPath> {
  return eliminationAlgorithm(model, query, options, undefined)
}

function eliminationAlgorithm(
  model: DiscreteModel,
  query: readonly string[],
  options: EliminationAlgorithmOptions,
  label: string | undefined,
): Algorithm<EliminationRunState, EliminationRunPath> {
  const sr =
    typeof options.semiring === 'object'
      ? options.semiring
      : eliminationSemirings.make({ kind: options.semiring ?? 'sum' })
  const p = prepare(
    model,
    query,
    options,
    label ?? (sr.name === 'sum' ? 'variableElimination' : 'maxProductElimination'),
  )
  return {
    name: sr.name === 'sum' ? 'variable-elimination' : 'max-product-elimination',
    config: { query: [...query], order: [...p.order], semiring: sr.name },
    init: () => ({ order: p.order, factors: p.factors, eliminated: [], products: [], largest: 0, width: 0 }),
    step: (s, ctx) => {
      const v = s.order[s.eliminated.length] as string
      const involved = s.factors.filter((f) => f.variables.includes(v))
      if (involved.length === 0) {
        ctx.emit('ve/eliminated', { variable: v, scope: [], size: 0, factors: 0 })
        ctx.log('ve/factorSize', 0)
        ctx.log('ve/width', s.width)
        ctx.log('ve/factors', s.factors.length)
        return { ...s, eliminated: [...s.eliminated, v] }
      }
      const product = factorProductAll(involved)
      const out = sr.eliminate(product, [v])
      const factors = [...s.factors.filter((f) => !f.variables.includes(v)), out]
      const width = Math.max(s.width, product.variables.length - 1)
      ctx.run.message(
        s.eliminated.length,
        sr.name === 'sum' ? 'eliminate' : 'maxEliminate',
        { variable: v, scope: [...out.variables], size: out.logValues.length, factors: involved.length },
        { from: v },
      )
      ctx.emit('ve/eliminated', {
        variable: v,
        scope: [...out.variables],
        size: out.logValues.length,
        factors: involved.length,
      })
      ctx.log('ve/factorSize', out.logValues.length)
      ctx.log('ve/width', width)
      ctx.log('ve/factors', factors.length)
      return {
        order: s.order,
        factors,
        eliminated: [...s.eliminated, v],
        products: sr.traceback ? [...s.products, { variable: v, factor: product }] : s.products,
        largest: Math.max(s.largest, product.logValues.length),
        width,
      }
    },
    done: (s) => (s.eliminated.length >= s.order.length ? 'converged' : undefined),
    observe: (s) => ({ metrics: { largestFactor: s.largest } }),
    serialise: (s): Json => ({
      order: [...s.order],
      factors: s.factors.map(factorToJson),
      eliminated: [...s.eliminated],
      products: s.products.map((q) => ({ variable: q.variable, factor: factorToJson(q.factor) })),
      largest: s.largest,
      width: s.width,
    }),
    deserialise: (j) => {
      const o = j as {
        order: string[]
        factors: Json[]
        eliminated: string[]
        products: { variable: string; factor: Json }[]
        largest: number
        width: number
      }
      return {
        order: o.order,
        factors: o.factors.map(factorFromJson),
        eliminated: o.eliminated,
        products: o.products.map((q) => ({ variable: q.variable, factor: factorFromJson(q.factor) })),
        largest: o.largest,
        width: o.width,
      }
    },
  }
}

/** Run the algorithm to the end under the caller's tracer and policy. */
function runElimination(
  model: DiscreteModel,
  query: readonly string[],
  options: EliminationOptions,
  semiring: 'sum' | 'max',
  label?: string,
): EliminationRunState {
  const { tracer, policy, ...rest } = options
  const alg = eliminationAlgorithm(model, query, { ...rest, semiring }, label)
  const order = (alg.config as { order: string[] }).order
  return runAlgorithm(alg, {
    maxSteps: order.length,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
  }).state
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
 * @throws MdmlError with code `value` for unknown or observed query variables, or an incomplete order.
 * @concept variable-elimination
 * @index function Exact inference by variable elimination.
 * @example
 * ```ts
 * import { bayesNet, variableElimination, factorValues } from 'mdml-engine/pgm'
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
  const st = runElimination(model, query, options, 'sum')
  const restricted = factorMarginal(factorProductAll(st.factors), [...query])
  return {
    posterior: normaliseFactor(restricted),
    logEvidence: logTotal(restricted),
    order: st.order,
    largestFactor: st.largest,
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
 * import { isingModel, logPartition } from 'mdml-engine/pgm'
 * logPartition(isingModel({ rows: 1, cols: 2, coupling: 0 })) === Math.log(4) // => true
 * ```
 */
export function logPartition(model: DiscreteModel, options: EliminationOptions = {}): number {
  return logTotal(factorProductAll(runElimination(model, [], options, 'sum', 'logPartition').factors))
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
 * import { bayesNet, maxProductElimination } from 'mdml-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * maxProductElimination(bn, { evidence: { wet: 'yes' } }).assignment // => { rain: 'yes' }
 * ```
 */
export function maxProductElimination(model: DiscreteModel, options: EliminationOptions = {}): MaxProductResult {
  const st = runElimination(model, [], options, 'max')
  const logValue = factorProductAll(st.factors).logValues[0] ?? Number.NEGATIVE_INFINITY
  const products = new Map(st.products.map((q) => [q.variable, q.factor]))
  const indices: Record<string, number> = {}
  for (const v of [...st.order].reverse()) {
    const product = products.get(v)
    if (product === undefined) {
      indices[v] = 0
      continue
    }
    const fixed = Object.fromEntries(product.variables.filter((w) => w !== v).map((w) => [w, indices[w] ?? 0]))
    indices[v] = factorArgmax(reduceFactor(product, fixed))[v] ?? 0
  }
  const ordered = Object.fromEntries(
    modelVariables(model)
      .map((v) => v.name)
      .filter((v) => v in indices)
      .map((v) => [v, indices[v] ?? 0]),
  )
  return { assignment: stateNames(model, ordered), indices: ordered, logValue, order: st.order }
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
 * import { bayesNet, mapQuery } from 'mdml-engine/pgm'
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
