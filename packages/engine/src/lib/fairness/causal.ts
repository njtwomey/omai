/**
 * Causal fairness against a structural causal model written as a probabilistic program: counterfactual fairness
 * (Kusner, Loftus, Russell and Silva, 2017), comparing a predictor's output for each individual with its output in the
 * world where the protected attribute had been different (the same exogenous noise, the attribute intervened on), and
 * total, natural direct and natural indirect effects of the attribute on an outcome (Pearl; Zhang and Bareinboim's
 * decomposition of discrimination), by simulation with common random numbers.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { doIntervention, type Model, seed, traceOf } from '../probprog/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { toFlat, type Value } from '../tensor/index.ts'

/** The value of each sample and deterministic site of a run, as numbers (the first element of each). */
export type SiteNumbers = Readonly<Record<string, number>>

/** One run of a model with a key and interventions: its sites as numbers. */
function run(model: Model, k: Key, interventions: Readonly<Record<string, number>>): SiteNumbers {
  const m = Object.keys(interventions).length === 0 ? model : doIntervention(model, interventions)
  const t = traceOf(seed(m, k), undefined)
  const out: Record<string, number> = {}
  for (const [name, site] of t.sites) {
    const v = (site as { readonly value?: Value }).value
    if (v !== undefined) out[name] = toFlat(v)[0] ?? Number.NaN
  }
  // An intervened site is recorded as name__CF with the intervention value: read it under its own name.
  for (const [name, v] of Object.entries(interventions)) out[name] = v
  return out
}

/** Options of {@link counterfactualFairness}. */
export interface CounterfactualFairnessOptions {
  /** The protected attribute's site. */
  readonly attribute: string
  /** Its two values (default 0 and 1). */
  readonly values?: readonly [number, number]
  /** The predictor's output from an individual's site values (features read from the trace). */
  readonly predictor: (sites: SiteNumbers) => number
  /** Individuals simulated (default 2000). */
  readonly n?: number
  /** Decisions are outputs at or above this (default 0.5). */
  readonly threshold?: number
  readonly key?: Key
}

/** The result of {@link counterfactualFairness}. */
export interface CounterfactualFairness {
  /** Each individual's attribute value, output, and output had the attribute been the other value. */
  readonly attribute: readonly number[]
  readonly factual: readonly number[]
  readonly counterfactual: readonly number[]
  /** The mean and largest |counterfactual - factual|: 0 for a counterfactually fair predictor. */
  readonly meanGap: number
  readonly maxGap: number
  /** The share of individuals whose decision would change. */
  readonly flipRate: number
}

/**
 * Counterfactual fairness (Kusner et al., 2017): a predictor is counterfactually fair when each individual's output
 * would be the same had their protected attribute been different, all else (the exogenous noise) equal. Each simulated
 * individual runs the structural causal model twice with the same key, as is and with the attribute intervened on
 * (`doIntervention`), and the predictor reads both runs' sites.
 *
 * @param model - The structural causal model, a program sampling every variable by name.
 * @param options - The attribute's site and values, the predictor, the number of individuals, threshold and key.
 * @returns Each individual's factual and counterfactual output, the gaps and the share of decisions that flip.
 * @throws OmaiError `value` when the attribute's site is not in the model.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { counterfactualFairness } from 'omai-engine/fairness'
 * import { bernoulli, normal } from 'omai-engine/dists'
 * import { sample } from 'omai-engine/probprog'
 * import { add, mul } from 'omai-engine/tensor'
 * const scm = () => {
 *   const a = sample('a', bernoulli({ probs: 0.5 }))
 *   const u = sample('u', normal(0, 1))
 *   sample('x', normal(add(mul(a, 2), u), 1e-9))
 * }
 * // A predictor of x inherits the attribute's effect; one of u alone does not.
 * counterfactualFairness(scm, { attribute: 'a', predictor: (s) => s['x'] ?? 0, n: 50 }).meanGap.toFixed(3) // => '2.000'
 * counterfactualFairness(scm, { attribute: 'a', predictor: (s) => s['u'] ?? 0, n: 50 }).meanGap // => 0
 * ```
 */
export function counterfactualFairness(model: Model, options: CounterfactualFairnessOptions): CounterfactualFairness {
  const [a0, a1] = options.values ?? [0, 1]
  const n = options.n ?? 2000
  const t = options.threshold ?? 0.5
  const k = options.key ?? makeKey(0)
  const attribute: number[] = []
  const factual: number[] = []
  const counterfactual: number[] = []
  for (let i = 0; i < n; i++) {
    const ki = foldIn(k, i)
    const f = run(model, ki, {})
    const a = f[options.attribute]
    if (a === undefined) throw new OmaiError('value', `counterfactualFairness: no site ${options.attribute}`)
    const cf = run(model, ki, { [options.attribute]: a === a1 ? a0 : a1 })
    attribute.push(a)
    factual.push(options.predictor(f))
    counterfactual.push(options.predictor(cf))
  }
  const gaps = factual.map((v, i) => Math.abs((counterfactual[i] ?? 0) - v))
  const flips = factual.filter((v, i) => v >= t !== (counterfactual[i] ?? 0) >= t).length
  return {
    attribute,
    factual,
    counterfactual,
    meanGap: gaps.reduce((s, v) => s + v, 0) / Math.max(1, n),
    maxGap: Math.max(0, maxOf(gaps)),
    flipRate: flips / Math.max(1, n),
  }
}

/** Options of {@link pathSpecificEffects}. */
export interface PathSpecificOptions {
  readonly attribute: string
  /** The reference and the compared value (default 0 and 1). */
  readonly values?: readonly [number, number]
  /** The outcome from a run's sites (a site's value, or a predictor's output). */
  readonly outcome: (sites: SiteNumbers) => number
  /** The mediators' sites: the indirect effect runs through them, the direct effect around them. */
  readonly mediators: readonly string[]
  readonly n?: number
  readonly key?: Key
}

/** Total, natural direct and natural indirect effects of the attribute on the outcome. */
export interface PathSpecificEffects {
  /** E[Y(a1)] - E[Y(a0)]. */
  readonly total: number
  /** E[Y(a1, M(a0))] - E[Y(a0)]: the effect not through the mediators. */
  readonly direct: number
  /** total - direct: the effect through the mediators. */
  readonly indirect: number
  /** The mean outcomes behind them. */
  readonly means: { readonly y0: number; readonly y1: number; readonly y1m0: number }
}

/**
 * Path-specific effects of a protected attribute (Pearl, 2001; Zhang and Bareinboim's direct and indirect
 * discrimination): the total effect of setting the attribute from a0 to a1, the natural direct effect (a1 with the
 * mediators held at what they would have been under a0, by a nested intervention on the same exogenous noise) and the
 * natural indirect effect, the rest. A direct effect is discrimination along the attribute's own path; an indirect one
 * runs through mediators that may or may not be legitimate (resolving variables).
 *
 * @param model - The structural causal model.
 * @param options - The attribute, its values, the outcome, the mediators, the number of individuals and key.
 * @returns The three effects and the means behind them.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { pathSpecificEffects } from 'omai-engine/fairness'
 * import { bernoulli, normal } from 'omai-engine/dists'
 * import { sample } from 'omai-engine/probprog'
 * import { add, mul } from 'omai-engine/tensor'
 * const scm = () => {
 *   const a = sample('a', bernoulli({ probs: 0.5 }))
 *   const m = sample('m', normal(mul(a, 2), 1))
 *   sample('y', normal(add(a, mul(m, 3)), 1))
 * }
 * const e = pathSpecificEffects(scm, { attribute: 'a', outcome: (s) => s['y'] ?? 0, mediators: ['m'], n: 400 })
 * e.direct.toFixed(6) // => '1.000000'
 * e.indirect.toFixed(6) // => '6.000000'
 * ```
 */
export function pathSpecificEffects(model: Model, options: PathSpecificOptions): PathSpecificEffects {
  const [a0, a1] = options.values ?? [0, 1]
  const n = options.n ?? 2000
  const k = options.key ?? makeKey(0)
  let y0 = 0
  let y1 = 0
  let y1m0 = 0
  for (let i = 0; i < n; i++) {
    const ki = foldIn(k, i)
    const w0 = run(model, ki, { [options.attribute]: a0 })
    const w1 = run(model, ki, { [options.attribute]: a1 })
    const held: Record<string, number> = { [options.attribute]: a1 }
    for (const m of options.mediators) {
      const v = w0[m]
      if (v === undefined) throw new OmaiError('value', `pathSpecificEffects: no site ${m}`)
      held[m] = v
    }
    const nested = run(model, ki, held)
    y0 += options.outcome(w0)
    y1 += options.outcome(w1)
    y1m0 += options.outcome(nested)
  }
  const means = { y0: y0 / n, y1: y1 / n, y1m0: y1m0 / n }
  const total = means.y1 - means.y0
  const direct = means.y1m0 - means.y0
  return { total, direct, indirect: total - direct, means }
}
