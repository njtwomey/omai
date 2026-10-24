import { type Estimate, estimate, WhiteboxError } from '../base/index.ts'
import type { Structure } from '../scene/index.ts'
import { argmax, full, max, reshape, softmax, stack, sub, type Tensor, toFlat, type Value } from '../tensor/index.ts'

/**
 * The generating probabilistic program of a dataset (a `whitebox-engine/probprog` model), kept opaque here because
 * programs live in a higher tier.
 */
export type GeneratingProgram = (...args: never[]) => unknown

/** Known causal effects of a treatment on an outcome. */
export interface CausalEffects {
  /** The average treatment effect E[Y(1) - Y(0)]. */
  readonly ate?: number
  /** The average treatment effect on the treated. */
  readonly att?: number
  /** The conditional average treatment effect as a function of covariates `[n, d] -> [n]`. */
  readonly cate?: (x: Value) => Value
  /** Each unit's individual effect `[n]` (both potential outcomes are known for synthetic data). */
  readonly ite?: Tensor
  /** Both potential outcomes `[n]` each. */
  readonly potentialOutcomes?: { readonly control: Tensor; readonly treated: Tensor }
}

/**
 * What is known about how a dataset was generated, so methods can be checked against the truth rather than only against
 * held-out data. Every field is optional; generators fill what they know, and {@link completeTruth} derives what can
 * be derived from the rest.
 *
 * @concept dataset-truth
 */
export interface Truth {
  /** The generating program, so likelihoods and posteriors can be queried by inference. */
  readonly program?: GeneratingProgram
  /** The log density of rows: `log p(x, y)` for supervised data, `log p(x)` without `y`. */
  readonly logProb?: (x: Value, y?: Value) => Value
  /** The posterior of the target given features: class probabilities `[n, k]` for classification. */
  readonly posterior?: (x: Value) => Value
  /** E[y | x], the function a regressor should recover. */
  readonly regressionFunction?: (x: Value) => Value
  /** The conditional quantile of y given x at level `q`. */
  readonly quantile?: (x: Value, q: number) => Value
  /** The standard deviation of additive Gaussian noise on y. */
  readonly noiseScale?: number
  /** The error of the best possible predictor: the noise variance (regression) or the Bayes error rate. */
  readonly bayesError?: number
  /** The true cluster of each row `[n]`. */
  readonly clusters?: Tensor<'int32'>
  /** Indices (rows or time steps) where the generating regime changes. */
  readonly changepoints?: readonly number[]
  /** Latent variables: manifold coordinates, topics, skills, clean labels, missingness masks. */
  readonly latents?: Readonly<Record<string, unknown>>
  /** Causal effects, for treatment data. */
  readonly effects?: CausalEffects
  /** A true graph or model structure (a causal DAG, a planted community structure). */
  readonly structure?: Structure
  /** The features that influence y (indices into the columns of x). */
  readonly relevantFeatures?: readonly number[]
  /** Free-form further facts (true coefficients, parameters). */
  readonly extra?: Readonly<Record<string, unknown>>
}

/**
 * The class posterior implied by a joint log density, by Bayes' rule: `p(k | x) = softmax_k log p(x, k)`.
 *
 * @param logJoint - `log p(x, y)` for features `[n, d]` and labels `[n]`.
 * @param nClasses - The number of classes.
 * @returns A function from features `[n, d]` to class probabilities `[n, k]`.
 * @example
 * ```ts
 * import { posteriorFromJoint } from 'whitebox-engine/data'
 * import { mul, sub, tensor, toFlat } from 'whitebox-engine/tensor'
 * // Two classes with log p(x, k) = -(x - k)^2 / 2: at x = 0.5 they are equally likely.
 * const post = posteriorFromJoint((x, y) => mul(-0.5, mul(sub(x, y), sub(x, y))), 2)
 * toFlat(post(tensor([[0.5]]))) // => [0.5, 0.5]
 * ```
 */
export function posteriorFromJoint(logJoint: (x: Value, y: Value) => Value, nClasses: number): (x: Value) => Value {
  return (x) => {
    const n = x.shape[0] ?? 0
    const columns = Array.from({ length: nClasses }, (_, k) => {
      const lp = logJoint(x, full([n], k, { dtype: 'int32' }))
      return reshape(lp, [n])
    })
    return softmax(stack(columns, 1), 1)
  }
}

/**
 * Predictions of the Bayes-optimal rule: the most probable class under the true posterior, or the true regression
 * function.
 *
 * @param truth - The truth of a generated dataset.
 * @param x - Features `[n, d]`.
 * @returns Class indices `[n]` (int32) or predicted values.
 * @throws WhiteboxError with code `value` when the truth has neither a posterior nor a regression function.
 * @example
 * ```ts
 * import { bayesPredict } from 'whitebox-engine/data'
 * import { mul, tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(bayesPredict({ regressionFunction: (x) => mul(x, 2) }, tensor([1, 2]))) // => [2, 4]
 * ```
 */
export function bayesPredict(truth: Truth, x: Value): Value {
  if (truth.posterior !== undefined) return argmax(truth.posterior(x), 1)
  if (truth.regressionFunction !== undefined) return truth.regressionFunction(x)
  throw new WhiteboxError('value', 'bayesPredict: the truth has neither a posterior nor a regression function')
}

/**
 * A Monte Carlo estimate of the Bayes error rate, `E[1 - max_k p(k | x)]`, over feature rows drawn from the
 * generating distribution, with its standard error.
 *
 * @param posterior - The true class posterior.
 * @param x - Features `[n, d]` sampled from the generator.
 * @returns The estimate.
 * @example
 * ```ts
 * import { bayesErrorEstimate } from 'whitebox-engine/data'
 * import { tensor } from 'whitebox-engine/tensor'
 * bayesErrorEstimate(() => tensor([[0.9, 0.1], [0.7, 0.3]]), tensor([[0], [1]])).value // => 0.2
 * ```
 */
export function bayesErrorEstimate(posterior: (x: Value) => Value, x: Value): Estimate {
  const p = posterior(x)
  const loss = toFlat(sub(1, max(p, 1)))
  const n = loss.length
  const m = loss.reduce((s, v) => s + v, 0) / n
  const sd = Math.sqrt(loss.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, n - 1))
  return { ...estimate(Math.round(m * 1e12) / 1e12, sd / Math.sqrt(n)), method: 'monte-carlo' }
}

/** Options of {@link completeTruth}. */
export interface CompleteTruthOptions {
  /** The number of classes, to derive a posterior from a joint log density. */
  readonly nClasses?: number
  /** Feature rows from the generating distribution, to estimate the Bayes error from the posterior. */
  readonly x?: Value
}

/**
 * Fill the pieces of a truth that follow from the others: the posterior from the joint log density (Bayes' rule),
 * and the Bayes error from the posterior (Monte Carlo over `x`) or from the noise scale (the noise variance).
 *
 * @param truth - A partial truth.
 * @param options - The class count and sample rows used for derivations.
 * @returns The completed truth (given fields are never overwritten).
 * @example
 * ```ts
 * import { completeTruth } from 'whitebox-engine/data'
 * completeTruth({ noiseScale: 0.5 }).bayesError // => 0.25
 * ```
 */
export function completeTruth(truth: Truth, options: CompleteTruthOptions = {}): Truth {
  let out: Truth = truth
  const logProb = truth.logProb
  if (out.posterior === undefined && logProb !== undefined && options.nClasses !== undefined) {
    out = { ...out, posterior: posteriorFromJoint((x, y) => logProb(x, y), options.nClasses) }
  }
  if (out.bayesError === undefined) {
    if (out.noiseScale !== undefined) out = { ...out, bayesError: out.noiseScale ** 2 }
    else if (out.posterior !== undefined && options.x !== undefined)
      out = { ...out, bayesError: bayesErrorEstimate(out.posterior, options.x).value }
  }
  return out
}

/**
 * The fraction of rows whose true cluster agrees with a labelling, maximised over relabellings by greedy matching
 * of the largest overlaps (a quick check; `metrics` has the adjusted indices).
 *
 * @param truth - A truth with `clusters`.
 * @param labels - Predicted cluster labels `[n]`.
 * @returns The matched accuracy in [0, 1].
 * @throws WhiteboxError with code `value` when the truth has no clusters.
 * @example
 * ```ts
 * import { clusterAgreement } from 'whitebox-engine/data'
 * import { tensor } from 'whitebox-engine/tensor'
 * clusterAgreement({ clusters: tensor([0, 0, 1, 1], { dtype: 'int32' }) }, [1, 1, 0, 0]) // => 1
 * ```
 */
export function clusterAgreement(truth: Truth, labels: Value | readonly number[]): number {
  if (truth.clusters === undefined) throw new WhiteboxError('value', 'clusterAgreement: the truth has no clusters')
  const a = truth.clusters.data
  const b = Array.isArray(labels) ? labels : toFlat(labels as Value)
  const counts = new Map<string, number>()
  for (let i = 0; i < a.length; i++) {
    const k = `${a[i]},${b[i]}`
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  const pairs = [...counts.entries()].sort((p, q) => q[1] - p[1])
  const usedA = new Set<string>()
  const usedB = new Set<string>()
  let matched = 0
  for (const [k, c] of pairs) {
    const [i, j] = k.split(',') as [string, string]
    if (usedA.has(i) || usedB.has(j)) continue
    usedA.add(i)
    usedB.add(j)
    matched += c
  }
  return a.length === 0 ? 1 : matched / a.length
}

/**
 * The mean squared distance between a function and the true regression function over feature rows: the excess risk
 * of a regressor under squared loss.
 *
 * @param truth - A truth with a regression function.
 * @param predict - The fitted function.
 * @param x - Feature rows `[n, d]`.
 * @returns `mean((predict(x) - f(x))^2)`.
 * @throws WhiteboxError with code `value` when the truth has no regression function.
 * @example
 * ```ts
 * import { excessRisk } from 'whitebox-engine/data'
 * import { add, tensor } from 'whitebox-engine/tensor'
 * excessRisk({ regressionFunction: (x) => x }, (x) => add(x, 1), tensor([1, 2, 3])) // => 1
 * ```
 */
export function excessRisk(truth: Truth, predict: (x: Value) => Value, x: Value): number {
  const f = truth.regressionFunction
  if (f === undefined) throw new WhiteboxError('value', 'excessRisk: the truth has no regression function')
  const d = sub(predict(x), f(x))
  const flat = toFlat(d)
  const s = flat.reduce((acc, v) => acc + v * v, 0)
  return s / flat.length
}
