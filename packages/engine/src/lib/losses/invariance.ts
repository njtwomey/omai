/**
 * Penalties that ask a predictor to behave the same across environments (domains, groups, sites) so that it relies on
 * features whose relation to the label is stable and not on those that vary: the IRMv1 gradient penalty and the
 * V-REx variance of risks (the two invariant-risk-minimisation penalties), Deep CORAL's covariance alignment and the
 * MMD between environments' features (feature-distribution alignment), and gradient-based criteria, matching the
 * environments' gradients (Fish) and their per-sample gradient variances (Fishr). Each is differentiable in the
 * predictor's parameters, through the gradients taken inside them.
 */

import { grad } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import {
  add,
  div,
  matmul,
  mean,
  mul,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type TensorLike,
  tensor,
  transpose,
  type Value,
  variance,
} from '../tensor/index.ts'
import { binaryCrossEntropyWithLogits } from './classification.ts'
import { mmd } from './divergence.ts'

/** A risk of scores against targets (a mean over the batch). */
export type Risk = (scores: Value, target: Value) => Value

/** The default risk: binary cross-entropy on logits (targets in {0, 1}). */
const bce: Risk = (scores, target) => binaryCrossEntropyWithLogits(scores, target)

/** One environment's scores (logits or predictions) and targets. */
export interface EnvironmentBatch {
  readonly scores: TensorLike
  readonly target: TensorLike
}

/**
 * The IRMv1 penalty (Arjovsky et al., 2019): the squared norm of the gradient of the environment's risk with respect to
 * a dummy classifier scale w multiplying the scores, taken at w = 1. It is zero exactly when the scores are already an
 * optimal classifier for this environment. The `unbiased` form splits the batch in two and multiplies the halves'
 * gradients, an unbiased estimate of the squared gradient.
 *
 * @param scores - The scores (logits) of the representation's classifier `[N]` or `[N, K]`.
 * @param target - The targets, the shape the risk expects.
 * @param options - risk (default binary cross-entropy on logits; use `mse` for regression) and unbiased.
 * @returns The scalar penalty, differentiable in the scores (a gradient of a gradient).
 * @concept invariant-risk-minimisation
 * @example
 * ```ts
 * import { irmPenalty, mse } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * // mean (w s - y)^2 has derivative 2 mean (s - y) s at w = 1: here 2 * ((3 - 1) * 3 + (1 - 1) * 1) / 2 = 6.
 * item(irmPenalty([3, 1], [1, 1], { risk: mse })) // => 36
 * ```
 */
export function irmPenalty(
  scores: TensorLike,
  target: TensorLike,
  options: { readonly risk?: Risk; readonly unbiased?: boolean } = {},
): Value {
  const [s, y] = promoteFloat(scores, target) as [Value, Value]
  const risk = options.risk ?? bce
  const gradient = (a: Value, b: Value): Value => grad((w: Value) => risk(mul(a, w), b))(tensor(1))
  if (options.unbiased !== true) return square(gradient(s, y))
  const n = s.shape[0] ?? 0
  if (n < 2) throw new MdmlError('shape', 'irmPenalty: the unbiased estimate needs two samples')
  const half = Math.floor(n / 2)
  const part = (v: Value, lo: number, hi: number): Value => sliceAxis(v, 0, lo, hi)
  return mul(gradient(part(s, 0, half), part(y, 0, half)), gradient(part(s, half, 2 * half), part(y, half, 2 * half)))
}

/** The result of an invariance objective across environments. */
export interface InvariantObjective {
  /** The mean of the environments' risks. */
  readonly risk: Value
  /** The mean penalty across environments (IRM) or the variance of the risks (V-REx). */
  readonly penalty: Value
  /** risk + penaltyWeight * penalty. */
  readonly total: Value
}

/**
 * The IRMv1 objective: the mean of the environments' risks plus penaltyWeight times the mean of their
 * {@link irmPenalty}. A weight of zero is empirical risk minimisation over the pooled environments.
 *
 * @param environments - Scores and targets of each environment.
 * @param options - penaltyWeight (default 1), risk and unbiased as {@link irmPenalty}.
 * @returns The mean risk, the mean penalty and their weighted sum.
 * @example
 * ```ts
 * import { irmObjective, mse } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const env = { scores: [1, 2], target: [1, 2] }
 * item(irmObjective([env, env], { risk: mse, penaltyWeight: 10 }).total) // => 0
 * ```
 */
export function irmObjective(
  environments: readonly EnvironmentBatch[],
  options: { readonly penaltyWeight?: number; readonly risk?: Risk; readonly unbiased?: boolean } = {},
): InvariantObjective {
  const risk = options.risk ?? bce
  const parts = environments.map((e) => {
    const [s, y] = promoteFloat(e.scores, e.target) as [Value, Value]
    return {
      risk: risk(s, y),
      penalty: irmPenalty(s, y, { risk, ...(options.unbiased === undefined ? {} : { unbiased: options.unbiased }) }),
    }
  })
  const meanRisk = mean(
    stack(
      parts.map((p) => p.risk),
      0,
    ),
  )
  const meanPenalty = mean(
    stack(
      parts.map((p) => p.penalty),
      0,
    ),
  )
  return { risk: meanRisk, penalty: meanPenalty, total: add(meanRisk, mul(meanPenalty, options.penaltyWeight ?? 1)) }
}

/**
 * Risk extrapolation (V-REx; Krueger et al., 2021): the variance of the environments' risks, zero when the predictor
 * does equally well everywhere. Penalising it is a softer, convex relative of IRM.
 *
 * @param risks - The risk of each environment (scalars).
 * @returns The scalar variance (population, as the paper).
 * @example
 * ```ts
 * import { vrexPenalty } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(vrexPenalty([1, 3])) // => 1
 * ```
 */
export function vrexPenalty(risks: readonly TensorLike[]): Value {
  if (risks.length < 2) throw new MdmlError('value', 'vrexPenalty: need at least two environments')
  return variance(
    stack(
      risks.map((r) => promoteFloat(r)[0] as Value),
      0,
    ),
  )
}

/**
 * The V-REx objective: the mean risk plus penaltyWeight times the variance of the environments' risks.
 *
 * @param environments - Scores and targets of each environment.
 * @param options - penaltyWeight (default 1) and risk (default binary cross-entropy on logits).
 * @returns The mean risk, the variance and their weighted sum.
 * @example
 * ```ts
 * import { mse, vrexObjective } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const a = { scores: [0], target: [1] }
 * const b = { scores: [0], target: [3] }
 * item(vrexObjective([a, b], { risk: mse, penaltyWeight: 2 }).total) // => 5 + 2 * 16
 * ```
 */
export function vrexObjective(
  environments: readonly EnvironmentBatch[],
  options: { readonly penaltyWeight?: number; readonly risk?: Risk } = {},
): InvariantObjective {
  const risk = options.risk ?? bce
  const risks = environments.map((e) => risk(...(promoteFloat(e.scores, e.target) as [Value, Value])))
  const meanRisk = mean(stack(risks, 0))
  const penalty = vrexPenalty(risks)
  return { risk: meanRisk, penalty, total: add(meanRisk, mul(penalty, options.penaltyWeight ?? 1)) }
}

/** The covariance matrix of features `[N, D]` (rows are samples, divisor N - 1). */
function covariance(x: Value): Value {
  const n = x.shape[0] ?? 1
  const centred = sub(x, mean(x, 0, true))
  return div(matmul(transpose(centred), centred), n - 1)
}

/**
 * The CORAL loss (Sun and Saenko, 2016; Deep CORAL, 2016): the squared Frobenius distance between the source and
 * target feature covariances over 4 d^2, which aligns second-order statistics across domains. With `matchMean` the
 * squared distance of the means (over d) is added.
 *
 * @param source - Features of the source domain `[N, D]`.
 * @param target - Features of the target domain `[M, D]` (the sample counts may differ).
 * @param options - matchMean (default false).
 * @returns The scalar loss, differentiable in both feature sets.
 * @example
 * ```ts
 * import { coral } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(coral([[1], [-1]], [[0.5], [-0.5]])) // => 0.5625
 * ```
 */
export function coral(source: TensorLike, target: TensorLike, options: { readonly matchMean?: boolean } = {}): Value {
  const [s, t] = promoteFloat(source, target) as [Value, Value]
  const d = s.shape[1] ?? 1
  const covariancePart = div(sum(square(sub(covariance(s), covariance(t)))), 4 * d * d)
  if (options.matchMean !== true) return covariancePart
  return add(covariancePart, div(sum(square(sub(mean(s, 0), mean(t, 0)))), d))
}

/**
 * The mean MMD^2 over all pairs of environments' features: the distribution-matching penalty of domain-generalisation
 * methods (MMD-AAE and Deep Domain Confusion), zero when every environment's features have the same distribution as
 * seen by the kernel.
 *
 * @param features - The features of each environment `[N_e, D]` (at least two).
 * @param options - Passed to {@link mmd} (kernel and bandwidths).
 * @returns The scalar penalty.
 * @example
 * ```ts
 * import { mmdPenalty } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(mmdPenalty([[[0], [1]], [[0], [1]]], { bandwidths: [1] })) // => 0
 * ```
 */
export function mmdPenalty(features: readonly TensorLike[], options: Parameters<typeof mmd>[2] = {}): Value {
  if (features.length < 2) throw new MdmlError('value', 'mmdPenalty: need at least two environments')
  const pairs: Value[] = []
  for (let i = 0; i < features.length; i++)
    for (let j = i + 1; j < features.length; j++)
      pairs.push(mmd(features[i] as TensorLike, features[j] as TensorLike, options))
  return mean(stack(pairs, 0))
}

/**
 * The gradient of an environment's risk with respect to the parameters, kept differentiable (a graph through the
 * gradient), for the gradient-matching penalties.
 *
 * @param risk - The risk as a function of the parameters vector, a scalar.
 * @param params - The parameters `[P]`.
 * @returns The gradient `[P]`.
 * @example
 * ```ts
 * import { environmentGradient } from 'mdml-engine/losses'
 * import { square, sum, toArray, type Value } from 'mdml-engine/tensor'
 * toArray(environmentGradient((w: Value) => sum(square(w)), [1, 2])) // => [2, 4]
 * ```
 */
export function environmentGradient(risk: (params: Value) => Value, params: TensorLike): Value {
  return grad(risk)(promoteFloat(params)[0] as Value)
}

/**
 * Gradient agreement across environments: one minus the mean pairwise cosine of the environments' gradients ('cosine',
 * zero when all point the same way), or minus the mean pairwise inner product ('inner', the Fish criterion of Shi et al.,
 * 2022, whose minimum makes the environments' gradients agree and so improves each when it takes a step).
 *
 * @param gradients - The gradient of each environment's risk `[P]` (at least two).
 * @param options - kind ('cosine' or 'inner', default 'cosine') and eps for the norms (default 1e-12).
 * @returns The scalar loss, differentiable in the gradients (so in the parameters).
 * @example
 * ```ts
 * import { gradientAlignment } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(gradientAlignment([[1, 0], [0, 1]])) // => 1
 * ```
 */
export function gradientAlignment(
  gradients: readonly TensorLike[],
  options: { readonly kind?: 'cosine' | 'inner'; readonly eps?: number } = {},
): Value {
  if (gradients.length < 2) throw new MdmlError('value', 'gradientAlignment: need at least two environments')
  const eps = options.eps ?? 1e-12
  const g = gradients.map((v) => promoteFloat(v)[0] as Value)
  const pairs: Value[] = []
  for (let i = 0; i < g.length; i++)
    for (let j = i + 1; j < g.length; j++) {
      const [a, b] = [g[i] as Value, g[j] as Value]
      const inner = sum(mul(a, b))
      if (options.kind === 'inner') pairs.push(inner)
      else pairs.push(div(inner, mul(sqrt(add(sum(square(a)), eps)), sqrt(add(sum(square(b)), eps)))))
    }
  const m = mean(stack(pairs, 0))
  return options.kind === 'inner' ? mul(m, -1) : sub(1, m)
}

/**
 * The gradient of a per-sample risk with respect to the parameters for every sample of a batch, `[N, P]`, kept
 * differentiable: the input of the Fishr penalty. The risk takes the parameters and one sample (with a leading axis of
 * one) and returns a scalar.
 *
 * @param risk - The risk of one sample as a function of the parameters.
 * @param params - The parameters `[P]`.
 * @param inputs - Samples `[N, ...]`.
 * @param target - Targets `[N, ...]`.
 * @returns The per-sample gradients `[N, P]`.
 * @example
 * ```ts
 * import { perSampleGradients } from 'mdml-engine/losses'
 * import { mul, sub, sum, toArray, type Value } from 'mdml-engine/tensor'
 * // risk = (w x - y)^2: the gradient is 2 (w x - y) x.
 * const risk = (w: Value, x: Value, y: Value): Value => sum(mul(sub(mul(x, w), y), sub(mul(x, w), y)))
 * toArray(perSampleGradients(risk, [1], [[2], [3]], [[1], [1]])) // => [[4], [12]]
 * ```
 */
export function perSampleGradients(
  risk: (params: Value, input: Value, target: Value) => Value,
  params: TensorLike,
  inputs: TensorLike,
  target: TensorLike,
): Value {
  const [x, y] = promoteFloat(inputs, target) as [Value, Value]
  const n = x.shape[0] ?? 0
  const rows = Array.from({ length: n }, (_, i) =>
    grad((w: Value) => risk(w, sliceAxis(x, 0, i, i + 1), sliceAxis(y, 0, i, i + 1)))(promoteFloat(params)[0] as Value),
  )
  return stack(rows, 0)
}

/**
 * The Fishr penalty (Rame et al., 2022): match the environments' gradient covariances, in practice their diagonals, the
 * variance over samples of each parameter's gradient (which, for a log-likelihood, is the Fisher information diagonal):
 * the mean over environments of the squared distance of an environment's gradient variance to the average one.
 *
 * @param gradients - Per-sample gradients of each environment `[N_e, P]` (see {@link perSampleGradients}).
 * @returns The scalar penalty, differentiable in the gradients.
 * @example
 * ```ts
 * import { fishrPenalty } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * // Gradient variances 1 and 4 against their mean 2.5: 1.5^2 each.
 * item(fishrPenalty([[[1], [-1]], [[2], [-2]]])) // => 2.25
 * ```
 */
export function fishrPenalty(gradients: readonly TensorLike[]): Value {
  if (gradients.length < 2) throw new MdmlError('value', 'fishrPenalty: need at least two environments')
  const variances = gradients.map((g) => variance(promoteFloat(g)[0] as Value, 0))
  const average = mean(stack(variances, 0), 0)
  return mean(
    stack(
      variances.map((v) => sum(square(sub(v, average)))),
      0,
    ),
  )
}
