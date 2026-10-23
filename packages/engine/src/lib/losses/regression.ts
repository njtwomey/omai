import { WhiteboxError } from '../base/index.ts'
import { logaddexp, xlogy } from '../numerics/special/index.ts'
import {
  abs,
  add,
  div,
  exp,
  greater,
  less,
  log,
  maximum,
  mul,
  pow,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { type LossOptions, reduce } from './_reduce.ts'

const pair = (input: TensorLike, target: TensorLike): [Value, Value] => promoteFloat(input, target) as [Value, Value]

/**
 * Mean squared error (torch.nn.functional.mse_loss).
 *
 * @param input - Predictions.
 * @param target - Targets.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @concept losses
 * @example
 * ```ts
 * import { mse } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(mse([1, 2], [1, 4])) // => 2
 * ```
 */
export function mse(input: TensorLike, target: TensorLike, options: LossOptions = {}): Value {
  const [x, y] = pair(input, target)
  return reduce(square(sub(x, y)), options)
}

/**
 * Mean absolute error (torch.nn.functional.l1_loss).
 *
 * @param input - Predictions.
 * @param target - Targets.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable away from zero residuals.
 * @example
 * ```ts
 * import { mae } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(mae([1, 2], [1, 4])) // => 1
 * ```
 */
export function mae(input: TensorLike, target: TensorLike, options: LossOptions = {}): Value {
  const [x, y] = pair(input, target)
  return reduce(abs(sub(x, y)), options)
}

/**
 * Huber loss: quadratic for |r| <= delta, linear beyond (torch.nn.functional.huber_loss).
 *
 * @param input - Predictions.
 * @param target - Targets.
 * @param options - Delta (default 1), reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { huber } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(huber([0], [3], { delta: 1 })) // => 2.5
 * ```
 */
export function huber(
  input: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly delta?: number } = {},
): Value {
  const [x, y] = pair(input, target)
  const delta = options.delta ?? 1
  const r = abs(sub(x, y))
  const quadratic = mul(square(r), 0.5)
  const linear = mul(sub(r, 0.5 * delta), delta)
  return reduce(where(greater(r, delta), linear, quadratic), options)
}

/**
 * log(cosh(r)), computed as |r| + softplus(-2|r|) - log 2 to avoid overflow.
 *
 * @param input - Predictions.
 * @param target - Targets.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { logCosh } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(logCosh([0], [0])) // => 0
 * ```
 */
export function logCosh(input: TensorLike, target: TensorLike, options: LossOptions = {}): Value {
  const [x, y] = pair(input, target)
  const r = abs(sub(x, y))
  return reduce(sub(add(r, logaddexp(0, mul(r, -2))), Math.LN2), options)
}

/**
 * Pinball (quantile) loss for quantile level tau: max(tau r, (tau - 1) r) with r = target - input.
 *
 * @param input - Predicted quantiles.
 * @param target - Targets.
 * @param options - The level tau (default 0.5), reduction and weights.
 * @returns The loss, differentiable away from zero residuals.
 * @example
 * ```ts
 * import { pinball } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(pinball([0], [2], { tau: 0.9 })) // => 1.8
 * ```
 */
export function pinball(
  input: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly tau?: number } = {},
): Value {
  const [x, y] = pair(input, target)
  const tau = options.tau ?? 0.5
  const r = sub(y, x)
  return reduce(maximum(mul(r, tau), mul(r, tau - 1)), options)
}

/**
 * Poisson negative log-likelihood (torch.nn.functional.poisson_nll_loss): exp(input) - target input with log-rate
 * inputs, or input - target log(input + eps) with rates; `full` adds the Stirling term of log(target!).
 *
 * @param input - Log-rates (or rates when `logInput` is false).
 * @param target - Counts.
 * @param options - logInput (default true), full (default false), eps, reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { poissonNll } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(poissonNll([0], [1])) // => 1
 * ```
 */
export function poissonNll(
  input: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly logInput?: boolean; readonly full?: boolean; readonly eps?: number } = {},
): Value {
  const [x, y] = pair(input, target)
  let terms = (options.logInput ?? true) ? sub(exp(x), mul(y, x)) : sub(x, mul(y, log(add(x, options.eps ?? 1e-8))))
  if (options.full === true) {
    // torch adds y log y - y + 0.5 log(2 pi y) where y > 1.
    const stirling = add(sub(mul(y, log(y)), y), mul(log(mul(y, 2 * Math.PI)), 0.5))
    terms = add(terms, where(greater(y, 1), stirling, 0))
  }
  return reduce(terms, options)
}

/**
 * Gaussian negative log-likelihood with predicted variances (torch.nn.functional.gaussian_nll_loss): 0.5 (log
 * max(var, eps) + (input - target)^2 / max(var, eps)), plus 0.5 log(2 pi) when `full`.
 *
 * @param input - Predicted means.
 * @param target - Targets.
 * @param variance - Predicted variances (positive).
 * @param options - full, eps (default 1e-6), reduction and weights.
 * @returns The loss, differentiable in means and variances.
 * @example
 * ```ts
 * import { gaussianNll } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(gaussianNll([0], [0], [1])) // => 0
 * ```
 */
export function gaussianNll(
  input: TensorLike,
  target: TensorLike,
  variance: TensorLike,
  options: LossOptions & { readonly full?: boolean; readonly eps?: number } = {},
): Value {
  const [x, y, v] = promoteFloat(input, target, variance) as [Value, Value, Value]
  const clamped = maximum(v, options.eps ?? 1e-6)
  let terms = mul(add(log(clamped), div(square(sub(x, y)), clamped)), 0.5)
  if (options.full === true) terms = add(terms, 0.5 * Math.log(2 * Math.PI))
  return reduce(terms, options)
}

/**
 * Smooth L1 loss: 0.5 r^2 / beta for |r| < beta, |r| - 0.5 beta beyond (torch.nn.functional.smooth_l1_loss). It equals
 * {@link huber} divided by beta.
 *
 * @param input - Predictions.
 * @param target - Targets.
 * @param options - Beta (default 1; 0 gives the absolute error), reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { smoothL1 } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(smoothL1([0], [3], { beta: 2 })) // => 2
 * ```
 */
export function smoothL1(
  input: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly beta?: number } = {},
): Value {
  const [x, y] = pair(input, target)
  const beta = options.beta ?? 1
  const r = abs(sub(x, y))
  if (beta === 0) return reduce(r, options)
  return reduce(where(less(r, beta), div(mul(square(r), 0.5), beta), sub(r, 0.5 * beta)), options)
}

/**
 * Expectile loss for level tau: |tau - 1(r < 0)| r^2 with r = target - input, the squared-error analogue of
 * {@link pinball} (its minimiser is the tau-expectile; tau = 0.5 gives half the squared error).
 *
 * @param input - Predicted expectiles.
 * @param target - Targets.
 * @param options - The level tau (default 0.5), reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { expectile } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(expectile([0], [2], { tau: 0.75 })) // => 3
 * ```
 */
export function expectile(
  input: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly tau?: number } = {},
): Value {
  const [x, y] = pair(input, target)
  const tau = options.tau ?? 0.5
  const r = sub(y, x)
  return reduce(mul(where(less(r, 0), 1 - tau, tau), square(r)), options)
}

/**
 * Tweedie unit deviance for power p (sklearn.metrics.mean_tweedie_deviance, elementwise): p = 0 is the squared error,
 * p = 1 Poisson, p = 2 gamma, 1 < p < 2 compound Poisson-gamma; p in (0, 1) has no distribution and is rejected. With
 * `logInput` the input is log(mu), the usual parameterisation for training (as LightGBM's tweedie objective).
 *
 * @param input - Predicted means mu (positive), or log-means with `logInput`.
 * @param target - Targets (non-negative for p < 2, positive for p >= 2).
 * @param options - power (default 1.5), logInput (default false), reduction and weights.
 * @returns The loss, differentiable in the input.
 * @throws WhiteboxError with code `value` for a power in (0, 1).
 * @example
 * ```ts
 * import { tweedieDeviance } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(tweedieDeviance([2], [2], { power: 1.5 })) // => 0
 * ```
 */
export function tweedieDeviance(
  input: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly power?: number; readonly logInput?: boolean } = {},
): Value {
  const [x, y] = pair(input, target)
  const p = options.power ?? 1.5
  if (p > 0 && p < 1) throw new WhiteboxError('value', `tweedieDeviance: power ${p} is not in (-inf, 0] or [1, inf)`)
  const mu = options.logInput === true ? exp(x) : x
  let terms: Value
  if (p === 0) terms = square(sub(y, mu))
  else if (p === 1) terms = mul(add(sub(xlogy(y, div(y, mu)), y), mu), 2)
  else if (p === 2) terms = mul(sub(add(log(div(mu, y)), div(y, mu)), 1), 2)
  else {
    const first = div(pow(maximum(y, 0), 2 - p), (1 - p) * (2 - p))
    const second = div(mul(y, pow(mu, 1 - p)), 1 - p)
    const third = div(pow(mu, 2 - p), 2 - p)
    terms = mul(add(sub(first, second), third), 2)
  }
  return reduce(terms, options)
}
