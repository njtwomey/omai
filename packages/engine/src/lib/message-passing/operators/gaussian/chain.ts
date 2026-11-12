/**
 * The Gaussian chain with a shared random precision: `x_t ~ N(a x_(t-1) + c . u + d, 1 / (scale tau + offset))` for the steps
 * `t = 1 .. n - 1` of a series of Gaussian variables `x_0 .. x_(n-1)` (the first one has a prior of its own), with `u` a few
 * Gaussian drivers that every step reads (the level an autoregression returns to) and one gamma precision `tau`, as ONE factor.
 * The scalar form of this chain (a `linear` factor and a `gaussian-gamma-precision` factor per step) sends each variable a message
 * from the cavities of its two neighbours taken as independent, which they are not once a driver or a precision is shared: with the
 * level `u` read by every step the beliefs of the level came out twice as wide as the exact ones, and with the shared precision the
 * sixty messages to it oscillated. Here the factor integrates the whole chain at once. Given the precision the chain and the
 * Gaussian cavities of all its variables are one Gaussian whose marginals a pass along the chain gives in time linear in `n` (a
 * tridiagonal elimination with the drivers as a border), and the precision, on a grid in its log, is the one non-Gaussian variable:
 * each variable's tilted moments are the mixture of those marginals over the grid, the precision's the moments of the grid. It is
 * the `gaussian-mv-precision-scale` of a dense vector, for a chain, and its messages, run in the order of a sweep, are the
 * Kalman smoother's wherever the precision is known. Because the precision is integrated against the cavities of all the states at
 * once, a state whose reading has an improper tail (a cavity with almost no precision and a mean far away) would let the precision run
 * to zero; the optional `ridge` gives such a state a weak prior at the chain's own estimate of it (`weakPriors`).
 */

import { OmaiError } from '../../../base/index.ts'
import { arg, errorText, keyOf, memoised } from '../../core/_messages.ts'
import { etaOf, vectorParam } from '../../core/args.ts'
import { defineOperator, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gammaShapeRate } from '../../core/families.ts'
import { param } from '../../core/params.ts'
import type { FactorOperator, OperatorContext, Params, Port } from '../../core/types.ts'
import { centredMoments } from '../../numerics/_moments.ts'
import { gammaLogWindow, tiltGrid, windowProfiles } from '../../numerics/_windows.ts'

/** A small symmetric positive-definite matrix (the drivers' Schur complement) factored by Cholesky on plain arrays: its log determinant, solve and inverse. */
interface Small {
  readonly logDet: number
  solve(b: Float64Array): Float64Array
  inverse(): Float64Array
}

/** The Cholesky factor of a small row-major matrix, or `undefined` when it is not positive definite. */
function small(a: Float64Array, k: number): Small | undefined {
  // Exempt from the tensor rule, on purpose: a small Cholesky on typed buffers, called for every node of the log-precision grid.
  const l = new Float64Array(k * k)
  let logDet = 0
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a[i * k + j] as number
      for (let m = 0; m < j; m++) sum -= (l[i * k + m] as number) * (l[j * k + m] as number)
      if (i === j) {
        if (!(sum > 0) || !Number.isFinite(sum)) return undefined
        l[i * k + i] = Math.sqrt(sum)
        logDet += Math.log(sum)
      } else l[i * k + j] = sum / (l[j * k + j] as number)
    }
  }
  const solve = (b: Float64Array): Float64Array => {
    const y = new Float64Array(k)
    for (let i = 0; i < k; i++) {
      let sum = b[i] as number
      for (let m = 0; m < i; m++) sum -= (l[i * k + m] as number) * (y[m] as number)
      y[i] = sum / (l[i * k + i] as number)
    }
    const x = new Float64Array(k)
    for (let i = k - 1; i >= 0; i--) {
      let sum = y[i] as number
      for (let m = i + 1; m < k; m++) sum -= (l[m * k + i] as number) * (x[m] as number)
      x[i] = sum / (l[i * k + i] as number)
    }
    return x
  }
  return {
    logDet,
    solve,
    inverse: () => {
      const inv = new Float64Array(k * k)
      for (let c = 0; c < k; c++) {
        const e = new Float64Array(k)
        e[c] = 1
        inv.set(solve(e), c * k)
      }
      return inv
    },
  }
}

/** The nodes of each grid over the log precision (`tiltGrid` lays up to three, zooming onto the mass). */
const POINTS = 161

/** What the chain is: the number of states, the step `x_t = a x_(t-1) + sum_k c_k u_k + d + noise`, the precision `scale tau + offset`. */
export interface ChainSpec {
  readonly n: number
  readonly a: number
  readonly drive: readonly number[]
  readonly intercept: number
  readonly scale: number
  readonly offset: number
  /** The weak prior that a state with an uninformative cavity carries inside the factor, in precision (see `chainTilt`; zero for none). */
  readonly ridge: number
}

/** The chain of the parameters, checked. */
export function chainSpec(params: Params): ChainSpec {
  const n = params['length']
  if (typeof n !== 'number' || !Number.isInteger(n) || n < 2)
    throw new OmaiError('value', 'gaussian-chain: length must be an integer of at least 2')
  const num = (name: string, fallback: number): number => {
    const v = params[name]
    return typeof v === 'number' ? v : fallback
  }
  const drive = params['drive'] === undefined ? [] : vectorParam(params, 'drive')
  const scale = num('scale', 1)
  const offset = num('offset', 0)
  if (!(scale >= 0 && offset >= 0 && scale + offset > 0))
    throw new OmaiError('value', 'gaussian-chain: the precision needs a non-negative scale and offset, not both zero')
  const ridge = num('ridge', 0)
  if (!(ridge >= 0)) throw new OmaiError('value', 'gaussian-chain: the ridge must not be negative')
  return { n, a: num('coefficient', 1), drive, intercept: num('intercept', 0), scale, offset, ridge }
}

/** The ports of a chain: the precision, then its `n` states and its drivers. */
function chainPorts(params: Params): readonly Port[] {
  const spec = chainSpec(params)
  return [
    { name: 'tau', accepts: ['gamma'] },
    ...Array.from({ length: spec.n }, (_, t): Port => ({ name: `x${t}`, accepts: ['gaussian'] })),
    ...spec.drive.map((_, k): Port => ({ name: `u${k}`, accepts: ['gaussian'] })),
  ]
}

/** What one precision gives: the log of the integral of the chain against the cavities, and the mean and variance of every variable. */
export interface ChainAt {
  readonly logZ: number
  /** The means of `x_0 .. x_(n-1)` then the drivers. */
  readonly mean: Float64Array
  /** The variances, in the same order. */
  readonly variance: Float64Array
}

/**
 * The chain integrated against the Gaussian cavities `exp(h_j x_j - lambda_j x_j^2 / 2)` of its variables (the states, then the
 * drivers), at the precision `lambda` of the steps: the log of the integral and, when asked, the marginal mean and variance of each
 * variable. Returns `undefined` where the joint precision is not positive definite (a flat cavity on every variable the steps leave
 * free, a negative one that the steps do not cover).
 *
 * The joint precision is tridiagonal on the states with a border for the drivers. A pass down the chain eliminates the states
 * (pivots, reduced information and reduced border), the drivers are solved from what is left (a matrix of the order of their number),
 * and the states come back up; the marginal variances are the diagonal of the inverse of the tridiagonal block (the pivots of a pass
 * down and of a pass up) plus the drivers' covariance carried through the border.
 *
 * @param spec - The chain.
 * @param h - The cavities' linear natural parameters, states first.
 * @param j - The cavities' precisions.
 * @param lambda - The precision of the steps.
 * @param moments - Whether to compute the marginals (the integral alone is cheaper).
 * @returns The integral and the marginals, or `undefined`.
 */
export function chainAt(
  spec: ChainSpec,
  h: ArrayLike<number>,
  j: ArrayLike<number>,
  lambda: number,
  moments: boolean,
): ChainAt | undefined {
  const { n, a, intercept: d } = spec
  const c = spec.drive
  const K = c.length
  const steps = n - 1
  const O = -lambda * a
  const dp = new Float64Array(n)
  const rp = new Float64Array(n)
  const wp = new Float64Array(n * K)
  // The elimination of the states, top to bottom.
  for (let t = 0; t < n; t++) {
    const first = t >= 1
    const last = t <= n - 2
    const diag = (j[t] ?? 0) + (first ? lambda : 0) + (last ? lambda * a * a : 0)
    const info = (h[t] ?? 0) + lambda * d * ((first ? 1 : 0) - (last ? a : 0))
    if (t === 0) {
      dp[0] = diag
      rp[0] = info
      for (let k = 0; k < K; k++)
        wp[k] = (last ? lambda * a * (c[k] as number) : 0) - (first ? lambda * (c[k] as number) : 0)
    } else {
      const f = O / (dp[t - 1] as number)
      dp[t] = diag - f * O
      rp[t] = info - f * (rp[t - 1] as number)
      for (let k = 0; k < K; k++) {
        const border = (last ? lambda * a * (c[k] as number) : 0) - (first ? lambda * (c[k] as number) : 0)
        wp[t * K + k] = border - f * (wp[(t - 1) * K + k] as number)
      }
    }
    if (!((dp[t] as number) > 0)) return undefined
  }
  // The drivers from what the states leave: the Schur complement and its information.
  const S = new Float64Array(K * K)
  const ru = new Float64Array(K)
  for (let k = 0; k < K; k++) {
    for (let l = 0; l < K; l++)
      S[k * K + l] = (k === l ? (j[n + k] ?? 0) : 0) + lambda * steps * (c[k] as number) * (c[l] as number)
    ru[k] = (h[n + k] ?? 0) - lambda * d * (c[k] as number) * steps
  }
  for (let t = 0; t < n; t++) {
    const w = dp[t] as number
    const r = rp[t] as number
    for (let k = 0; k < K; k++) {
      ru[k] = (ru[k] as number) - ((wp[t * K + k] as number) * r) / w
      for (let l = 0; l < K; l++)
        S[k * K + l] = (S[k * K + l] as number) - ((wp[t * K + k] as number) * (wp[t * K + l] as number)) / w
    }
  }
  let logDet = 0
  let quad = 0
  for (let t = 0; t < n; t++) {
    logDet += Math.log(dp[t] as number)
    quad += (rp[t] as number) ** 2 / (dp[t] as number)
  }
  const factor = K === 0 ? undefined : small(S, K)
  if (K > 0 && factor === undefined) return undefined
  const uMean = factor === undefined ? new Float64Array(0) : factor.solve(ru)
  if (factor !== undefined) {
    logDet += factor.logDet
    for (let k = 0; k < K; k++) quad += (ru[k] as number) * (uMean[k] as number)
  }
  const size = n + K
  const logZ =
    0.5 * steps * (Math.log(lambda) - Math.log(2 * Math.PI)) -
    0.5 * lambda * steps * d * d +
    0.5 * (size * Math.log(2 * Math.PI) - logDet + quad)
  if (!moments) return { logZ, mean: new Float64Array(0), variance: new Float64Array(0) }
  const mean = new Float64Array(size)
  const variance = new Float64Array(size)
  // The states back up: row t of the eliminated system is dp x_t + O x_(t+1) + wp . u = rp.
  for (let t = n - 1; t >= 0; t--) {
    let rest = rp[t] as number
    for (let k = 0; k < K; k++) rest -= (wp[t * K + k] as number) * (uMean[k] as number)
    if (t < n - 1) rest -= O * (mean[t + 1] as number)
    mean[t] = rest / (dp[t] as number)
  }
  // The inverse of the tridiagonal block: its diagonal from the pivots of both passes, and its product with the border.
  const dq = new Float64Array(n)
  for (let t = n - 1; t >= 0; t--) {
    const diag = (j[t] ?? 0) + (t >= 1 ? lambda : 0) + (t <= n - 2 ? lambda * a * a : 0)
    dq[t] = t === n - 1 ? diag : diag - (O * O) / (dq[t + 1] as number)
  }
  const g = new Float64Array(n * K)
  for (let t = n - 1; t >= 0; t--)
    for (let k = 0; k < K; k++) {
      let v = wp[t * K + k] as number
      if (t < n - 1) v -= O * (g[(t + 1) * K + k] as number)
      g[t * K + k] = v / (dp[t] as number)
    }
  const cov = factor === undefined ? new Float64Array(0) : factor.inverse()
  for (let t = 0; t < n; t++) {
    const diag = (j[t] ?? 0) + (t >= 1 ? lambda : 0) + (t <= n - 2 ? lambda * a * a : 0)
    let v = 1 / ((dp[t] as number) + (dq[t] as number) - diag)
    for (let k = 0; k < K; k++)
      for (let l = 0; l < K; l++) v += (g[t * K + k] as number) * (cov[k * K + l] as number) * (g[t * K + l] as number)
    variance[t] = v
  }
  for (let k = 0; k < K; k++) {
    mean[n + k] = uMean[k] as number
    variance[n + k] = cov[k * K + k] as number
  }
  return { logZ, mean, variance }
}

/** The tilted distribution of the chain against its cavities: the log integral, the moments of the precision, and the moments of each variable. */
export interface ChainTilt {
  readonly logZ: number
  /** `E log tau`. */
  readonly logTau: number
  /** `E tau`. */
  readonly tau: number
  /** The tilted mean of each variable (the states, then the drivers). */
  readonly mean: Float64Array
  /** The tilted variance of each variable. */
  readonly variance: Float64Array
}

/**
 * The weak priors of a chain's regularisation (`ridge`): to each state a Gaussian of precision `ridge exp(-j / ridge)` (the whole ridge
 * for a flat cavity of precision `j`, a millionth of it for a cavity ten times as precise), centred at the state's mean under the
 * chain at the precision `lambda0` of the gamma cavity's mean. A reading with an improper tail (the log-variance likelihood of a return
 * near zero, an exponential in the state with no curvature) sends its state a cavity with almost no precision and a mean far away; the
 * integral over the chain's precision would then let the precision run to zero, where the chain lets that state go. The prior holds
 * the state where the rest of the chain puts it, so it adds precision to the states that have none and moves nothing that has some
 * and, centred at the state's own mean, no mean at a settled state. Returns the cavities with the priors in them and the log of the
 * constant `-ridge_t c^2 / 2` that turns the natural form `exp(ridge_t c x - ridge_t x^2 / 2)` of those priors into `exp(-ridge_t (x - c)^2 / 2)`
 * (so the integral of the regularised chain is the integral against the priors themselves), or `undefined` for no ridge.
 */
function weakPriors(
  spec: ChainSpec,
  h: ArrayLike<number>,
  j: ArrayLike<number>,
  lambda0: number,
): { readonly h: Float64Array; readonly j: Float64Array; readonly logZ: number } | undefined {
  if (!(spec.ridge > 0)) return undefined
  const start = chainAt(spec, h, j, lambda0, true)
  const size = spec.n + spec.drive.length
  const nh = Float64Array.from({ length: size }, (_, p) => h[p] ?? 0)
  const nj = Float64Array.from({ length: size }, (_, p) => j[p] ?? 0)
  let logZ = 0
  for (let t = 0; t < spec.n; t++) {
    const ridge = spec.ridge * Math.exp(-Math.max(j[t] ?? 0, 0) / spec.ridge)
    const centre = start === undefined ? 0 : (start.mean[t] as number)
    nh[t] = (nh[t] as number) + ridge * centre
    nj[t] = (nj[t] as number) + ridge
    logZ -= 0.5 * ridge * centre * centre
  }
  return { h: nh, j: nj, logZ }
}

/**
 * The tilted distribution of the chain against the cavities of its variables and of its precision `tau`, on a grid in `log tau`
 * zoomed onto where the mass is (`tiltGrid`): the weight of a node is the gamma cavity times the integral of the chain at that
 * precision (`chainAt`), and each variable's tilted mean and variance are the mixture of the chain's marginals over the nodes.
 *
 * @param spec - The chain.
 * @param h - The cavities' linear natural parameters of the states and then the drivers.
 * @param j - Their precisions.
 * @param tau - The natural parameters of the gamma cavity of the precision.
 * @returns The tilt.
 * @throws ImproperMessageError when the gamma cavity has no density or no precision makes the chain proper.
 */
export function chainTilt(
  spec: ChainSpec,
  h: ArrayLike<number>,
  j: ArrayLike<number>,
  tau: readonly number[],
): ChainTilt {
  const { shape, rate } = gammaShapeRate(tau)
  if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
    throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
  const precision = (t: number): number => spec.scale * Math.exp(t) + spec.offset
  const weak = weakPriors(spec, h, j, precision(Math.log(shape / rate)))
  const reg = weak === undefined ? { h, j, logZ: 0 } : weak
  const grid = tiltGrid(gammaLogWindow(shape, rate, windowProfiles.logTilt), POINTS, (t) => {
    const at = chainAt(spec, reg.h, reg.j, precision(t), false)
    return at === undefined ? Number.NEGATIVE_INFINITY : shape * t - rate * Math.exp(t) + at.logZ
  })
  if (!Number.isFinite(grid.top)) throw new ImproperMessageError('gaussian-chain: no precision makes the chain proper')
  const size = spec.n + spec.drive.length
  const means: Float64Array[] = []
  const variances: Float64Array[] = []
  const live: number[] = []
  const weights: number[] = []
  for (let i = 0; i < grid.t.length; i++) {
    const w = grid.weights[i] as number
    if (!(w > 1e-14)) continue
    const at = chainAt(spec, reg.h, reg.j, precision(grid.t[i] as number), true)
    if (at === undefined) continue
    means.push(at.mean)
    variances.push(at.variance)
    live.push(grid.t[i] as number)
    weights.push(w)
  }
  const total = weights.reduce((s, w) => s + w, 0)
  const mean = new Float64Array(size)
  const variance = new Float64Array(size)
  const column = new Float64Array(weights.length)
  for (let p = 0; p < size; p++) {
    for (let i = 0; i < weights.length; i++) column[i] = (means[i] as Float64Array)[p] as number
    const m = centredMoments(weights, total, column)
    let within = 0
    for (let i = 0; i < weights.length; i++)
      within += (weights[i] as number) * ((variances[i] as Float64Array)[p] as number)
    mean[p] = m.mean
    variance[p] = within / total + m.variance
  }
  const t = centredMoments(weights, total, live)
  let eTau = 0
  for (let i = 0; i < weights.length; i++) eTau += (weights[i] as number) * Math.exp(live[i] as number)
  return {
    logZ: grid.top + Math.log(grid.total * grid.step) + reg.logZ,
    logTau: t.mean,
    tau: eTau / total,
    mean,
    variance,
  }
}

/** The tilt of a call's cavities, kept in the engine's cache: the ports of one sweep's update of the factor share one pass. */
function tiltOf(c: OperatorContext): ChainTilt {
  const spec = chainSpec(c.params)
  const size = spec.n + spec.drive.length
  const h = new Float64Array(size)
  const j = new Float64Array(size)
  for (let p = 0; p < size; p++) {
    const a = arg(c, 1 + p)
    // The compiler refuses an observed variable at these ports (they are not `constant`); a hand-built call that shows one is a mistake
    // of the model, not an improper message that the engine would repair and skip.
    if (a.value !== undefined)
      throw new ParameterError(
        `gaussian-chain: the state or driver at port ${p} cannot be observed (its observation belongs in the cavities, as the factor that reads it)`,
      )
    const eta = etaOf(a)
    h[p] = eta[0] ?? 0
    j[p] = eta[1] ?? 0
  }
  const tau = etaOf(arg(c, 0))
  return memoised(
    c,
    `gaussian-chain:${keyOf([spec.n, spec.a, spec.drive, spec.intercept, spec.scale, spec.offset, spec.ridge, h, j, tau])}`,
    () => chainTilt(spec, h, j, tau),
  )
}

/**
 * A chain of Gaussian variables with Gaussian steps of one shared gamma precision, `x_t ~ N(a x_(t-1) + c . u + d, 1 / (scale tau +
 * offset))` for `t = 1 .. length - 1`, and drivers `u` that every step reads: an autoregression around a random level, a random
 * walk, a stochastic volatility log-variance. The ports are the precision, then the `length` states, then the drivers (`drive` lists their
 * coefficients). Expectation propagation integrates the chain against the cavities in one pass (see the file's
 * summary): each state and driver gets the Gaussian with the tilted mean and variance, the precision the gamma with the tilted `E log
 * tau` and `E tau`, and the log evidence is the integral. The first state's prior is another factor. `ridge` (default none) is the
 * precision of the weak prior that a state with an uninformative cavity carries (`weakPriors`); with `ridge` the messages are those of
 * the chain times those priors, and so is the log evidence (the integral of the regularised chain against the cavities, an
 * approximation of the declared factor's, which has none when a state's cavity is improper: equal to it for no ridge). No variational
 * rule (the mean field of a chain is the scalar one).
 *
 * @concept factor-operators
 * @index structure A chain of Gaussian steps with shared drivers and a shared gamma precision, one factor.
 * @example
 * ```ts
 * import { gamma, gammaEta, gaussian, gaussianChain, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const x = [0, 1, 2].map((m) => messageArg(gaussian, gaussianEta(m, 1)))
 * const args = [messageArg(gamma, gammaEta(3, 1)), ...x]
 * const m = gaussianChain.ep?.({ mode: 'ep', to: 2, args, params: { length: 3, coefficient: 0.9 }, power: 1 }).message
 * m?.length // => 2
 * ```
 */
export const gaussianChain: FactorOperator = defineOperator({
  name: 'gaussian-chain',
  description: 'x_t ~ N(a x_(t-1) + c . u + d, 1 / (scale tau + offset)) down a chain of Gaussians, one gamma tau',
  ports: chainPorts,
  params: {
    length: param.integer({ min: 2, example: 3 }),
    coefficient: param.number({ optional: true }),
    drive: param.vector({ optional: true }),
    intercept: param.number({ optional: true }),
    scale: param.nonNegative({ optional: true }),
    offset: param.nonNegative({ optional: true }),
    ridge: param.nonNegative({ optional: true }),
  },
  check: (params) => {
    try {
      chainSpec(params)
      return undefined
    } catch (e) {
      return errorText(e)
    }
  },
  rules: {
    ep: siteRule((c) => {
      const tilt = tiltOf(c)
      if (c.to === 0) return { moments: [tilt.logTau, tilt.tau] }
      return { mean: tilt.mean[c.to - 1] as number, variance: tilt.variance[c.to - 1] as number }
    }),
  },
  evidence: { ep: (c) => tiltOf(c).logZ },
  // The precision's own cavity must be a gamma density.
  traits: { readsTarget: (to) => to === 0 },
  noRule: {
    vmp: 'the variational form of a chain is the scalar one: a linear factor and a gaussian-gamma-precision factor per step',
  },
})
