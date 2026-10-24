import { valueAndGrad } from '../autodiff/index.ts'
import { type Json, WhiteboxError } from '../base/index.ts'
import { ddot, dgemv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal, subsample } from '../random/index.ts'
import { add, asTensor, fromStorage, mul, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { inverse, lowerCholesky, squareMatrix, vectorOf } from './_linear.ts'
import { acceptWith, checkStepSize } from './_mcmc.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import { medianOf } from './diagnostics/_draws.ts'
import { metropolisAcceptance } from './metropolis.ts'
import type { LogDensity } from './target.ts'

/** The state of the Langevin kernels: the position with its log density and gradient. */
export interface LangevinState extends MCMCState {
  readonly grad: Float64Array
}

/** Options of {@link malaKernel} and {@link ulaKernel}. */
export interface LangevinOptions {
  /** The step size epsilon of x' = x + eps A grad log pi(x) + sqrt(2 eps) A^1/2 z (positive). */
  readonly stepSize: number
  /** The preconditioner A (an inverse mass matrix): a number, a diagonal or a `[dim, dim]` matrix (default I). */
  readonly preconditioner?: TensorLike | number
}

interface Preconditioner {
  readonly apply: (v: Float64Array) => Float64Array
  readonly sqrtApply: (z: Float64Array) => Float64Array
  /** d^T A^-1 d. */
  readonly quadInverse: (d: Float64Array) => number
}

function preconditioner(a: TensorLike | number | undefined, dim: number, name: string): Preconditioner {
  if (a !== undefined && typeof a !== 'number' && (asTensor(a) as Tensor).shape.length === 2) {
    const m = squareMatrix(a, dim, name)
    const l = lowerCholesky(m, dim, name)
    const inv = inverse(m, dim)
    return {
      apply: (v) => dgemv(m, dim, dim, v),
      sqrtApply: (z) => dgemv(l, dim, dim, z),
      quadInverse: (d) => ddot(d, dgemv(inv, dim, dim, d)),
    }
  }
  const diag = vectorOf(a ?? 1, dim, name)
  if (!diag.every((v) => v > 0 && Number.isFinite(v))) {
    throw new WhiteboxError('value', `${name}: the preconditioner must be positive definite`)
  }
  return {
    apply: (v) => v.map((x, i) => x * (diag[i] ?? 1)),
    sqrtApply: (z) => z.map((x, i) => x * Math.sqrt(diag[i] ?? 1)),
    quadInverse: (d) => {
      let s = 0
      for (let i = 0; i < d.length; i++) s += ((d[i] ?? 0) * (d[i] ?? 0)) / (diag[i] ?? 1)
      return s
    },
  }
}

function langevinState(target: LogDensity, position: ArrayLike<number>, name: string): LangevinState {
  if (!target.hasGradient) throw new WhiteboxError('value', `${name}: the target has no gradient`)
  if (position.length !== target.dim) {
    throw new WhiteboxError('shape', `${name}: the position has ${position.length} values, need ${target.dim}`)
  }
  const x = Float64Array.from(position)
  const r = target.valueAndGrad(x)
  return { position: x, logDensity: r.value, grad: r.grad }
}

/** The overdamped Langevin proposal x + eps A g + sqrt(2 eps) A^1/2 z. */
function langevinMove(s: LangevinState, eps: number, pre: Preconditioner, z: Float64Array): Float64Array {
  const drift = pre.apply(s.grad)
  const noise = pre.sqrtApply(z)
  const c = Math.sqrt(2 * eps)
  return s.position.map((x, i) => x + eps * (drift[i] ?? 0) + c * (noise[i] ?? 0))
}

/** log q(to | from) up to a constant: -(to - from - eps A g(from))^T A^-1 (...) / (4 eps). */
function logTransition(to: Float64Array, from: LangevinState, eps: number, pre: Preconditioner): number {
  const drift = pre.apply(from.grad)
  const d = to.map((v, i) => v - (from.position[i] ?? 0) - eps * (drift[i] ?? 0))
  return -pre.quadInverse(d) / (4 * eps)
}

/**
 * The Metropolis-adjusted Langevin algorithm (Roberts and Tweedie 1996; BlackJAX `mala`): propose a step of the
 * discretised Langevin diffusion, x' = x + eps A grad log pi(x) + sqrt(2 eps) A^1/2 z, and accept with the
 * Metropolis-Hastings ratio of the asymmetric Gaussian proposal, so the target is exactly invariant. The optimal
 * acceptance rate in high dimension is about 0.574.
 *
 * @param target - The target (with a gradient).
 * @param options - Step size and preconditioner.
 * @returns The kernel.
 * @throws WhiteboxError with code `value` for a target without a gradient, a non-positive step size or a
 *   preconditioner that is not positive definite.
 * @concept langevin-monte-carlo
 * @reference G. O. Roberts and R. L. Tweedie, Exponential convergence of Langevin distributions and their discrete
 *   approximations (Bernoulli 1996).
 * @example
 * ```ts
 * import { malaKernel, numericTarget } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const k = malaKernel(t, { stepSize: 0.1 })
 * k.step(key(0), k.init([0.2])).info.acceptanceRate > 0.9 // => true
 * ```
 */
export function malaKernel(target: LogDensity, options: LangevinOptions): MCMCKernel<LangevinState> {
  const eps = options.stepSize
  checkStepSize(eps, 'malaKernel')
  const pre = preconditioner(options.preconditioner, target.dim, 'malaKernel')
  return {
    name: 'mala',
    dim: target.dim,
    config: { dim: target.dim, stepSize: eps },
    init: (position) => langevinState(target, position, 'malaKernel'),
    step: (key, state) => {
      const proposal = langevinMove(state, eps, pre, normal(foldIn(key, 0), [target.dim]).data)
      const r = target.valueAndGrad(proposal)
      const next: LangevinState = { position: proposal, logDensity: r.value, grad: r.grad }
      const alpha = Number.isFinite(r.value)
        ? metropolisAcceptance(
            state.logDensity,
            r.value,
            logTransition(proposal, state, eps, pre),
            logTransition(state.position, next, eps, pre),
          )
        : 0
      const accepted = acceptWith(foldIn(key, 1), alpha)
      return { state: accepted ? next : state, info: { acceptanceRate: alpha, accepted, stepSize: eps } }
    },
  }
}

/**
 * The unadjusted Langevin algorithm: the same discretised diffusion as {@link malaKernel} without the
 * Metropolis correction, so every step is taken and the chain targets a distribution that is O(eps) away from pi
 * (for a standard normal, variance 1 / (1 - eps / 2)). Cheaper per step, and the basis of stochastic-gradient methods.
 *
 * @param target - The target (with a gradient).
 * @param options - Step size and preconditioner.
 * @returns The kernel (acceptance rate always 1).
 * @throws WhiteboxError with code `value` for a target without a gradient or a non-positive step size.
 * @concept langevin-monte-carlo
 * @reference A. S. Dalalyan, Theoretical guarantees for approximate sampling from smooth and log-concave densities
 *   (JRSS B 2017).
 * @example
 * ```ts
 * import { numericTarget, ulaKernel } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const k = ulaKernel(t, { stepSize: 0.1 })
 * k.step(key(0), k.init([0.2])).info.accepted // => true
 * ```
 */
export function ulaKernel(target: LogDensity, options: LangevinOptions): MCMCKernel<LangevinState> {
  const eps = options.stepSize
  checkStepSize(eps, 'ulaKernel')
  const pre = preconditioner(options.preconditioner, target.dim, 'ulaKernel')
  return {
    name: 'ula',
    dim: target.dim,
    config: { dim: target.dim, stepSize: eps },
    init: (position) => langevinState(target, position, 'ulaKernel'),
    step: (key, state) => {
      const proposal = langevinMove(state, eps, pre, normal(foldIn(key, 0), [target.dim]).data)
      const r = target.valueAndGrad(proposal)
      return {
        state: { position: proposal, logDensity: r.value, grad: r.grad },
        info: { acceptanceRate: 1, accepted: true, stepSize: eps },
      }
    },
  }
}

/** A stochastic estimate of grad log pi at x, drawn with a key (a minibatch gradient, say). */
export type StochasticGradient = (key: Key, x: Float64Array) => ArrayLike<number>

/**
 * The step-size schedule eps_t = a (b + t)^-gamma of Welling and Teh (2011): it satisfies sum eps_t = inf and
 * sum eps_t^2 < inf for gamma in (0.5, 1], so stochastic-gradient Langevin dynamics converges.
 *
 * @param a - The scale.
 * @param b - The offset (default 1).
 * @param gamma - The decay exponent (default 0.55).
 * @returns The schedule t -> eps_t.
 * @throws WhiteboxError with code `value` for a non-positive scale or offset, or gamma outside (0, 1].
 * @concept stochastic-gradient-mcmc
 * @example
 * ```ts
 * import { polynomialStepSize } from 'whitebox-engine/sampling'
 * polynomialStepSize(1, 1, 0.5)(3) // => 0.5
 * ```
 */
export function polynomialStepSize(a: number, b = 1, gamma = 0.55): (t: number) => number {
  if (!(a > 0 && b > 0 && gamma > 0 && gamma <= 1)) {
    throw new WhiteboxError('value', 'polynomialStepSize: need a > 0, b > 0 and gamma in (0, 1]')
  }
  return (t) => a * (b + t) ** -gamma
}

/** The state of {@link sgldKernel}: the position and the iteration (for the step-size schedule). */
export interface SGLDState extends MCMCState {
  readonly t: number
}

/** Options of {@link sgldKernel}. */
export interface SGLDOptions {
  /** A constant step size or a schedule t -> eps_t (see {@link polynomialStepSize}). */
  readonly stepSize: number | ((t: number) => number)
  /** The temperature (default 1): the noise is sqrt(2 eps T) z. */
  readonly temperature?: number
  /** A target to evaluate log densities for traces (default none: the log density is reported as NaN). */
  readonly target?: LogDensity
}

/**
 * Stochastic-gradient Langevin dynamics (Welling and Teh 2011; BlackJAX `sgld`): unadjusted Langevin steps
 * x' = x + eps_t g_hat(x) + sqrt(2 eps_t T) z with a noisy gradient (a minibatch estimate, see
 * {@link minibatchGradient}), and step sizes that decrease so the discretisation error vanishes. Each step costs one
 * minibatch, so it scales to large data sets.
 *
 * @param gradient - The stochastic gradient estimator.
 * @param dim - The dimension.
 * @param options - Step size or schedule, temperature, and a target for traced log densities.
 * @returns The kernel (acceptance rate always 1).
 * @throws WhiteboxError with code `value` for a non-positive constant step size or temperature.
 * @concept stochastic-gradient-mcmc
 * @reference M. Welling and Y. W. Teh, Bayesian learning via stochastic gradient Langevin dynamics (ICML 2011).
 * @example
 * ```ts
 * import { sgldKernel } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const k = sgldKernel((_, x) => [-(x[0] ?? 0)], 1, { stepSize: 0.01 })
 * k.step(key(0), k.init([1])).state.t // => 1
 * ```
 */
export function sgldKernel(gradient: StochasticGradient, dim: number, options: SGLDOptions): MCMCKernel<SGLDState> {
  const schedule =
    typeof options.stepSize === 'number' ? ((() => options.stepSize) as (t: number) => number) : options.stepSize
  if (typeof options.stepSize === 'number') checkStepSize(options.stepSize, 'sgldKernel')
  const temperature = options.temperature ?? 1
  if (!(temperature > 0)) throw new WhiteboxError('value', 'sgldKernel: the temperature must be positive')
  const t = options.target
  const density = (x: Float64Array): number => (t === undefined ? Number.NaN : t.logDensity(x))
  return {
    name: 'sgld',
    dim,
    config: { dim, temperature, stepSize: typeof options.stepSize === 'number' ? options.stepSize : 'schedule' },
    init: (position) => {
      if (position.length !== dim) throw new WhiteboxError('shape', `sgldKernel: the position needs ${dim} values`)
      const x = Float64Array.from(position)
      return { position: x, logDensity: density(x), t: 0 }
    },
    step: (key, state) => {
      const eps = schedule(state.t)
      const g = gradient(foldIn(key, 0), state.position)
      const z = normal(foldIn(key, 1), [dim]).data
      const c = Math.sqrt(2 * eps * temperature)
      const x = state.position.map((v, i) => v + eps * (g[i] ?? 0) + c * (z[i] ?? 0))
      return {
        state: { position: x, logDensity: density(x), t: state.t + 1 },
        info: { acceptanceRate: 1, accepted: true, stepSize: eps },
      }
    },
  }
}

/** Options of {@link minibatchGradient}. */
export interface MinibatchOptions {
  /** log p(theta) for theta of shape `[dim]`. */
  readonly logPrior: (theta: Value) => Value
  /** The summed log likelihood of the data points with the given indices. */
  readonly logLikelihood: (theta: Value, batch: Int32Array) => Value
  /** The number of data points N. */
  readonly numData: number
  /** The minibatch size n (drawn without replacement). */
  readonly batchSize: number
}

/**
 * The unbiased minibatch gradient of a log posterior: grad log p(theta) + (N / n) grad sum_{i in batch} log p(y_i |
 * theta), with the batch drawn without replacement from the key and the gradient by autodiff.
 *
 * @param options - Prior, batched likelihood, data size and batch size.
 * @returns The stochastic gradient estimator (for {@link sgldKernel}).
 * @throws WhiteboxError with code `value` unless 1 <= batchSize <= numData.
 * @concept stochastic-gradient-mcmc
 * @example
 * ```ts
 * import { minibatchGradient } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * import { mul, neg, square, sub, sum, tensor, take } from 'whitebox-engine/tensor'
 * const y = tensor([1, 2, 3, 4])
 * const g = minibatchGradient({
 *   logPrior: () => tensor(0),
 *   logLikelihood: (m, batch) => mul(-0.5, sum(square(sub(take(y, Array.from(batch)), m)))),
 *   numData: 4,
 *   batchSize: 4,
 * })
 * // The full batch: sum (y_i - m) at m = 0 is 10.
 * Array.from(g(key(0), new Float64Array([0]))) // => [10]
 * ```
 */
export function minibatchGradient(options: MinibatchOptions): StochasticGradient {
  const { numData: n, batchSize: b } = options
  if (!(Number.isInteger(n) && Number.isInteger(b) && b >= 1 && b <= n)) {
    throw new WhiteboxError('value', 'minibatchGradient: need integers 1 <= batchSize <= numData')
  }
  const scale = n / b
  return (key, x) => {
    const batch = Int32Array.from(subsample(key, n, b).data)
    const vg = valueAndGrad((theta: Value) =>
      add(options.logPrior(theta), mul(scale, options.logLikelihood(theta, batch))),
    )
    return Float64Array.from(toFlat(vg(fromStorage(Float64Array.from(x), [x.length], 'float64')).grad))
  }
}

/**
 * The Stein variational gradient direction for particles x_1..x_n (Liu and Wang 2016) with the RBF kernel
 * k(x, y) = exp(-|x - y|^2 / h): phi(x_i) = (1/n) sum_j [k(x_j, x_i) grad log pi(x_j) + grad_{x_j} k(x_j, x_i)], the
 * steepest descent direction of the KL divergence in the kernel's RKHS. The first term pulls particles towards high
 * density, the second pushes them apart.
 *
 * @param particles - The particles, row-major `[n, dim]`.
 * @param grads - grad log pi at each particle, row-major `[n, dim]`.
 * @param dim - The dimension.
 * @param lengthScale - The RBF bandwidth h.
 * @returns phi at each particle, row-major `[n, dim]`.
 * @concept stein-variational-gradient-descent
 * @reference Q. Liu and D. Wang, Stein variational gradient descent (NeurIPS 2016).
 * @example
 * ```ts
 * import { svgdDirection } from 'whitebox-engine/sampling'
 * // One particle: only the density term, k(x, x) grad log pi(x).
 * Array.from(svgdDirection(new Float64Array([1]), new Float64Array([-1]), 1, 1)) // => [-1]
 * ```
 */
export function svgdDirection(
  particles: Float64Array,
  grads: Float64Array,
  dim: number,
  lengthScale: number,
): Float64Array {
  const n = particles.length / dim
  const out = new Float64Array(particles.length)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let d2 = 0
      for (let k = 0; k < dim; k++) d2 += ((particles[j * dim + k] ?? 0) - (particles[i * dim + k] ?? 0)) ** 2
      const kij = Math.exp(-d2 / lengthScale)
      for (let k = 0; k < dim; k++) {
        const diff = (particles[j * dim + k] ?? 0) - (particles[i * dim + k] ?? 0)
        out[i * dim + k] = (out[i * dim + k] ?? 0) + kij * (grads[j * dim + k] ?? 0) - (2 * diff * kij) / lengthScale
      }
    }
    for (let k = 0; k < dim; k++) out[i * dim + k] = (out[i * dim + k] ?? 0) / n
  }
  return out
}

/**
 * The median heuristic for the SVGD bandwidth (BlackJAX `median_heuristic`): h = median(|x_i - x_j|)^2 / log n over
 * the n (n - 1) / 2 distinct pairs.
 *
 * @param particles - The particles, row-major `[n, dim]` (n >= 2).
 * @param dim - The dimension.
 * @returns The bandwidth.
 * @throws WhiteboxError with code `value` for fewer than two particles.
 * @concept stein-variational-gradient-descent
 * @example
 * ```ts
 * import { medianBandwidth } from 'whitebox-engine/sampling'
 * // One pair at distance 2: 2^2 / log 2.
 * medianBandwidth(new Float64Array([0, 2]), 1) === 4 / Math.log(2) // => true
 * ```
 */
export function medianBandwidth(particles: Float64Array, dim: number): number {
  const n = particles.length / dim
  if (n < 2) throw new WhiteboxError('value', 'medianBandwidth: need at least two particles')
  const d = new Float64Array((n * (n - 1)) / 2)
  let p = 0
  for (let i = 1; i < n; i++)
    for (let j = 0; j < i; j++) {
      let s = 0
      for (let k = 0; k < dim; k++) s += ((particles[i * dim + k] ?? 0) - (particles[j * dim + k] ?? 0)) ** 2
      d[p++] = Math.sqrt(s)
    }
  const m = medianOf(d)
  return (m * m) / Math.log(n)
}

/** The state of {@link svgdAlgorithm}. */
export interface SVGDState {
  /** The particles, row-major `[n, dim]`. */
  readonly particles: Float64Array
  readonly lengthScale: number
  /** The largest particle move in the last step. */
  readonly change: number
  /** AdaGrad's running average of squared directions (when used). */
  readonly history?: Float64Array
}

/** Options of {@link svgdAlgorithm}. */
export interface SVGDOptions {
  /** The step size (default 0.1). */
  readonly stepSize?: number
  /** A fixed RBF bandwidth, or `median` to recompute the median heuristic every step (default). */
  readonly lengthScale?: number | 'median'
  /** Plain steps (default) or Liu and Wang's AdaGrad with momentum 0.9. */
  readonly optimizer?: 'sgd' | 'adagrad'
  /** Stop when no particle moves more than this (default 1e-8). */
  readonly tol?: number
}

/**
 * Stein variational gradient descent as a traced algorithm: particles move along {@link svgdDirection},
 * x <- x + eps phi(x) (or AdaGrad-scaled steps), with the bandwidth from the median heuristic, until they stop
 * moving. The particles approximate the target as a deterministic sample; the trace records the mean log density,
 * the bandwidth and the largest move, and the particles as a state tensor.
 *
 * @param target - The target (with a gradient).
 * @param initialParticles - The starting particles `[n, dim]` (n >= 2).
 * @param options - Step size, bandwidth, optimizer and tolerance.
 * @returns The algorithm.
 * @throws WhiteboxError with code `shape` for particles that are not `[n, dim]`, and `value` for a target without a
 *   gradient or a non-positive step size.
 * @concept stein-variational-gradient-descent
 * @example
 * ```ts
 * import { numericTarget, svgdAlgorithm } from 'whitebox-engine/sampling'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * runAlgorithm(svgdAlgorithm(t, [[-1], [0], [1]]), { maxSteps: 5 }).steps // => 5
 * ```
 */
export function svgdAlgorithm(
  target: LogDensity,
  initialParticles: TensorLike,
  options: SVGDOptions = {},
): Algorithm<SVGDState> {
  if (!target.hasGradient) throw new WhiteboxError('value', 'svgdAlgorithm: the target has no gradient')
  const x0 = asTensor(initialParticles) as Tensor
  const dim = target.dim
  if (x0.shape.length !== 2 || x0.shape[1] !== dim || (x0.shape[0] ?? 0) < 2) {
    throw new WhiteboxError('shape', `svgdAlgorithm: particles must be [n >= 2, ${dim}]`)
  }
  const n = x0.shape[0] ?? 0
  const eps = options.stepSize ?? 0.1
  checkStepSize(eps, 'svgdAlgorithm')
  const bandwidth = options.lengthScale ?? 'median'
  const adagrad = options.optimizer === 'adagrad'
  const tol = options.tol ?? 1e-8
  const start = Float64Array.from(toFlat(x0))
  const grads = (p: Float64Array): { readonly g: Float64Array; readonly mean: number } => {
    const g = new Float64Array(p.length)
    let mean = 0
    for (let i = 0; i < n; i++) {
      const r = target.valueAndGrad(p.subarray(i * dim, (i + 1) * dim))
      g.set(r.grad, i * dim)
      mean += r.value / n
    }
    return { g, mean }
  }
  const h = (p: Float64Array): number => (bandwidth === 'median' ? medianBandwidth(p, dim) || 1 : bandwidth)
  return {
    name: 'svgd',
    config: {
      dim,
      particles: n,
      stepSize: eps,
      lengthScale: bandwidth,
      optimizer: adagrad ? 'adagrad' : 'sgd',
      tol,
    } as Json,
    schema: channels({
      metrics: {
        meanLogDensity: { description: 'mean log density of the particles', direction: 'higher' },
        lengthScale: { description: 'RBF bandwidth' },
        change: { description: 'largest particle move in the step', direction: 'lower' },
      },
      tensors: { particles: { role: 'state', description: 'the particles', shape: [n, dim] } },
    }),
    init: () => ({ particles: start, lengthScale: h(start), change: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const ls = h(s.particles)
      const phi = svgdDirection(s.particles, grads(s.particles).g, dim, ls)
      let history = s.history
      let move = phi.map((v) => eps * v)
      if (adagrad) {
        history = phi.map((v, i) => (s.history === undefined ? v * v : 0.9 * (s.history[i] ?? 0) + 0.1 * v * v))
        const hist = history
        move = phi.map((v, i) => (eps * v) / (1e-6 + Math.sqrt(hist[i] ?? 0)))
      }
      let change = 0
      const particles = s.particles.map((v, i) => {
        const m = move[i] ?? 0
        change = Math.max(change, Math.abs(m))
        return v + m
      })
      return { particles, lengthScale: ls, change, ...(history === undefined ? {} : { history }) }
    },
    done: (s) => (Number.isNaN(s.change) ? 'diverged' : s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        meanLogDensity: grads(s.particles).mean,
        lengthScale: s.lengthScale,
        ...(Number.isFinite(s.change) ? { change: s.change } : {}),
      },
      tensors: { particles: { role: 'state', value: s.particles, shape: [n, dim] } },
    }),
  }
}

/**
 * Run Stein variational gradient descent ({@link svgdAlgorithm}) and return the particles.
 *
 * @param target - The target (with a gradient).
 * @param initialParticles - The starting particles `[n, dim]`.
 * @param options - SVGD options, an iteration cap (default 1000), tracer and policy.
 * @returns The particles `[n, dim]`, the iterations and whether they settled within the tolerance.
 * @concept stein-variational-gradient-descent
 * @example
 * ```ts
 * import { numericTarget, svgd } from 'whitebox-engine/sampling'
 * const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) - 3) ** 2, 1, { grad: (x) => [3 - (x[0] ?? 0)] })
 * const r = svgd(t, [[-1], [0], [1], [2]], { maxIter: 500 })
 * const m = r.particles.data.reduce((a, b) => a + b, 0) / 4
 * Math.abs(m - 3) < 0.05 // => true
 * ```
 */
export function svgd(
  target: LogDensity,
  initialParticles: TensorLike,
  options: SVGDOptions & { readonly maxIter?: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): { readonly particles: Tensor<'float64'>; readonly iterations: number; readonly converged: boolean } {
  const { maxIter, stepSize, lengthScale, optimizer, tol, ...run } = options
  const algorithm = svgdAlgorithm(target, initialParticles, {
    ...(stepSize === undefined ? {} : { stepSize }),
    ...(lengthScale === undefined ? {} : { lengthScale }),
    ...(optimizer === undefined ? {} : { optimizer }),
    ...(tol === undefined ? {} : { tol }),
  })
  const r = runAlgorithm(algorithm, { ...run, maxSteps: maxIter ?? 1000 })
  const n = r.state.particles.length / target.dim
  return {
    particles: fromStorage(r.state.particles, [n, target.dim], 'float64'),
    iterations: r.steps,
    converged: r.status === 'converged',
  }
}
