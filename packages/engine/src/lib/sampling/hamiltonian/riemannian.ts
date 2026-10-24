import { hessian, jacobian } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { ddot, dgemv, eigh } from '../../numerics/linalg/index.ts'
import { foldIn, normal } from '../../random/index.ts'
import { fromStorage, neg, reshape, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { allFinite, inverse, lowerCholesky } from '../_linear.ts'
import { acceptWith, checkStepSize } from '../_mcmc.ts'
import type { MCMCKernel, MCMCState, Transition } from '../chains.ts'
import type { LogDensity } from '../target.ts'

/** Options of {@link riemannianHmcKernel}. */
export interface RiemannianHMCOptions {
  readonly stepSize: number
  /** Generalised leapfrog steps per transition (a positive integer, at most 10000). */
  readonly numIntegrationSteps: number
  /**
   * The metric G(x) as a differentiable function of a `[dim]` value returning a symmetric positive-definite
   * `[dim, dim]` matrix (the expected Fisher information, say). Default: the SoftAbs map of the Hessian of
   * -log pi, which needs a target built with `target` (so it can be differentiated three times).
   */
  readonly metric?: (x: Value) => Value
  /** SoftAbs sharpness alpha: eigenvalues lambda become lambda coth(alpha lambda) >= 1 / alpha (default 1e6). */
  readonly softAbsAlpha?: number
  /** Fixed-point iterations of the implicit updates (default 8). */
  readonly fixedPointIterations?: number
  /** Convergence tolerance of the fixed-point iterations (default 1e-10). */
  readonly tolerance?: number
  /** The energy error above which a transition counts as divergent (default 1000). */
  readonly divergenceThreshold?: number
}

/** The geometry of the metric at a point: what one generalised leapfrog step needs there. */
interface Geometry {
  readonly position: Float64Array
  readonly logDensity: number
  readonly grad: Float64Array
  readonly metricInverse: Float64Array
  readonly logDetMetric: number
  /** A square root S of the metric (S S^T = G), for drawing momenta p ~ N(0, G). */
  readonly sqrt: Float64Array
  /** d/dx_k of the metric terms of H: 0.5 tr(G^-1 dG_k) - 0.5 p^T G^-1 dG_k G^-1 p. */
  metricForce(p: Float64Array): Float64Array
}

/** The SoftAbs map lambda coth(alpha lambda) and its derivative, with their limits near 0 and far from it. */
function softAbs(w: number, alpha: number): { readonly value: number; readonly slope: number } {
  const a = alpha * w
  if (Math.abs(a) < 1e-4) return { value: 1 / alpha + (alpha * w * w) / 3, slope: (2 * a) / 3 }
  if (Math.abs(a) > 20) return { value: Math.abs(w), slope: Math.sign(w) }
  const c = 1 / Math.tanh(a)
  const s = Math.sinh(a)
  return { value: w * c, slope: c - a / (s * s) }
}

/** Q diag(d) Q^T for eigenvectors Q as columns (row-major). */
function reassemble(q: Float64Array, d: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += (q[i * n + k] ?? 0) * (d[k] ?? 0) * (q[j * n + k] ?? 0)
      out[i * n + j] = s
    }
  return out
}

function symmetrise(a: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) out[i * n + j] = 0.5 * ((a[i * n + j] ?? 0) + (a[j * n + i] ?? 0))
  return out
}

function point(x: Float64Array): Value {
  return fromStorage(Float64Array.from(x), [x.length], 'float64')
}

/** The pieces of Riemannian dynamics: the geometry at a point, one generalised leapfrog step and the energy. */
interface Dynamics {
  readonly geometry: (x: Float64Array) => Geometry
  readonly leap: (
    g: Geometry,
    p: Float64Array,
  ) => { readonly geometry: Geometry; readonly momentum: Float64Array } | undefined
  readonly energy: (g: Geometry, p: Float64Array) => number
}

function riemannianDynamics(target: LogDensity, options: Omit<RiemannianHMCOptions, 'numIntegrationSteps'>): Dynamics {
  const { stepSize } = options
  checkStepSize(stepSize, 'riemannianHmcKernel')
  const fn = target.fn
  const userMetric = options.metric
  if (userMetric === undefined && fn === undefined) {
    throw new WhiteboxError(
      'value',
      'riemannianHmcKernel: the SoftAbs metric needs a differentiable target (use target())',
    )
  }
  const n = target.dim
  const alpha = options.softAbsAlpha ?? 1e6
  const iterations = options.fixedPointIterations ?? 8
  const tol = options.tolerance ?? 1e-10

  const negHessian = fn === undefined ? undefined : (z: Value): Value => neg(hessian(fn)(z))
  const flatNegHessian = negHessian === undefined ? undefined : (z: Value): Value => reshape(negHessian(z), [n * n])
  const flatMetric = userMetric === undefined ? undefined : (z: Value): Value => reshape(userMetric(z), [n * n])

  /** The metric's inverse alone (the implicit position update evaluates it repeatedly). */
  const metricInverseAt = (x: Float64Array): Float64Array => {
    if (userMetric !== undefined) return inverse(symmetrise(Float64Array.from(toFlat(userMetric(point(x)))), n), n)
    const h = symmetrise(Float64Array.from(toFlat((negHessian as (z: Value) => Value)(point(x)))), n)
    const e = eigh(fromStorage(h, [n, n], 'float64'))
    const w = toFlat(e.eigenvalues)
    const q = Float64Array.from(toFlat(e.eigenvectors))
    return reassemble(
      q,
      Float64Array.from(w, (v) => 1 / softAbs(v, alpha).value),
      n,
    )
  }

  const geometry = (x: Float64Array): Geometry => {
    const r = target.valueAndGrad(x)
    if (userMetric !== undefined) {
      const g = symmetrise(Float64Array.from(toFlat(userMetric(point(x)))), n)
      const sqrt = lowerCholesky(g, n, 'riemannianHmcKernel')
      const ginv = inverse(g, n)
      let logDet = 0
      for (let i = 0; i < n; i++) logDet += 2 * Math.log(sqrt[i * n + i] ?? 0)
      // dG[(i n + j) n + k] = d G_ij / d x_k.
      const dG = (jacobian(flatMetric as (z: Value) => Value)(point(x)) as Tensor).data as Float64Array
      return {
        position: x,
        logDensity: r.value,
        grad: r.grad,
        metricInverse: ginv,
        logDetMetric: logDet,
        sqrt,
        metricForce: (p) => {
          const v = dgemv(ginv, n, n, p)
          return Float64Array.from({ length: n }, (_, k) => {
            let tr = 0
            let quad = 0
            for (let i = 0; i < n; i++)
              for (let j = 0; j < n; j++) {
                const d = dG[(i * n + j) * n + k] ?? 0
                tr += (ginv[j * n + i] ?? 0) * d
                quad += (v[i] ?? 0) * d * (v[j] ?? 0)
              }
            return 0.5 * tr - 0.5 * quad
          })
        },
      }
    }
    const h = symmetrise(Float64Array.from(toFlat((negHessian as (z: Value) => Value)(point(x)))), n)
    const e = eigh(fromStorage(h, [n, n], 'float64'))
    const w = Float64Array.from(toFlat(e.eigenvalues))
    const q = Float64Array.from(toFlat(e.eigenvectors))
    const sa = Array.from(w, (v) => softAbs(v, alpha))
    const lam = Float64Array.from(sa, (s) => s.value)
    // J_ij: divided differences of the SoftAbs map, its slope on (near-)ties.
    const jm = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const gap = (w[i] ?? 0) - (w[j] ?? 0)
        jm[i * n + j] =
          i !== j && Math.abs(gap) > 1e-10 * (1 + Math.abs(w[i] ?? 0))
            ? ((lam[i] ?? 0) - (lam[j] ?? 0)) / gap
            : (sa[i]?.slope ?? 0)
      }
    // dH[(i n + j) n + k] = d (-Hessian)_ij / d x_k, and M_k = Q^T dH_k Q.
    const dH = (jacobian(flatNegHessian as (z: Value) => Value)(point(x)) as Tensor).data as Float64Array
    const rotated: Float64Array[] = Array.from({ length: n }, (_, k) => {
      const m = new Float64Array(n * n)
      for (let a = 0; a < n; a++)
        for (let b = 0; b < n; b++) {
          let s = 0
          for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++) s += (q[i * n + a] ?? 0) * (dH[(i * n + j) * n + k] ?? 0) * (q[j * n + b] ?? 0)
          m[a * n + b] = s
        }
      return m
    })
    let logDet = 0
    for (const l of lam) logDet += Math.log(l)
    const sqrt = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < n; k++) sqrt[i * n + k] = (q[i * n + k] ?? 0) * Math.sqrt(lam[k] ?? 0)
    return {
      position: x,
      logDensity: r.value,
      grad: r.grad,
      metricInverse: reassemble(
        q,
        lam.map((l) => 1 / l),
        n,
      ),
      logDetMetric: logDet,
      sqrt,
      metricForce: (p) => {
        // u = diag(1 / lambda') Q^T p, the momentum's velocity in the eigenbasis.
        const qp = dgemv(q, n, n, p, true)
        const u = qp.map((v, i) => v / (lam[i] ?? 1))
        return Float64Array.from({ length: n }, (_, k) => {
          const m = rotated[k] ?? new Float64Array(n * n)
          let tr = 0
          let quad = 0
          for (let a = 0; a < n; a++) {
            tr += ((jm[a * n + a] ?? 0) * (m[a * n + a] ?? 0)) / (lam[a] ?? 1)
            for (let b = 0; b < n; b++) quad += (u[a] ?? 0) * (jm[a * n + b] ?? 0) * (m[a * n + b] ?? 0) * (u[b] ?? 0)
          }
          return 0.5 * tr - 0.5 * quad
        })
      },
    }
  }

  const energy = (g: Geometry, p: Float64Array): number => {
    const h = -g.logDensity + 0.5 * g.logDetMetric + 0.5 * ddot(p, dgemv(g.metricInverse, n, n, p))
    return Number.isNaN(h) ? Number.POSITIVE_INFINITY : h
  }

  const forceAt = (g: Geometry, p: Float64Array): Float64Array => {
    const f = g.metricForce(p)
    return f.map((v, k) => v - (g.grad[k] ?? 0))
  }

  const maxChange = (a: Float64Array, b: Float64Array): number => {
    let m = 0
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)))
    return m
  }

  /** One generalised leapfrog step, or undefined where the geometry fails. */
  const leap = (
    g: Geometry,
    p: Float64Array,
  ): { readonly geometry: Geometry; readonly momentum: Float64Array } | undefined => {
    const h = stepSize / 2
    let half = p
    for (let i = 0; i < iterations; i++) {
      const f = forceAt(g, half)
      const next = p.map((v, k) => v - h * (f[k] ?? 0))
      const change = maxChange(next, half)
      half = next
      if (change < tol) break
    }
    const v0 = dgemv(g.metricInverse, n, n, half)
    let x = g.position.map((v, k) => v + stepSize * (v0[k] ?? 0))
    for (let i = 0; i < iterations; i++) {
      const v1 = dgemv(metricInverseAt(x), n, n, half)
      const next = g.position.map((v, k) => v + h * ((v0[k] ?? 0) + (v1[k] ?? 0)))
      const change = maxChange(next, x)
      x = next
      if (change < tol) break
    }
    if (!allFinite(x) || !allFinite(half)) return undefined
    const g1 = geometry(x)
    const f1 = forceAt(g1, half)
    return { geometry: g1, momentum: half.map((v, k) => v - h * (f1[k] ?? 0)) }
  }

  return { geometry, leap, energy }
}

function attempt<T>(f: () => T): T | undefined {
  try {
    return f()
  } catch {
    return undefined
  }
}

/**
 * Riemannian manifold HMC (Girolami and Calderhead 2011) for small problems: momenta p ~ N(0, G(x)) with a
 * position-dependent metric, H(x, p) = -log pi(x) + log det G(x) / 2 + p^T G(x)^-1 p / 2, integrated by the
 * generalised leapfrog (implicit momentum and position half-updates solved by fixed-point iteration, then an explicit
 * momentum update), and a Metropolis correction. The default metric is Betancourt's SoftAbs map of the Hessian of
 * -log pi, positive definite everywhere and smooth; its derivative uses divided differences of the eigenvalues, so
 * repeated eigenvalues are fine. With a constant metric the method reduces to HMC with mass matrix G.
 *
 * @param target - The target (with `fn`, as built by `target`, when the metric is the default).
 * @param options - Step size, steps, metric, SoftAbs alpha, fixed-point iterations and tolerance, divergence threshold.
 * @returns The kernel. Transitions where the metric cannot be evaluated or factored count as divergent.
 * @throws WhiteboxError with code `value` for a target without a differentiable function (default metric), a
 *   non-positive step size or an invalid number of steps.
 * @concept riemannian-hmc
 * @reference M. Girolami and B. Calderhead, Riemann manifold Langevin and Hamiltonian Monte Carlo methods (JRSS B
 *   2011); M. Betancourt, A general metric for Riemannian manifold Hamiltonian Monte Carlo (2013).
 * @example
 * ```ts
 * import { riemannianHmcKernel, target } from 'whitebox-engine/sampling'
 * import { mul, square, sum } from 'whitebox-engine/tensor'
 * import { key } from 'whitebox-engine/random'
 * const t = target((x) => mul(-0.5, sum(square(x))), 2)
 * const k = riemannianHmcKernel(t, { stepSize: 0.3, numIntegrationSteps: 3 })
 * k.step(key(0), k.init([0.5, -0.5])).info.acceptanceRate > 0.9 // => true
 * ```
 */
export function riemannianHmcKernel(target: LogDensity, options: RiemannianHMCOptions): MCMCKernel {
  const { stepSize, numIntegrationSteps: steps } = options
  if (!(Number.isInteger(steps) && steps >= 1 && steps <= 10000)) {
    throw new WhiteboxError('value', 'riemannianHmcKernel: numIntegrationSteps must be an integer in [1, 10000]')
  }
  const { geometry, leap, energy } = riemannianDynamics(target, options)
  const n = target.dim
  const threshold = options.divergenceThreshold ?? 1000
  return {
    name: 'riemannianHmc',
    dim: n,
    config: {
      dim: n,
      stepSize,
      numIntegrationSteps: steps,
      softAbsAlpha: options.softAbsAlpha ?? 1e6,
      metric: options.metric === undefined ? 'softAbs' : 'given',
    },
    init: (position) => {
      if (position.length !== n) {
        throw new WhiteboxError('shape', `riemannianHmcKernel: the position has ${position.length} values, need ${n}`)
      }
      const x = Float64Array.from(position)
      const s: MCMCState = { position: x, logDensity: target.logDensity(x) }
      return s
    },
    step: (key, state) => {
      const reject = (numSteps: number): Transition<MCMCState> => ({
        state,
        info: { acceptanceRate: 0, accepted: false, divergent: true, numSteps, stepSize },
      })
      const g0 = Number.isFinite(state.logDensity) ? attempt(() => geometry(state.position)) : undefined
      if (g0 === undefined || !allFinite(g0.sqrt)) return reject(0)
      const p0 = dgemv(g0.sqrt, n, n, normal(foldIn(key, 0), [n]).data)
      const h0 = energy(g0, p0)
      let g = g0
      let p = p0
      let taken = 0
      while (taken < steps) {
        const next = attempt(() => leap(g, p))
        taken += 1
        if (next === undefined || !Number.isFinite(next.geometry.logDensity)) return reject(taken)
        g = next.geometry
        p = next.momentum
      }
      const h1 = energy(g, p)
      const raw = h0 - h1
      const delta = Number.isNaN(raw) ? Number.NEGATIVE_INFINITY : raw
      const alpha = delta >= 0 ? 1 : Math.exp(delta)
      const accepted = acceptWith(foldIn(key, 1), alpha)
      return {
        state: accepted ? { position: g.position, logDensity: g.logDensity } : state,
        info: {
          acceptanceRate: alpha,
          accepted,
          divergent: -delta > threshold,
          numSteps: taken,
          energy: accepted ? h1 : h0,
          stepSize,
        },
      }
    },
  }
}

/** A state of {@link riemannianTrajectory}. */
export interface RiemannianPoint {
  readonly position: Float64Array
  readonly momentum: Float64Array
  /** H(x, p) = -log pi(x) + log det G(x) / 2 + p^T G(x)^-1 p / 2. */
  readonly energy: number
}

/**
 * Integrate the generalised leapfrog of Riemannian HMC from a position and momentum for `numSteps` steps and return
 * every state with its energy (for figures, and to see that the energy error is second order in the step size).
 * The integration stops at the first state where the geometry cannot be evaluated.
 *
 * @param target - The target.
 * @param position - The starting position.
 * @param momentum - The starting momentum.
 * @param options - Step size, number of steps, metric and fixed-point settings, as {@link riemannianHmcKernel}.
 * @returns The states, starting with the initial one.
 * @throws WhiteboxError with code `shape` for inputs of the wrong length, and `value` as {@link riemannianHmcKernel}.
 * @concept riemannian-hmc
 * @example
 * ```ts
 * import { riemannianTrajectory, target } from 'whitebox-engine/sampling'
 * import { mul, square, sum } from 'whitebox-engine/tensor'
 * const t = target((x) => mul(-0.5, sum(square(x))), 2)
 * const path = riemannianTrajectory(t, [1, 0], [0, 1], { stepSize: 0.1, numIntegrationSteps: 5 })
 * Math.abs((path[5]?.energy ?? 0) - (path[0]?.energy ?? 0)) < 1e-2 // => true
 * ```
 */
export function riemannianTrajectory(
  target: LogDensity,
  position: ArrayLike<number>,
  momentum: ArrayLike<number>,
  options: RiemannianHMCOptions,
): RiemannianPoint[] {
  const n = target.dim
  if (position.length !== n || momentum.length !== n) {
    throw new WhiteboxError('shape', `riemannianTrajectory: position and momentum need ${n} values`)
  }
  const steps = options.numIntegrationSteps
  if (!(Number.isInteger(steps) && steps >= 0 && steps <= 10000)) {
    throw new WhiteboxError('value', 'riemannianTrajectory: numIntegrationSteps must be an integer in [0, 10000]')
  }
  const { geometry, leap, energy } = riemannianDynamics(target, options)
  let g = geometry(Float64Array.from(position))
  let p: Float64Array = Float64Array.from(momentum)
  const out: RiemannianPoint[] = [{ position: g.position, momentum: p, energy: energy(g, p) }]
  for (let i = 0; i < steps; i++) {
    const next = attempt(() => leap(g, p))
    if (next === undefined) break
    g = next.geometry
    p = next.momentum
    out.push({ position: g.position, momentum: p, energy: energy(g, p) })
  }
  return out
}
