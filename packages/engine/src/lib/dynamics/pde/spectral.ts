/**
 * Spectral methods, which represent a solution by global basis functions (Fourier modes or Chebyshev polynomials) for
 * very high accuracy on smooth problems. Fourier wavenumbers, FFT derivatives and dealiasing; Chebyshev points,
 * differentiation matrices and a collocation BVP solver; and ETDRK4 in Fourier space with the Kuramoto-Sivashinsky, KdV
 * and Allen-Cahn equations as ready solvers.
 */

import { MdmlError } from '../../base/index.ts'
import { fft, ifft } from '../../numerics/fourier/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import { asTensor, fromStorage, mul, primalOf, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import type { PdeTraceOptions } from './finite-difference.ts'

/**
 * The angular wavenumbers of an n-point periodic grid of length L, in FFT order: (2 pi / L) [0, 1, ..., n/2 - 1,
 * -n/2, ..., -1].
 *
 * @param n - Grid points.
 * @param length - Period L.
 * @returns The wavenumbers.
 * @concept spectral-method
 * @example
 * ```ts
 * import { fourierWavenumbers } from 'mdml-engine/dynamics/pde'
 * Array.from(fourierWavenumbers(4, 2 * Math.PI)) // => [0, 1, -2, -1]
 * ```
 */
export function fourierWavenumbers(n: number, length: number): Float64Array {
  return Float64Array.from({ length: n }, (_, j) => ((2 * Math.PI) / length) * (j < n / 2 ? j : j - n))
}

/**
 * The derivative of a periodic function sampled at n equispaced points by the FFT: multiply the coefficients by
 * (i k)^order (the Nyquist mode is dropped for odd orders). Built from FFT operations, so it is differentiable in u.
 *
 * @param u - Samples `[n]` on [0, L).
 * @param length - The period L.
 * @param order - The derivative order (default 1).
 * @returns The derivative samples.
 * @concept spectral-method
 * @example
 * ```ts
 * import { fourierDerivative } from 'mdml-engine/dynamics/pde'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 16 }, (_, j) => (2 * Math.PI * j) / 16)
 * const d = toFlat(fourierDerivative(x.map(Math.sin), 2 * Math.PI))
 * Math.abs((d[3] ?? 0) - Math.cos(x[3] ?? 0)) < 1e-12 // => true
 * ```
 */
export function fourierDerivative(u: TensorLike, length: number, order = 1): Value {
  const v = asTensor(u)
  const n = v.shape[0] ?? 0
  const k = fourierWavenumbers(n, length)
  // (i k)^m = k^m i^m: i^m cycles through 1, i, -1, -i.
  const km = Float64Array.from(k, (kj, j) => (order % 2 === 1 && n % 2 === 0 && j === n / 2 ? 0 : kj ** order))
  const phase = ((order % 4) + 4) % 4
  const hat = fft(v)
  const w = fromStorage(km, [n], 'float64')
  const re = mul(hat.re, w)
  const im = mul(hat.im, w)
  const rotated =
    phase === 0
      ? { re, im }
      : phase === 1
        ? { re: mul(im, -1), im: re }
        : phase === 2
          ? { re: mul(re, -1), im: mul(im, -1) }
          : { re: im, im: mul(re, -1) }
  return ifft(rotated).re
}

/**
 * The 2/3 dealiasing mask of Orszag: 1 for |k| <= n/3 (in index units), 0 above, to remove aliasing of quadratic
 * nonlinearities.
 *
 * @param n - Grid points.
 * @returns The mask in FFT order.
 * @concept dealiasing
 * @example
 * ```ts
 * import { dealiasMask } from 'mdml-engine/dynamics/pde'
 * Array.from(dealiasMask(6)) // => [1, 1, 1, 0, 1, 1]
 * ```
 */
export function dealiasMask(n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, j) => (Math.abs(j < n / 2 ? j : j - n) <= n / 3 ? 1 : 0))
}

/**
 * The Chebyshev points x_j = cos(pi j / n), j = 0..n, on [-1, 1].
 *
 * @param n - The polynomial degree (n + 1 points).
 * @returns The points, from 1 down to -1.
 * @concept spectral-method
 * @example
 * ```ts
 * import { chebyshevPoints } from 'mdml-engine/dynamics/pde'
 * Array.from(chebyshevPoints(2)).map((x) => Math.round(x * 1e12) / 1e12) // => [1, 0, -1]
 * ```
 */
export function chebyshevPoints(n: number): Float64Array {
  if (!Number.isInteger(n) || n < 1) throw new MdmlError('value', 'chebyshevPoints: n >= 1')
  return Float64Array.from({ length: n + 1 }, (_, j) => Math.cos((Math.PI * j) / n))
}

/**
 * The Chebyshev differentiation matrix D on the n + 1 Chebyshev points (Trefethen's cheb, with the negative-sum trick
 * for the diagonal): (D u)_i is the derivative at x_i of the interpolant of u.
 *
 * @param n - The degree.
 * @returns D as `[n + 1, n + 1]`.
 * @concept spectral-method
 * @example
 * ```ts
 * import { chebyshevMatrix } from 'mdml-engine/dynamics/pde'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(chebyshevMatrix(1)) // => [[0.5, -0.5], [0.5, -0.5]]
 * ```
 */
export function chebyshevMatrix(n: number): Tensor {
  const x = chebyshevPoints(n)
  const m = n + 1
  const c = Float64Array.from({ length: m }, (_, i) => (i === 0 || i === n ? 2 : 1) * (i % 2 === 0 ? 1 : -1))
  const d = new Float64Array(m * m)
  for (let i = 0; i < m; i++) {
    let rowSum = 0
    for (let j = 0; j < m; j++) {
      if (i === j) continue
      const v = (c[i] ?? 1) / (c[j] ?? 1) / ((x[i] ?? 0) - (x[j] ?? 0))
      d[i * m + j] = v
      rowSum += v
    }
    d[i * m + i] = -rowSum
  }
  return fromStorage(d, [m, m], 'float64')
}

/**
 * Solve the linear two-point problem a2 u'' + a1 u' + a0 u = f on [a, b] with Dirichlet values by Chebyshev
 * collocation (spectral accuracy for smooth data).
 *
 * @param f - The right-hand side as a function of x.
 * @param options - Degree n, interval, coefficients (constants or functions of x; default a2 = 1, a1 = a0 = 0) and
 * the boundary values at a and b.
 * @returns The Chebyshev points mapped to [a, b] and the solution there.
 * @concept spectral-method
 * @example
 * ```ts
 * import { chebyshevBvp } from 'mdml-engine/dynamics/pde'
 * // u'' = exp(4x), u(+-1) = 0 (Trefethen's p13).
 * const r = chebyshevBvp((x) => Math.exp(4 * x), { n: 16 })
 * const exact = (Math.exp(0) - Math.sinh(4) * 0 - Math.cosh(4)) / 16
 * Math.abs((r.u.data[8] ?? 0) - exact) < 1e-10 // => true
 * ```
 */
export function chebyshevBvp(
  f: (x: number) => number,
  options: {
    readonly n: number
    readonly interval?: readonly [number, number]
    readonly a2?: number | ((x: number) => number)
    readonly a1?: number | ((x: number) => number)
    readonly a0?: number | ((x: number) => number)
    readonly left?: number
    readonly right?: number
  },
): { readonly x: Float64Array; readonly u: Tensor } {
  const n = options.n
  const [a, b] = options.interval ?? [-1, 1]
  const s = 2 / (b - a)
  const t = chebyshevPoints(n)
  const x = Float64Array.from(t, (v) => a + ((v + 1) * (b - a)) / 2)
  const D = chebyshevMatrix(n).data
  const m = n + 1
  const D2 = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      let acc = 0
      for (let k = 0; k < m; k++) acc += (D[i * m + k] ?? 0) * (D[k * m + j] ?? 0)
      D2[i * m + j] = acc
    }
  const val = (c: number | ((x: number) => number) | undefined, d: number, xi: number): number =>
    c === undefined ? d : typeof c === 'number' ? c : c(xi)
  const A = new Float64Array(m * m)
  const rhs = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    const xi = x[i] ?? 0
    if (i === 0 || i === n) {
      A[i * m + i] = 1
      // x_0 = b (t = 1), x_n = a (t = -1).
      rhs[i] = i === 0 ? (options.right ?? 0) : (options.left ?? 0)
      continue
    }
    for (let j = 0; j < m; j++) {
      A[i * m + j] =
        val(options.a2, 1, xi) * s * s * (D2[i * m + j] ?? 0) +
        val(options.a1, 0, xi) * s * (D[i * m + j] ?? 0) +
        (i === j ? val(options.a0, 0, xi) : 0)
    }
    rhs[i] = f(xi)
  }
  const u = primalOf(solve(fromStorage(A, [m, m], 'float64'), fromStorage(rhs, [m], 'float64')))
  return { x, u }
}

/** A complex vector as parts. */
interface Cvec {
  re: Float64Array
  im: Float64Array
}

const cmul = (ar: number, ai: number, br: number, bi: number): [number, number] => [
  ar * br - ai * bi,
  ar * bi + ai * br,
]

/**
 * The ETDRK4 coefficients for a complex diagonal linear part (Kassam and Trefethen: contour integrals of the phi
 * functions over M points on a circle of radius 1 around each h L, which avoids cancellation).
 */
function etdCoefficients(
  lr: Float64Array,
  li: number[] | Float64Array,
  h: number,
  M = 32,
): Record<'E' | 'E2' | 'Q' | 'f1' | 'f2' | 'f3', Cvec> {
  const n = lr.length
  const out = {
    E: { re: new Float64Array(n), im: new Float64Array(n) },
    E2: { re: new Float64Array(n), im: new Float64Array(n) },
    Q: { re: new Float64Array(n), im: new Float64Array(n) },
    f1: { re: new Float64Array(n), im: new Float64Array(n) },
    f2: { re: new Float64Array(n), im: new Float64Array(n) },
    f3: { re: new Float64Array(n), im: new Float64Array(n) },
  }
  for (let k = 0; k < n; k++) {
    const zr = h * (lr[k] ?? 0)
    const zi = h * (li[k] ?? 0)
    const e = Math.exp(zr)
    out.E.re[k] = e * Math.cos(zi)
    out.E.im[k] = e * Math.sin(zi)
    const e2 = Math.exp(zr / 2)
    out.E2.re[k] = e2 * Math.cos(zi / 2)
    out.E2.im[k] = e2 * Math.sin(zi / 2)
    const acc = { Q: [0, 0], f1: [0, 0], f2: [0, 0], f3: [0, 0] }
    for (let j = 1; j <= M; j++) {
      const th = (Math.PI * (j - 0.5)) / M
      const rr = zr + Math.cos(th)
      const ri = zi + Math.sin(th)
      // Q: (exp(r/2) - 1) / r; f1: (-4 - r + e^r (4 - 3r + r^2)) / r^3; f2: (2 + r + e^r (r - 2)) / r^3;
      // f3: (-4 - 3r - r^2 + e^r (4 - r)) / r^3.
      const er = Math.exp(rr)
      const [exr, exi] = [er * Math.cos(ri), er * Math.sin(ri)]
      const eh = Math.exp(rr / 2)
      const [ehr, ehi] = [eh * Math.cos(ri / 2), eh * Math.sin(ri / 2)]
      const inv = (ar: number, ai: number): [number, number] => {
        const d = ar * ar + ai * ai
        return [ar / d, -ai / d]
      }
      const [r1r, r1i] = inv(rr, ri)
      const [r2r, r2i] = cmul(rr, ri, rr, ri)
      const [r3r, r3i] = cmul(r2r, r2i, rr, ri)
      const [i3r, i3i] = inv(r3r, r3i)
      const q = cmul(ehr - 1, ehi, r1r, r1i)
      const t1 = cmul(exr, exi, 4 - 3 * rr + r2r, -3 * ri + r2i)
      const f1 = cmul(-4 - rr + t1[0], -ri + t1[1], i3r, i3i)
      const t2 = cmul(exr, exi, rr - 2, ri)
      const f2 = cmul(2 + rr + t2[0], ri + t2[1], i3r, i3i)
      const t3 = cmul(exr, exi, 4 - rr, -ri)
      const f3 = cmul(-4 - 3 * rr - r2r + t3[0], -3 * ri - r2i + t3[1], i3r, i3i)
      acc.Q[0] = (acc.Q[0] ?? 0) + q[0] / M
      acc.Q[1] = (acc.Q[1] ?? 0) + q[1] / M
      acc.f1[0] = (acc.f1[0] ?? 0) + f1[0] / M
      acc.f1[1] = (acc.f1[1] ?? 0) + f1[1] / M
      acc.f2[0] = (acc.f2[0] ?? 0) + f2[0] / M
      acc.f2[1] = (acc.f2[1] ?? 0) + f2[1] / M
      acc.f3[0] = (acc.f3[0] ?? 0) + f3[0] / M
      acc.f3[1] = (acc.f3[1] ?? 0) + f3[1] / M
    }
    // Real parts suffice for real L (KT take the real part); for complex L keep both (the contour mean is the value).
    const realL = (li[k] ?? 0) === 0
    out.Q.re[k] = h * (acc.Q[0] ?? 0)
    out.Q.im[k] = realL ? 0 : h * (acc.Q[1] ?? 0)
    out.f1.re[k] = h * (acc.f1[0] ?? 0)
    out.f1.im[k] = realL ? 0 : h * (acc.f1[1] ?? 0)
    out.f2.re[k] = h * (acc.f2[0] ?? 0)
    out.f2.im[k] = realL ? 0 : h * (acc.f2[1] ?? 0)
    out.f3.re[k] = h * (acc.f3[0] ?? 0)
    out.f3.im[k] = realL ? 0 : h * (acc.f3[1] ?? 0)
  }
  return out
}

/** A periodic spectral problem in Fourier space: v_t = L v + N(v) with diagonal complex L. */
export interface SpectralProblem {
  /** Real and imaginary parts of the diagonal of L, in FFT order. */
  readonly linear: { readonly re: ArrayLike<number>; readonly im: ArrayLike<number> }
  /** The nonlinear term in Fourier space. */
  readonly nonlinear: (v: { re: Float64Array; im: Float64Array }) => { re: Float64Array; im: Float64Array }
}

/** The result of a spectral time integration: physical-space snapshots. */
export interface SpectralSolution {
  readonly t: readonly number[]
  readonly x: Float64Array
  /** `[nt, n]` in physical space. */
  readonly u: Tensor
  readonly status: 'success' | 'failed'
}

function fftNumbers(re: Float64Array, im: Float64Array): Cvec {
  const out = fft({ re, im })
  return { re: Float64Array.from(primalOf(out.re).data), im: Float64Array.from(primalOf(out.im).data) }
}

function ifftNumbers(re: Float64Array, im: Float64Array): Cvec {
  const out = ifft({ re, im })
  return { re: Float64Array.from(primalOf(out.re).data), im: Float64Array.from(primalOf(out.im).data) }
}

/**
 * ETDRK4 in Fourier space for a periodic problem with a diagonal (possibly complex) linear part (Cox and Matthews with
 * the Kassam-Trefethen contour coefficients), as a traced algorithm with the L2 norm per step.
 *
 * @param problem - The linear diagonal and the nonlinear term.
 * @param u0 - The initial condition in physical space `[n]`.
 * @param x - The grid (for the result).
 * @param options - Step, final time and thinning.
 * @returns Physical-space snapshots.
 * @concept exponential-integrator
 * @example
 * ```ts
 * import { fourierWavenumbers, spectralEtdrk4 } from 'mdml-engine/dynamics/pde'
 * const n = 8
 * const k = fourierWavenumbers(n, 2 * Math.PI)
 * const x = Float64Array.from({ length: n }, (_, j) => (2 * Math.PI * j) / n)
 * const heat = { linear: { re: Array.from(k, (q) => -q * q), im: new Array(n).fill(0) }, nonlinear: (v) => ({ re: v.re.map(() => 0), im: v.im.map(() => 0) }) }
 * const s = spectralEtdrk4(heat, Array.from(x, Math.sin), x, { dt: 0.1, tEnd: 1 })
 * Math.abs((s.u.data[s.u.data.length - 6] ?? 0) - Math.exp(-1)) < 1e-12 // => true
 * ```
 */
export function spectralEtdrk4(
  problem: SpectralProblem,
  u0: ArrayLike<number>,
  x: Float64Array,
  options: PdeTraceOptions & { readonly dt: number; readonly tEnd: number; readonly every?: number },
): SpectralSolution {
  const n = u0.length
  const h = options.dt
  if (!(h > 0)) throw new MdmlError('value', 'spectralEtdrk4: dt must be positive')
  const lr = Float64Array.from(problem.linear.re)
  const li = Float64Array.from(problem.linear.im)
  const co = etdCoefficients(lr, li, h)
  const apply = (c: Cvec, v: Cvec): Cvec => {
    const re = new Float64Array(n)
    const im = new Float64Array(n)
    for (let k = 0; k < n; k++) {
      const [a, b] = cmul(c.re[k] ?? 0, c.im[k] ?? 0, v.re[k] ?? 0, v.im[k] ?? 0)
      re[k] = a
      im[k] = b
    }
    return { re, im }
  }
  const lin = (...terms: [number, Cvec][]): Cvec => {
    const re = new Float64Array(n)
    const im = new Float64Array(n)
    for (const [w, v] of terms) {
      for (let k = 0; k < n; k++) {
        re[k] = (re[k] ?? 0) + w * (v.re[k] ?? 0)
        im[k] = (im[k] ?? 0) + w * (v.im[k] ?? 0)
      }
    }
    return { re, im }
  }
  const steps = Math.round(options.tEnd / h)
  const every = Math.max(1, options.every ?? 1)
  const N = problem.nonlinear
  const physical = (v: Cvec): Float64Array => ifftNumbers(v.re, v.im).re
  const v0 = fftNumbers(Float64Array.from(u0), new Float64Array(n))
  const algorithm: Algorithm<{ v: Cvec; t: number; ts: number[]; us: Float64Array[]; failed: boolean }> = {
    name: 'spectral-etdrk4',
    config: { dt: h, tEnd: options.tEnd, n },
    schema: channels({ metrics: { t: { description: 'time' }, norm: { description: 'the L2 norm of u' } } }),
    init: () => ({ v: v0, t: 0, ts: [0], us: [Float64Array.from(u0)], failed: false }),
    step: (s, ctx) => {
      const v = s.v
      const Nv = N(v)
      const a = lin([1, apply(co.E2, v)], [1, apply(co.Q, Nv)])
      const Na = N(a)
      const b = lin([1, apply(co.E2, v)], [1, apply(co.Q, Na)])
      const Nb = N(b)
      const c = lin([1, apply(co.E2, a)], [1, apply(co.Q, lin([2, Nb], [-1, Nv]))])
      const Nc = N(c)
      const next = lin(
        [1, apply(co.E, v)],
        [1, apply(co.f1, Nv)],
        [2, apply(co.f2, lin([1, Na], [1, Nb]))],
        [1, apply(co.f3, Nc)],
      )
      const t = (ctx.t + 1) * h
      const failed = !next.re.every((q) => Number.isFinite(q))
      if ((ctx.t + 1) % every === 0 || ctx.t + 1 === steps || failed) {
        s.ts.push(t)
        s.us.push(physical(next))
      }
      return { ...s, v: next, t, failed }
    },
    done: (s) => (s.failed ? 'diverged' : undefined),
    observe: (s) => {
      const u = physical(s.v)
      let q = 0
      for (const w of u) q += w * w
      return { metrics: { t: s.t, norm: Math.sqrt(q / n) } }
    },
  }
  const run = runAlgorithm(algorithm, { maxSteps: steps, ...options })
  const flat = new Float64Array(run.state.us.length * n)
  run.state.us.forEach((u, i) => {
    flat.set(u, i * n)
  })
  return {
    t: run.state.ts,
    x,
    u: fromStorage(flat, [run.state.us.length, n], 'float64'),
    status: run.state.failed ? 'failed' : 'success',
  }
}

/** Options of the named spectral PDE solvers. */
export interface SpectralPdeOptions extends PdeTraceOptions {
  /** Grid points (even). */
  readonly n: number
  /** The period L. */
  readonly length: number
  readonly dt: number
  readonly tEnd: number
  readonly every?: number
  /** Apply the 2/3 rule to the nonlinear term (default true). */
  readonly dealias?: boolean
}

function setup(options: SpectralPdeOptions): { x: Float64Array; k: Float64Array; mask: Float64Array } {
  if (!Number.isInteger(options.n) || options.n < 4 || options.n % 2 !== 0) {
    throw new MdmlError('value', 'spectral solvers need an even n >= 4')
  }
  const x = Float64Array.from({ length: options.n }, (_, j) => (options.length * j) / options.n)
  return {
    x,
    k: fourierWavenumbers(options.n, options.length),
    mask: options.dealias === false ? new Float64Array(options.n).fill(1) : dealiasMask(options.n),
  }
}

function samples(u0: ArrayLike<number> | ((x: number) => number), x: Float64Array): Float64Array {
  return typeof u0 === 'function' ? Float64Array.from(x, (v) => u0(v)) : Float64Array.from(u0)
}

/**
 * The Kuramoto-Sivashinsky equation u_t = -u u_x - u_xx - u_xxxx on a periodic domain (spatiotemporal chaos) by
 * Fourier collocation and ETDRK4, the example of Kassam and Trefethen.
 *
 * @param u0 - Initial values (or a function of x).
 * @param options - Grid size, period, step, final time, thinning and dealiasing.
 * @returns Physical-space snapshots.
 * @concept kuramoto-sivashinsky
 * @example
 * ```ts
 * import { kuramotoSivashinsky } from 'mdml-engine/dynamics/pde'
 * const s = kuramotoSivashinsky((x) => Math.cos(x / 16) * (1 + Math.sin(x / 16)), { n: 64, length: 32 * Math.PI, dt: 0.25, tEnd: 5 })
 * s.status // => 'success'
 * ```
 */
export function kuramotoSivashinsky(
  u0: ArrayLike<number> | ((x: number) => number),
  options: SpectralPdeOptions,
): SpectralSolution {
  const { x, k, mask } = setup(options)
  const n = options.n
  const linear = { re: Array.from(k, (q) => q * q - q ** 4), im: new Array<number>(n).fill(0) }
  const nonlinear = (v: Cvec): Cvec => {
    const u = ifftNumbers(v.re, v.im).re
    const sq = fftNumbers(
      Float64Array.from(u, (w) => w * w),
      new Float64Array(n),
    )
    // -0.5 i k fft(u^2), dealiased.
    return {
      re: Float64Array.from(sq.im, (w, j) => 0.5 * (k[j] ?? 0) * w * (mask[j] ?? 1)),
      im: Float64Array.from(sq.re, (w, j) => -0.5 * (k[j] ?? 0) * w * (mask[j] ?? 1)),
    }
  }
  return spectralEtdrk4({ linear, nonlinear }, samples(u0, x), x, options)
}

/**
 * The Korteweg-de Vries equation u_t + 6 u u_x + u_xxx = 0 on a periodic domain by Fourier collocation and ETDRK4
 * (the dispersive linear part i k^3 is integrated exactly). Solitons u = (c/2) sech^2(sqrt(c)/2 (x - c t)) travel
 * unchanged.
 *
 * @param u0 - Initial values (or a function of x).
 * @param options - Grid size, period, step, final time, thinning and dealiasing.
 * @returns Physical-space snapshots.
 * @concept korteweg-de-vries
 * @example
 * ```ts
 * import { kdv } from 'mdml-engine/dynamics/pde'
 * const sol = (x: number, t: number) => 2 / Math.cosh(x - 10 - 4 * t) ** 2
 * const s = kdv((x) => sol(x, 0), { n: 256, length: 40, dt: 0.0005, tEnd: 0.25 })
 * const j = 72
 * Math.abs((s.u.data[s.u.data.length - 256 + j] ?? 0) - sol((40 * j) / 256, 0.25)) < 1e-6 // => true
 * ```
 */
export function kdv(u0: ArrayLike<number> | ((x: number) => number), options: SpectralPdeOptions): SpectralSolution {
  const { x, k, mask } = setup(options)
  const n = options.n
  const linear = { re: new Array<number>(n).fill(0), im: Array.from(k, (q) => q ** 3) }
  const nonlinear = (v: Cvec): Cvec => {
    const u = ifftNumbers(v.re, v.im).re
    const sq = fftNumbers(
      Float64Array.from(u, (w) => w * w),
      new Float64Array(n),
    )
    // -3 i k fft(u^2)
    return {
      re: Float64Array.from(sq.im, (w, j) => 3 * (k[j] ?? 0) * w * (mask[j] ?? 1)),
      im: Float64Array.from(sq.re, (w, j) => -3 * (k[j] ?? 0) * w * (mask[j] ?? 1)),
    }
  }
  return spectralEtdrk4({ linear, nonlinear }, samples(u0, x), x, options)
}

/**
 * The Allen-Cahn equation u_t = eps u_xx + u - u^3 on a periodic domain by Fourier collocation and ETDRK4: phases
 * u = +-1 separated by fronts of width sqrt(eps) that coarsen slowly.
 *
 * @param u0 - Initial values (or a function of x).
 * @param options - Grid size, period, step, final time, thinning and the interface parameter eps (default 0.01).
 * @returns Physical-space snapshots.
 * @concept allen-cahn
 * @example
 * ```ts
 * import { allenCahn } from 'mdml-engine/dynamics/pde'
 * const s = allenCahn((x) => 0.5 * Math.sin(x), { n: 64, length: 2 * Math.PI, dt: 0.05, tEnd: 10, epsilon: 0.01 })
 * Math.abs(Math.abs(s.u.data[s.u.data.length - 48] ?? 0) - 1) < 1e-2 // => true
 * ```
 */
export function allenCahn(
  u0: ArrayLike<number> | ((x: number) => number),
  options: SpectralPdeOptions & { readonly epsilon?: number },
): SpectralSolution {
  const { x, k } = setup(options)
  const n = options.n
  const eps = options.epsilon ?? 0.01
  const linear = { re: Array.from(k, (q) => 1 - eps * q * q), im: new Array<number>(n).fill(0) }
  const nonlinear = (v: Cvec): Cvec => {
    const u = ifftNumbers(v.re, v.im).re
    const cube = fftNumbers(
      Float64Array.from(u, (w) => -w * w * w),
      new Float64Array(n),
    )
    return cube
  }
  return spectralEtdrk4({ linear, nonlinear }, samples(u0, x), x, options)
}
