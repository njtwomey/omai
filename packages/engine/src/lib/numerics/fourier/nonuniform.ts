/**
 * Fourier transforms off the uniform grid: non-uniform DFTs of type 1 and 2 computed directly and by Gaussian gridding
 * (the NUFFT of Greengard and Lee, as FINUFFT defines the types), and the chirp-z transform with zoom FFT for
 * evaluating a spectrum on a chosen band (scipy.signal `czt` and `zoom_fft`).
 */

import { MdmlError } from '../../base/index.ts'
import type { Complex } from '../../exposition/index.ts'
import {
  add,
  type ComplexLike,
  type ComplexValue,
  cMul,
  complex,
  concreteOrThrow,
  cos,
  fromStorage,
  matmul,
  mul,
  promoteFloat,
  reshape,
  sin,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { fft, ifft } from './fft.ts'

/** The integer modes k = -floor(M/2) .. ceil(M/2) - 1, as FINUFFT's default ordering. */
export function modes(m: number): number[] {
  return Array.from({ length: m }, (_, i) => i - Math.floor(m / 2))
}

function complexMatmul(a: ComplexValue, b: ComplexValue): ComplexValue {
  return { re: sub(matmul(a.re, b.re), matmul(a.im, b.im)), im: add(matmul(a.re, b.im), matmul(a.im, b.re)) }
}

/**
 * The type-1 non-uniform DFT, directly: f_k = sum_j c_j exp(i s k x_j) for the M modes k = -floor(M/2) ..
 * ceil(M/2) - 1 (FINUFFT's nufft1d1). O(M J); differentiable in the strengths and the points.
 *
 * @param x - Non-uniform points `[J]` (radians, any real values; 2 pi periodic).
 * @param c - Strengths `[J]`, real or complex.
 * @param m - Number of modes M.
 * @param sign - The exponent sign s (default +1).
 * @returns Mode coefficients `[M]`.
 * @concept nonuniform-fourier
 * @example
 * ```ts
 * import { nudft1 } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(nudft1([0], [2], 3).re) // => [2, 2, 2]
 * ```
 */
export function nudft1(x: TensorLike, c: ComplexLike, m: number, sign: 1 | -1 = 1): ComplexValue {
  const [xv] = promoteFloat(x) as [Value]
  const ks = fromStorage(
    Float64Array.from(modes(m), (k) => sign * k),
    [m, 1],
    'float64',
  )
  const angle = mul(ks, reshape(xv, [1, -1]))
  const z = complex(c)
  const col = { re: reshape(z.re, [-1, 1]), im: reshape(z.im, [-1, 1]) }
  const out = complexMatmul({ re: cos(angle), im: sin(angle) }, col)
  return { re: reshape(out.re, [m]), im: reshape(out.im, [m]) }
}

/**
 * The type-2 non-uniform DFT, directly: c_j = sum_k f_k exp(i s k x_j) (FINUFFT's nufft1d2), the adjoint of type 1.
 *
 * @param x - Non-uniform points `[J]`.
 * @param f - Mode coefficients `[M]`.
 * @param sign - The exponent sign s (default -1).
 * @returns Values at the points `[J]`.
 * @example
 * ```ts
 * import { nudft2 } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(nudft2([0, 1], [0, 1, 0]).re) // => [1, 1]
 * ```
 */
export function nudft2(x: TensorLike, f: ComplexLike, sign: 1 | -1 = -1): ComplexValue {
  const [xv] = promoteFloat(x) as [Value]
  const z = complex(f)
  const m = z.re.shape[0] ?? 0
  const ks = fromStorage(
    Float64Array.from(modes(m), (k) => sign * k),
    [1, m],
    'float64',
  )
  const angle = mul(reshape(xv, [-1, 1]), ks)
  const col = { re: reshape(z.re, [-1, 1]), im: reshape(z.im, [-1, 1]) }
  const out = complexMatmul({ re: cos(angle), im: sin(angle) }, col)
  const j = xv.shape[0] ?? 0
  return { re: reshape(out.re, [j]), im: reshape(out.im, [j]) }
}

/**
 * Gaussian gridding parameters (Greengard and Lee 2004). The grid oversamples the M modes by R >= 2 (more when M is
 * small, so the 2 w + 1 point window fits), and the Gaussian's variance tau = pi w / (M^2 R (R - 1/2)) is set from the
 * grid actually used: then the truncation error exp(-pi w (R - 1/2) / R) and the aliasing error
 * exp(-pi w (R - 1) / (R - 1/2)) are both at most exp(-2 pi w / 3).
 */
function gridding(m: number, spread: number): { grid: number; tau: number } {
  const grid = Math.max(2 * spread + 2, 2 * m)
  const ratio = grid / m
  return { grid, tau: (Math.PI * spread) / (m * m * ratio * (ratio - 0.5)) }
}

/** The spreading half-width w for a relative tolerance: the error bound exp(-2 pi w / 3) at oversampling 2. */
function halfWidth(options: { readonly spread?: number; readonly tolerance?: number }, name: string): number {
  if (options.spread !== undefined) {
    if (!Number.isInteger(options.spread) || options.spread < 1)
      throw new MdmlError('value', `${name}: spread must be a positive integer`)
    return options.spread
  }
  const tol = options.tolerance ?? 1e-12
  if (!(tol > 0 && tol < 1)) throw new MdmlError('value', `${name}: tolerance must be in (0, 1)`)
  return Math.max(2, Math.ceil((-3 * Math.log(tol)) / (2 * Math.PI)))
}

/** The spreading matrix S[l, j] = exp(-(xi_l - x_j)^2 / 4 tau) over the nearest grid points (periodic). */
function spreadingMatrix(xs: readonly number[], grid: number, tau: number, spread: number): Tensor {
  const s = new Float64Array(grid * xs.length)
  const h = (2 * Math.PI) / grid
  xs.forEach((x0, j) => {
    const x = ((x0 % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
    const centre = Math.round(x / h)
    for (let o = -spread; o <= spread; o++) {
      const l = centre + o
      const d = l * h - x
      const idx = ((l % grid) + grid) % grid
      s[idx * xs.length + j] = (s[idx * xs.length + j] ?? 0) + Math.exp((-d * d) / (4 * tau))
    }
  })
  return fromStorage(s, [grid, xs.length], 'float64')
}

/**
 * The type-1 NUFFT by Gaussian gridding (Greengard and Lee): spread the strengths onto an oversampled grid with a
 * Gaussian, FFT, and divide out the Gaussian's transform. The relative error (to the largest mode) is at most about
 * the requested tolerance, as FINUFFT's eps: the half-width is w = ceil(3 ln(1 / tol) / 2 pi), 14 at the default
 * 1e-12; below about 1e-13 rounding dominates. The spreading is held as a dense grid-by-points matrix, so the work is
 * O(M J) rather than FINUFFT's O(M log M + J w). Differentiable in the strengths.
 *
 * @param x - Non-uniform points `[J]` (radians).
 * @param c - Strengths `[J]`.
 * @param m - Number of modes.
 * @param options - Exponent sign (default +1), relative tolerance (default 1e-12), or the spreading half-width w
 *   directly (overrides the tolerance).
 * @returns Mode coefficients `[M]`, as {@link nudft1}.
 * @example
 * ```ts
 * import { maxOf } from 'mdml-engine/base'
 * import { nufft1, nudft1 } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * const x = [0.3, 1.7, -2.2, 3]
 * const fast = toArray(nufft1(x, [1, -1, 2, 0.5], 8).re)
 * const slow = toArray(nudft1(x, [1, -1, 2, 0.5], 8).re)
 * maxOf(fast.map((v, i) => Math.abs(v - slow[i]))) < 1e-10 // => true
 * ```
 */
export function nufft1(
  x: TensorLike,
  c: ComplexLike,
  m: number,
  options: { readonly sign?: 1 | -1; readonly spread?: number; readonly tolerance?: number } = {},
): ComplexValue {
  const sign = options.sign ?? 1
  const spread = halfWidth(options, 'nufft1')
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, 'nufft1'))
  const { grid, tau } = gridding(m, spread)
  const s = spreadingMatrix(xs, grid, tau, spread)
  const z = complex(c)
  const spreadRe = reshape(matmul(s, reshape(z.re, [-1, 1])), [grid])
  const spreadIm = reshape(matmul(s, reshape(z.im, [-1, 1])), [grid])
  // sum_l f(xi_l) exp(i s k xi_l): an inverse FFT (unscaled) for s = +1, a forward FFT for s = -1.
  const spectrum =
    sign === 1 ? ifft({ re: spreadRe, im: spreadIm }, grid, -1, 'forward') : fft({ re: spreadRe, im: spreadIm })
  const ks = modes(m)
  const pick = fromStorage(
    Int32Array.from(ks, (k) => ((k % grid) + grid) % grid),
    [m],
    'int32',
  )
  const scale = fromStorage(
    Float64Array.from(ks, (k) => ((2 * Math.PI) / grid) * (Math.exp(k * k * tau) / Math.sqrt(4 * Math.PI * tau))),
    [m],
    'float64',
  )
  return { re: mul(take(spectrum.re, pick, 0), scale), im: mul(take(spectrum.im, pick, 0), scale) }
}

/**
 * The type-2 NUFFT by Gaussian gridding: pre-scale the modes by the Gaussian's inverse transform, inverse FFT onto
 * the oversampled grid, and interpolate to the points with the Gaussian. Differentiable in the modes.
 *
 * @param x - Non-uniform points `[J]`.
 * @param f - Mode coefficients `[M]`.
 * @param options - Exponent sign (default -1), relative tolerance (default 1e-12) or spreading half-width, as
 *   {@link nufft1}.
 * @returns Values `[J]`, as {@link nudft2}.
 * @example
 * ```ts
 * import { maxOf } from 'mdml-engine/base'
 * import { nufft2, nudft2 } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * const x = [0.3, 1.7, -2.2]
 * const fast = toArray(nufft2(x, [1, 2, 3, 4]).im)
 * const slow = toArray(nudft2(x, [1, 2, 3, 4]).im)
 * maxOf(fast.map((v, i) => Math.abs(v - slow[i]))) < 1e-10 // => true
 * ```
 */
export function nufft2(
  x: TensorLike,
  f: ComplexLike,
  options: { readonly sign?: 1 | -1; readonly spread?: number; readonly tolerance?: number } = {},
): ComplexValue {
  const sign = options.sign ?? -1
  const spread = halfWidth(options, 'nufft2')
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, 'nufft2'))
  const z = complex(f)
  const m = z.re.shape[0] ?? 0
  const { grid, tau } = gridding(m, spread)
  const ks = modes(m)
  // u_l = sum_k f_k exp(k^2 tau) exp(i s k xi_l), placed into an oversampled grid.
  const place = new Float64Array(grid * m)
  ks.forEach((k, i) => {
    place[(((k % grid) + grid) % grid) * m + i] = Math.exp(k * k * tau)
  })
  const placement = fromStorage(place, [grid, m], 'float64')
  const gridded = {
    re: reshape(matmul(placement, reshape(z.re, [-1, 1])), [grid]),
    im: reshape(matmul(placement, reshape(z.im, [-1, 1])), [grid]),
  }
  const u = sign === 1 ? ifft(gridded, grid, -1, 'forward') : fft(gridded)
  const s = spreadingMatrix(xs, grid, tau, spread)
  const factor = (2 * Math.PI) / grid / Math.sqrt(4 * Math.PI * tau)
  const st = transpose(s)
  return {
    re: mul(reshape(matmul(st, reshape(u.re, [-1, 1])), [xs.length]), factor),
    im: mul(reshape(matmul(st, reshape(u.im, [-1, 1])), [xs.length]), factor),
  }
}

function transpose(m: Tensor): Tensor {
  const [r = 0, c = 0] = m.shape
  const out = new Float64Array(r * c)
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = m.data[i * c + j] ?? 0
  return fromStorage(out, [c, r], 'float64')
}

const cpow = (z: Complex, p: number): Complex => {
  const r = Math.hypot(z[0], z[1]) ** p
  const t = Math.atan2(z[1], z[0]) * p
  return [r * Math.cos(t), r * Math.sin(t)]
}

/**
 * The chirp-z transform (scipy.signal.czt): X_k = sum_n x_n a^-n w^(n k) for k < m, the z-transform on the spiral
 * z_k = a w^-k, by Bluestein's identity n k = (n^2 + k^2 - (k - n)^2) / 2 and an FFT convolution. Differentiable in x.
 *
 * @param x - Signal `[N]`, real or complex.
 * @param m - Number of outputs (default N).
 * @param w - Ratio between points (default exp(-2 pi i / m), giving the DFT).
 * @param a - Starting point (default 1).
 * @returns `[m]` values.
 * @concept chirp-z
 * @example
 * ```ts
 * import { czt } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(czt([1, 2, 3, 4]).re).map((v) => Math.round(v * 1e10) / 1e10) // => [10, -2, -2, -2]
 * ```
 */
export function czt(x: ComplexLike, m?: number, w?: Complex, a: Complex = [1, 0]): ComplexValue {
  const z = complex(x)
  const n = z.re.shape[0] ?? 0
  const outputs = m ?? n
  if (n < 1 || outputs < 1) throw new MdmlError('value', 'czt: empty input or output')
  const ratio = w ?? [Math.cos((-2 * Math.PI) / outputs), Math.sin((-2 * Math.PI) / outputs)]
  // w^(t^2 / 2) with the angle taken from the principal argument, as scipy.
  const chirp = (t: number): Complex => cpow(ratio, (t * t) / 2)
  const ainv = (t: number): Complex => cpow(a, -t)
  let len = 1
  while (len < n + outputs - 1) len <<= 1
  const pre = Array.from({ length: n }, (_, t) => {
    const c = chirp(t)
    const p = ainv(t)
    return [c[0] * p[0] - c[1] * p[1], c[0] * p[1] + c[1] * p[0]] as const
  })
  const y = cMul(z, {
    re: fromStorage(
      Float64Array.from(pre, (v) => v[0]),
      [n],
      'float64',
    ),
    im: fromStorage(
      Float64Array.from(pre, (v) => v[1]),
      [n],
      'float64',
    ),
  })
  // v_t = w^(-t^2 / 2) for t = -(n - 1) .. outputs - 1, laid out circularly.
  const vr = new Float64Array(len)
  const vi = new Float64Array(len)
  for (let t = -(n - 1); t < outputs; t++) {
    const c = cpow(ratio, -(t * t) / 2)
    const idx = (t + len) % len
    vr[idx] = c[0]
    vi[idx] = c[1]
  }
  const spectrum = cMul(
    fft(y, len),
    fft({ re: fromStorage(vr, [len], 'float64'), im: fromStorage(vi, [len], 'float64') }),
  )
  const conv = ifft(spectrum)
  const post = Array.from({ length: outputs }, (_, k) => chirp(k))
  const head = { re: sliceAxis(conv.re, 0, 0, outputs), im: sliceAxis(conv.im, 0, 0, outputs) }
  return cMul(head, {
    re: fromStorage(
      Float64Array.from(post, (v) => v[0]),
      [outputs],
      'float64',
    ),
    im: fromStorage(
      Float64Array.from(post, (v) => v[1]),
      [outputs],
      'float64',
    ),
  })
}

/**
 * The DFT on a frequency band [f1, f2) with m points (scipy.signal.zoom_fft, endpoint false): a chirp-z transform with
 * a = exp(2 pi i f1 / fs) and w = exp(-2 pi i (f2 - f1) / (m fs)). Fine resolution on a narrow band without a huge FFT.
 *
 * @param x - Signal.
 * @param band - [f1, f2], or f2 alone for [0, f2).
 * @param m - Number of points (default the signal length).
 * @param fs - Sampling frequency (default 2, as scipy).
 * @returns `[m]` values.
 * @example
 * ```ts
 * import { zoomFft } from 'mdml-engine/numerics/fourier'
 * zoomFft([1, 0, 0, 0], [0, 1], 4).re.shape // => [4]
 * ```
 */
export function zoomFft(x: ComplexLike, band: number | readonly [number, number], m?: number, fs = 2): ComplexValue {
  const [f1, f2] = typeof band === 'number' ? [0, band] : band
  const z = complex(x)
  const points = m ?? z.re.shape[0] ?? 0
  const w: Complex = [
    Math.cos((-2 * Math.PI * (f2 - f1)) / (points * fs)),
    Math.sin((-2 * Math.PI * (f2 - f1)) / (points * fs)),
  ]
  const a: Complex = [Math.cos((2 * Math.PI * f1) / fs), Math.sin((2 * Math.PI * f1) / fs)]
  return czt(z, points, w, a)
}
