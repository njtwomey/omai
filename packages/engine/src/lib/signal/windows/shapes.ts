import { WhiteboxError } from '../../base/index.ts'
import { i0 } from '../../numerics/special/index.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { constant } from '../_shared.ts'
import { build } from './_build.ts'
import { boxcar, hann, type WindowOptions } from './cosine.ts'

/**
 * The Bartlett (triangular, zero end points) window (scipy.signal.windows.bartlett).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { bartlett } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(bartlett(5)) // => [0, 0.5, 1, 0.5, 0]
 * ```
 */
export function bartlett(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => (n <= (len - 1) / 2 ? (2 * n) / (len - 1) : 2 - (2 * n) / (len - 1))),
  )
}

/**
 * The triangular window with non-zero end points (scipy.signal.windows.triang).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { triang } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(triang(3)) // => [0.5, 1, 0.5]
 * ```
 */
export function triang(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) => {
    const half = Math.floor((len + 1) / 2)
    const w = new Float64Array(len)
    for (let k = 1; k <= half; k++) {
      const v = len % 2 === 0 ? (2 * k - 1) / len : (2 * k) / (len + 1)
      w[k - 1] = v
      w[len - k] = v
    }
    return w
  })
}

/**
 * The Kaiser window (scipy.signal.windows.kaiser): I0(beta sqrt(1 - (2n / (M - 1) - 1)^2)) / I0(beta).
 *
 * @param m - Length.
 * @param beta - Shape (0 is rectangular; 5 is close to Hamming, 8.6 to Blackman).
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @concept window-functions
 * @example
 * ```ts
 * import { kaiser } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(kaiser(3, 0)) // => [1, 1, 1]
 * ```
 */
export function kaiser(m: number, beta: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) => {
    const alpha = (len - 1) / 2
    const args = Float64Array.from(
      { length: len },
      (_, n) => beta * Math.sqrt(Math.max(0, 1 - ((n - alpha) / alpha) ** 2)),
    )
    const num = toFlat(i0(constant(args)))
    const den = toFlat(i0(constant([beta])))[0] ?? 1
    return Float64Array.from(num, (v) => v / den)
  })
}

/**
 * The Kaiser-Bessel-derived window (scipy.signal.windows.kaiser_bessel_derived), for MDCT coding: even length,
 * satisfying the Princen-Bradley condition w_n^2 + w_(n + M/2)^2 = 1.
 *
 * @param m - Length (even).
 * @param beta - Kaiser shape.
 * @returns The window (always symmetric).
 * @throws WhiteboxError with code `value` for an odd length.
 * @example
 * ```ts
 * import { kaiserBesselDerived } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const w = toFlat(kaiserBesselDerived(4, 2))
 * Math.round(((w[0] ?? 0) ** 2 + (w[2] ?? 0) ** 2) * 1e12) / 1e12 // => 1
 * ```
 */
export function kaiserBesselDerived(m: number, beta: number): Tensor {
  if (m === 0) return constant([])
  if (m % 2 !== 0) throw new WhiteboxError('value', 'kaiserBesselDerived: the length must be even')
  const half = m / 2
  const k = toFlat(kaiser(half + 1, beta))
  const cum = new Float64Array(half + 1)
  let acc = 0
  for (let i = 0; i <= half; i++) {
    acc += k[i] ?? 0
    cum[i] = acc
  }
  const w = new Float64Array(m)
  for (let i = 0; i < half; i++) {
    const v = Math.sqrt((cum[i] ?? 0) / acc)
    w[i] = v
    w[m - 1 - i] = v
  }
  return constant(w)
}

/**
 * The Gaussian window (scipy.signal.windows.gaussian): exp(-n^2 / (2 std^2)) about the centre.
 *
 * @param m - Length.
 * @param std - Standard deviation in samples.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { gaussian } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(gaussian(3, 1)).map((v) => Math.round(v * 1e6) / 1e6) // => [0.606531, 1, 0.606531]
 * ```
 */
export function gaussian(m: number, std: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => Math.exp(-((n - (len - 1) / 2) ** 2) / (2 * std * std))),
  )
}

/**
 * The generalised Gaussian window (scipy.signal.windows.general_gaussian): exp(-0.5 |n / sig|^(2p)).
 *
 * @param m - Length.
 * @param p - Shape (1 is Gaussian, 0.5 Laplace-like).
 * @param sig - Width.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { generalGaussian } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(generalGaussian(3, 1, 1)).map((v) => Math.round(v * 1e6) / 1e6) // => [0.606531, 1, 0.606531]
 * ```
 */
export function generalGaussian(m: number, p: number, sig: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => Math.exp(-0.5 * Math.abs((n - (len - 1) / 2) / sig) ** (2 * p))),
  )
}

/**
 * The Tukey (tapered cosine) window (scipy.signal.windows.tukey): flat in the middle, cosine tapers over a fraction
 * alpha of the length; alpha 0 is rectangular and 1 is Hann.
 *
 * @param m - Length.
 * @param alpha - Taper fraction (default 0.5).
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { tukey } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(tukey(5, 0.5)) // => [0, 1, 1, 1, 0]
 * ```
 */
export function tukey(m: number, alpha = 0.5, options: WindowOptions = {}): Tensor {
  if (alpha <= 0) return boxcar(m, options)
  if (alpha >= 1) return hann(m, options)
  return build(m, options.sym ?? true, (len) => {
    const width = Math.floor((alpha * (len - 1)) / 2)
    return Float64Array.from({ length: len }, (_, n) => {
      if (n <= width) return 0.5 * (1 + Math.cos(Math.PI * (-1 + (2 * n) / alpha / (len - 1))))
      if (n >= len - width - 1) return 0.5 * (1 + Math.cos(Math.PI * (-2 / alpha + 1 + (2 * n) / alpha / (len - 1))))
      return 1
    })
  })
}

/**
 * The cosine (sine) window (scipy.signal.windows.cosine): sin(pi (n + 0.5) / M).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { cosine } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(cosine(2)).map((v) => Math.round(v * 1e12) / 1e12) // => [0.707106781187, 0.707106781187]
 * ```
 */
export function cosine(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => Math.sin((Math.PI / len) * (n + 0.5))),
  )
}

/**
 * The Parzen (de la Vallee Poussin) window (scipy.signal.windows.parzen): a piecewise cubic.
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { parzen } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(parzen(3)).map((v) => Math.round(v * 1e12) / 1e12) // => [0.074074074074, 1, 0.074074074074]
 * ```
 */
export function parzen(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, i) => {
      const n = Math.abs(-(len - 1) / 2 + i)
      const r = n / (len / 2)
      return n <= (len - 1) / 4 ? 1 - 6 * r ** 2 + 6 * r ** 3 : 2 * (1 - r) ** 3
    }),
  )
}

/**
 * The Bohman window (scipy.signal.windows.bohman): the convolution of two half-cosines.
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { bohman } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(bohman(3)) // => [0, 1, 0]
 * ```
 */
export function bohman(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => {
      if (n === 0 || n === len - 1) return 0
      const f = Math.abs(-1 + (2 * n) / (len - 1))
      return (1 - f) * Math.cos(Math.PI * f) + Math.sin(Math.PI * f) / Math.PI
    }),
  )
}

/**
 * The modified Bartlett-Hann window (scipy.signal.windows.barthann).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { barthann } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(barthann(3)).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1, 0]
 * ```
 */
export function barthann(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => {
      const f = Math.abs(n / (len - 1) - 0.5)
      return 0.62 - 0.48 * f + 0.38 * Math.cos(2 * Math.PI * f)
    }),
  )
}

/**
 * The exponential (Poisson) window (scipy.signal.windows.exponential): exp(-|n - centre| / tau).
 *
 * @param m - Length.
 * @param options - Centre (default the middle; only for periodic windows, as scipy), decay tau (default 1), symmetry.
 * @returns The window.
 * @example
 * ```ts
 * import { exponential } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(exponential(3)).map((v) => Math.round(v * 1e6) / 1e6) // => [0.367879, 1, 0.367879]
 * ```
 */
export function exponential(
  m: number,
  options: WindowOptions & { readonly center?: number; readonly tau?: number } = {},
): Tensor {
  const sym = options.sym ?? true
  if (sym && options.center !== undefined)
    throw new WhiteboxError('value', 'exponential: a centre is only allowed for periodic windows (as scipy)')
  const tau = options.tau ?? 1
  return build(m, sym, (len) => {
    const c = options.center ?? (len - 1) / 2
    return Float64Array.from({ length: len }, (_, n) => Math.exp(-Math.abs(n - c) / tau))
  })
}

/**
 * The Lanczos (sinc) window (scipy.signal.windows.lanczos): sinc(2n / (M - 1) - 1).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { lanczos } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(lanczos(3)).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1, 0]
 * ```
 */
export function lanczos(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) => {
    const w = new Float64Array(len)
    const sinc = (x: number): number => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))
    const half = len % 2 === 0 ? len / 2 : (len + 1) / 2
    if (len % 2 === 1) w[half - 1] = 1
    for (let k = half; k < len; k++) {
      const v = sinc((2 * k) / (len - 1) - 1)
      w[k] = v
      w[len - 1 - k] = v
    }
    return w
  })
}
