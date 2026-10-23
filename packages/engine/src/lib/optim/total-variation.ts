import type { Json } from '../base/index.ts'
import { WhiteboxError } from '../base/index.ts'
import {
  add,
  arg,
  asTensor,
  bind,
  concreteOrThrow,
  definePrimitive,
  div,
  fromStorage,
  mul,
  type Primitive,
  primalOf,
  promoteFloat,
  segmentSum,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'

/** Condat's direct algorithm (2013) for argmin_x |x - y|^2 / 2 + lambda sum_k |x_{k+1} - x_k|. */
function condat(input: ArrayLike<number>, lambda: number): Float64Array {
  const width = input.length
  const output = new Float64Array(width)
  if (width === 0) return output
  if (!(lambda > 0)) {
    output.set(Array.from(input))
    return output
  }
  const y = (i: number): number => input[i] ?? 0
  let k = 0
  let k0 = 0
  let umin = lambda
  let umax = -lambda
  let vmin = y(0) - lambda
  let vmax = y(0) + lambda
  let kplus = 0
  let kminus = 0
  const twolambda = 2 * lambda
  const minlambda = -lambda
  for (;;) {
    while (k === width - 1) {
      if (umin < 0) {
        do output[k0++] = vmin
        while (k0 <= kminus)
        k = k0
        kminus = k0
        vmin = y(k0)
        umin = lambda
        umax = vmin + umin - vmax
      } else if (umax > 0) {
        do output[k0++] = vmax
        while (k0 <= kplus)
        k = k0
        kplus = k0
        vmax = y(k0)
        umax = minlambda
        umin = vmax + umax - vmin
      } else {
        vmin += umin / (k - k0 + 1)
        do output[k0++] = vmin
        while (k0 <= k)
        return output
      }
    }
    umin += y(k + 1) - vmin
    if (umin < minlambda) {
      do output[k0++] = vmin
      while (k0 <= kminus)
      k = k0
      kminus = k0
      kplus = k0
      vmin = y(k0)
      vmax = vmin + twolambda
      umin = lambda
      umax = minlambda
      continue
    }
    umax += y(k + 1) - vmax
    if (umax > lambda) {
      do output[k0++] = vmax
      while (k0 <= kplus)
      k = k0
      kminus = k0
      kplus = k0
      vmax = y(k0)
      vmin = vmax - twolambda
      umin = lambda
      umax = minlambda
      continue
    }
    k += 1
    if (umin >= lambda) {
      kminus = k
      vmin += (umin - lambda) / (kminus - k0 + 1)
      umin = lambda
    }
    if (umax <= minlambda) {
      kplus = k
      vmax += (umax + lambda) / (kplus - k0 + 1)
      umax = minlambda
    }
  }
}

/** The constant runs of a TV solution: segment ids, sizes, and d x / d lambda per entry. */
function segments(x: ArrayLike<number>): { ids: Int32Array; count: number; dLambda: Float64Array } {
  const n = x.length
  const ids = new Int32Array(n)
  const starts: number[] = []
  for (let i = 0; i < n; i++) {
    if (i === 0 || x[i] !== x[i - 1]) starts.push(i)
    ids[i] = starts.length - 1
  }
  const dLambda = new Float64Array(n)
  for (const [s, a] of starts.entries()) {
    const b = (starts[s + 1] ?? n) - 1
    const left = a === 0 ? 0 : Math.sign((x[a] ?? 0) - (x[a - 1] ?? 0))
    const right = b === n - 1 ? 0 : Math.sign((x[b + 1] ?? 0) - (x[b] ?? 0))
    const d = -(left - right) / (b - a + 1)
    for (let i = a; i <= b; i++) dLambda[i] = d
  }
  return { ids, count: starts.length, dLambda }
}

/** The derivative of the TV prox along y: averaging over each constant run. */
function averageOverRuns(g: Value, x: Tensor): Value {
  const { ids, count } = segments(x.data as Float64Array)
  const idx = fromStorage(ids, [ids.length], 'int32')
  const sizes = segmentSum(fromStorage(new Float64Array(ids.length).fill(1), [ids.length], 'float64'), idx, count)
  return take(div(segmentSum(g, idx, count), sizes), idx)
}

const tv1dPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'proxTv1d',
  abstractEval: ([y]) => [{ shape: (y as { shape: readonly number[] }).shape, dtype: 'float64' }],
  impl: ([y, lam]) => {
    const yy = y as Tensor
    return [fromStorage(condat(yy.data as Float64Array, toFlat(lam as Tensor)[0] ?? 0), yy.shape, 'float64')]
  },
  vjp: (gs, _inputs, _p, outs) => {
    const g = arg(gs, 0)
    if (g === null) return [null, null]
    const x = primalOf(arg(outs, 0))
    const { dLambda } = segments(x.data as Float64Array)
    return [averageOverRuns(g, x), sum(mul(g, fromStorage(dLambda, x.shape, 'float64')))]
  },
  jvp: (_inputs, ts, _p, outs) => {
    const x = primalOf(arg(outs, 0))
    const [ty, tl] = ts
    const { dLambda } = segments(x.data as Float64Array)
    const fromY = ty === null || ty === undefined ? null : averageOverRuns(ty, x)
    const fromL = tl === null || tl === undefined ? null : mul(fromStorage(dLambda, x.shape, 'float64'), tl)
    if (fromY === null) return [fromL]
    return [fromL === null ? fromY : add(fromY, fromL)]
  },
})

/**
 * The proximal operator of one-dimensional total variation, lambda sum_k |x_{k+1} - x_k| (TV denoising of a signal),
 * by Condat's exact direct algorithm (2013), linear time in practice. Differentiable in y and lambda: the solution is
 * piecewise constant, each run the mean of its inputs shifted by lambda times the jump signs at its ends.
 *
 * @param y - The signal, a vector.
 * @param lambda - The regularisation weight (>= 0).
 * @returns The denoised signal.
 * @concept proximal-operators
 * @example
 * ```ts
 * import { proxTv1d } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(proxTv1d([0, 0, 4, 4], 1)) // => [0.5, 0.5, 3.5, 3.5]
 * ```
 */
export function proxTv1d(y: TensorLike, lambda: TensorLike): Value {
  const [yv, lv] = promoteFloat(y, lambda) as [Value, Value]
  if (yv.shape.length !== 1) throw new WhiteboxError('shape', 'proxTv1d: the signal must be a vector')
  return arg(bind(tv1dPrimitive, [yv, lv], null), 0)
}

/** Options of {@link proxTv2d} and {@link tv2dAlgorithm}. */
export interface Tv2dOptions {
  /** Isotropic (default) or anisotropic (sum of absolute differences) total variation. */
  readonly isotropic?: boolean
  /** Box constraints on the image (Beck and Teboulle's constrained denoising). */
  readonly lower?: number
  readonly upper?: number
  /** Stop once the relative change of the image falls to this (default 1e-10). */
  readonly tol?: number
}

/** State of {@link tv2dAlgorithm}: the dual pair (p, q), the extrapolated pair (r, s) and the image. */
export interface Tv2dState {
  readonly p: Float64Array
  readonly q: Float64Array
  readonly r: Float64Array
  readonly s: Float64Array
  readonly t: number
  readonly x: Float64Array
  readonly change: number
}

/**
 * Two-dimensional total-variation denoising, argmin_X |X - B|_F^2 / 2 + lambda TV(X), by the fast gradient projection
 * method on the dual (Beck and Teboulle 2009), as a traced algorithm. TV uses forward differences with zero beyond the
 * last row and column.
 *
 * @param b - The image `[m, n]` (concrete).
 * @param lambda - The regularisation weight.
 * @param options - Isotropy, box constraints and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { tv2dAlgorithm } from 'whitebox-engine/optim'
 * tv2dAlgorithm([[0, 1], [1, 0]], 0.1).name // => 'tv2d'
 * ```
 */
export function tv2dAlgorithm(b: TensorLike, lambda: number, options: Tv2dOptions = {}): Algorithm<Tv2dState> {
  const image = concreteOrThrow(promoteFloat(b)[0] as Value, 'proxTv2d') as Tensor
  if (image.shape.length !== 2) throw new WhiteboxError('shape', 'proxTv2d: the image must be a matrix')
  const m = image.shape[0] ?? 0
  const n = image.shape[1] ?? 0
  const bv = Float64Array.from(image.data)
  const isotropic = options.isotropic ?? true
  const lo = options.lower ?? -Infinity
  const hi = options.upper ?? Infinity
  const tol = options.tol ?? 1e-10
  // p: (m - 1) x n vertical differences, q: m x (n - 1) horizontal ones.
  const L = (p: Float64Array, q: Float64Array): Float64Array => {
    const out = new Float64Array(m * n)
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        let v = 0
        if (i < m - 1) v += p[i * n + j] ?? 0
        if (i > 0) v -= p[(i - 1) * n + j] ?? 0
        if (j < n - 1) v += q[i * (n - 1) + j] ?? 0
        if (j > 0) v -= q[i * (n - 1) + j - 1] ?? 0
        out[i * n + j] = v
      }
    return out
  }
  const primal = (p: Float64Array, q: Float64Array): Float64Array => {
    const lp = L(p, q)
    return Float64Array.from(bv, (v, i) => Math.min(hi, Math.max(lo, v - lambda * (lp[i] ?? 0))))
  }
  const project = (p: Float64Array, q: Float64Array): void => {
    if (!isotropic) {
      for (let i = 0; i < p.length; i++) p[i] = Math.max(-1, Math.min(1, p[i] ?? 0))
      for (let i = 0; i < q.length; i++) q[i] = Math.max(-1, Math.min(1, q[i] ?? 0))
      return
    }
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        const a = i < m - 1 ? (p[i * n + j] ?? 0) : 0
        const c = j < n - 1 ? (q[i * (n - 1) + j] ?? 0) : 0
        const scale = Math.max(1, Math.hypot(a, c))
        if (i < m - 1) p[i * n + j] = a / scale
        if (j < n - 1) q[i * (n - 1) + j] = c / scale
      }
  }
  return {
    name: 'tv2d',
    config: { lambda, isotropic, tol } as Json,
    schema: channels({
      metrics: { change: { description: 'relative change of the image' } },
      tensors: { x: { role: 'state', description: 'current image' } },
    }),
    init: () => {
      const p = new Float64Array((m - 1) * n)
      const q = new Float64Array(m * (n - 1))
      return { p, q, r: p, s: q, t: 1, x: primal(p, q), change: Infinity }
    },
    step: (st) => {
      const x = primal(st.r, st.s)
      const scale = 1 / (8 * lambda)
      const p = new Float64Array(st.p.length)
      const q = new Float64Array(st.q.length)
      for (let i = 0; i < m - 1; i++)
        for (let j = 0; j < n; j++)
          p[i * n + j] = (st.r[i * n + j] ?? 0) + scale * ((x[i * n + j] ?? 0) - (x[(i + 1) * n + j] ?? 0))
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n - 1; j++)
          q[i * (n - 1) + j] = (st.s[i * (n - 1) + j] ?? 0) + scale * ((x[i * n + j] ?? 0) - (x[i * n + j + 1] ?? 0))
      project(p, q)
      const t = (1 + Math.sqrt(1 + 4 * st.t * st.t)) / 2
      const w = (st.t - 1) / t
      const r = Float64Array.from(p, (v, i) => v + w * (v - (st.p[i] ?? 0)))
      const s = Float64Array.from(q, (v, i) => v + w * (v - (st.q[i] ?? 0)))
      const xNew = primal(p, q)
      let d = 0
      let nx = 0
      for (let i = 0; i < xNew.length; i++) {
        d += ((xNew[i] ?? 0) - (st.x[i] ?? 0)) ** 2
        nx += (xNew[i] ?? 0) ** 2
      }
      return { p, q, r, s, t, x: xNew, change: Math.sqrt(d) / Math.max(Math.sqrt(nx), 1e-300) }
    },
    done: (st) => (st.change <= tol ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { change: st.change },
      tensors: { x: { role: 'state', value: st.x, shape: [m, n] } },
    }),
  }
}

/**
 * The proximal operator of two-dimensional total variation (TV image denoising), by {@link tv2dAlgorithm}. Works on
 * concrete images (no derivative rule).
 *
 * @param b - The image `[m, n]`.
 * @param lambda - The regularisation weight.
 * @param options - Isotropy, box constraints, tolerance, iteration cap (default 2000) and run options.
 * @returns The denoised image.
 * @throws WhiteboxError with code `not-differentiable` for a traced image.
 * @concept proximal-operators
 * @example
 * ```ts
 * import { proxTv2d } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(proxTv2d([[1, 1], [1, 1]], 0.5)) // => [[1, 1], [1, 1]]
 * ```
 */
export function proxTv2d(
  b: TensorLike,
  lambda: number,
  options: Tv2dOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): Tensor {
  const { maxIterations, isotropic, lower, upper, tol, ...run } = options
  const image = asTensor(b)
  if (!(lambda > 0)) return concreteOrThrow(promoteFloat(image)[0] as Value, 'proxTv2d') as Tensor
  const result = runAlgorithm(
    tv2dAlgorithm(image, lambda, {
      ...(isotropic === undefined ? {} : { isotropic }),
      ...(lower === undefined ? {} : { lower }),
      ...(upper === undefined ? {} : { upper }),
      ...(tol === undefined ? {} : { tol }),
    }),
    { ...run, maxSteps: maxIterations ?? 2000 },
  )
  return fromStorage(result.state.x, image.shape, 'float64')
}
