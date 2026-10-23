import { maxOf, WhiteboxError } from '../../base/index.ts'
import {
  abs,
  add,
  concat,
  concreteOrThrow,
  div,
  fromStorage,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import type { PPoly } from './piecewise.ts'
import { cubicHermiteSpline } from './splines.ts'

function prepare(x: TensorLike, y: TensorLike, name: string): { xs: number[]; y: Value; h: Tensor; rank: number } {
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, name))
  const [yv] = promoteFloat(y) as [Value]
  if (yv.shape[0] !== xs.length) throw new WhiteboxError('shape', `${name}: x and y must have the same length`)
  if (xs.length < 2) throw new WhiteboxError('value', `${name}: at least two points are needed`)
  for (let i = 1; i < xs.length; i++) {
    if (!((xs[i] ?? 0) > (xs[i - 1] ?? 0))) throw new WhiteboxError('value', `${name}: x must be strictly increasing`)
  }
  const rank = yv.shape.length
  const dx = xs.slice(1).map((v, i) => v - (xs[i] ?? 0))
  const h = fromStorage(Float64Array.from(dx), [dx.length], 'float64')
  return {
    xs,
    y: yv,
    h: rank <= 1 ? h : (reshape(h, [dx.length, ...new Array<number>(rank - 1).fill(1)]) as Tensor),
    rank,
  }
}

const mask = (values: readonly boolean[], like: Value): Tensor =>
  fromStorage(Uint8Array.from(values, Number), like.shape, 'bool')

/**
 * The PCHIP interpolant (scipy.interpolate.PchipInterpolator, Fritsch-Butland): a monotone piecewise cubic whose knot
 * slopes are weighted harmonic means of the neighbouring secants (zero at local extrema), with shape-preserving
 * three-point end slopes. It never overshoots the data.
 *
 * @param x - Knots `[n]`, increasing.
 * @param y - Values `[n, ...]`.
 * @param extrapolate - Extend the end pieces (default true).
 * @returns The piecewise cubic.
 * @concept monotone-interpolation
 * @example
 * ```ts
 * import { pchip } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * // Flat data stays flat (a cubic spline would ring).
 * toArray(pchip([0, 1, 2, 3], [0, 0, 1, 1]).evaluate([0.5, 2.5])) // => [0, 1]
 * ```
 */
export function pchip(x: TensorLike, y: TensorLike, extrapolate = true): PPoly {
  const { xs, y: yv, h } = prepare(x, y, 'pchip')
  const n = xs.length
  const m = div(sub(sliceAxis(yv, 0, 1, n), sliceAxis(yv, 0, 0, n - 1)), h)
  if (n === 2) return cubicHermiteSpline(xs, yv, concat([m, m], 0), extrapolate)
  const mp = toFlat(primalOf(m))
  const width = mp.length / (n - 1)
  const hk = sliceAxis(h, 0, 1, n - 1)
  const hk1 = sliceAxis(h, 0, 0, n - 2)
  const mk = sliceAxis(m, 0, 1, n - 1)
  const mk1 = sliceAxis(m, 0, 0, n - 2)
  // Interior: 1 / d = (w1 / m_{k-1} + w2 / m_k) / (w1 + w2), or 0 where the secants differ in sign or vanish.
  const flat = Array.from({ length: (n - 2) * width }, (_, idx) => {
    const a = mp[idx] ?? 0
    const b = mp[idx + width] ?? 0
    return a === 0 || b === 0 || Math.sign(a) !== Math.sign(b)
  })
  const flatMask = mask(flat, mk)
  const safe = (v: Value): Value => where(flatMask, 1, v)
  const w1 = add(mul(hk, 2), hk1)
  const w2 = add(hk, mul(hk1, 2))
  const harmonic = div(add(w1, w2), add(div(w1, safe(mk1)), div(w2, safe(mk))))
  const interior = where(flatMask, 0, harmonic)
  const edge = (h0: Value, h1: Value, m0: Value, m1: Value): Value => {
    // ((2 h0 + h1) m0 - h0 m1) / (h0 + h1), set to 0 against the secant's sign, capped at 3 m0 when the secants
    // differ in sign.
    const d = div(sub(mul(add(mul(h0, 2), h1), m0), mul(h0, m1)), add(h0, h1))
    const dp = toFlat(primalOf(d))
    const m0p = toFlat(primalOf(m0))
    const m1p = toFlat(primalOf(m1))
    const zero = dp.map((v, i) => Math.sign(v) !== Math.sign(m0p[i] ?? 0))
    const cap = dp.map(
      (v, i) =>
        !zero[i] && Math.sign(m0p[i] ?? 0) !== Math.sign(m1p[i] ?? 0) && Math.abs(v) > Math.abs(3 * (m0p[i] ?? 0)),
    )
    return where(mask(zero, d), 0, where(mask(cap, d), mul(m0, 3), d))
  }
  const start = edge(sliceAxis(h, 0, 0, 1), sliceAxis(h, 0, 1, 2), sliceAxis(m, 0, 0, 1), sliceAxis(m, 0, 1, 2))
  const end = edge(
    sliceAxis(h, 0, n - 2, n - 1),
    sliceAxis(h, 0, n - 3, n - 2),
    sliceAxis(m, 0, n - 2, n - 1),
    sliceAxis(m, 0, n - 3, n - 2),
  )
  return cubicHermiteSpline(xs, yv, concat([start, interior, end], 0), extrapolate)
}

/**
 * The Akima interpolant (scipy.interpolate.Akima1DInterpolator): knot slopes are averages of the neighbouring secants
 * weighted by how much the secants change on the far sides, so outliers affect only nearby pieces. The `makima`
 * variant (modified Akima) adds |m_{i+1} + m_i| / 2 to the weights, which removes overshoot on flat stretches.
 *
 * @param x - Knots `[n]`, increasing.
 * @param y - Values `[n, ...]`.
 * @param method - `akima` (default) or `makima`.
 * @param extrapolate - Extend the end pieces (default true).
 * @returns The piecewise cubic.
 * @example
 * ```ts
 * import { akima } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * item(akima([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]).evaluate(2.5)) // => 2.5
 * ```
 */
export function akima(x: TensorLike, y: TensorLike, method: 'akima' | 'makima' = 'akima', extrapolate = true): PPoly {
  const { xs, y: yv, h } = prepare(x, y, 'akima')
  const n = xs.length
  if (n < 3) throw new WhiteboxError('value', 'akima: at least three points are needed')
  const inner = div(sub(sliceAxis(yv, 0, 1, n), sliceAxis(yv, 0, 0, n - 1)), h)
  // Two extrapolated secants at each end: m[1] = 2 m[2] - m[3], m[0] = 2 m[1] - m[2], and symmetrically.
  const at = (i: number): Value => sliceAxis(inner, 0, i, i + 1)
  const b1 = sub(mul(at(0), 2), at(1))
  const b0 = sub(mul(b1, 2), at(0))
  const e1 = sub(mul(at(n - 2), 2), at(n - 3))
  const e2 = sub(mul(e1, 2), at(n - 2))
  const m = concat([b0, b1, inner, e1, e2], 0)
  const len = n + 3
  const seg = (start: number, stop: number): Value => sliceAxis(m, 0, start, stop)
  const dm = abs(sub(seg(1, len), seg(0, len - 1)))
  let f1: Value = sliceAxis(dm, 0, 2, n + 2)
  let f2: Value = sliceAxis(dm, 0, 0, n)
  if (method === 'makima') {
    const pm = abs(add(seg(1, len), seg(0, len - 1)))
    f1 = add(f1, mul(sliceAxis(pm, 0, 2, n + 2), 0.5))
    f2 = add(f2, mul(sliceAxis(pm, 0, 0, n), 0.5))
  }
  const f12 = add(f1, f2)
  const f12p = toFlat(primalOf(f12))
  const max = maxOf(f12p)
  const small = mask(
    f12p.map((v) => !(v > 1e-9 * max)),
    f12,
  )
  const fallback = mul(add(seg(3, len), seg(0, len - 3)), 0.5)
  const weighted = div(add(mul(f1, seg(1, n + 1)), mul(f2, seg(2, n + 2))), where(small, 1, f12))
  return cubicHermiteSpline(xs, yv, where(small, fallback, weighted), extrapolate)
}
