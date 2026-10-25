/**
 * Applying digital filters to signals, scipy.signal style: `lfilter` for a transfer function, `sosfilt` for
 * second-order sections, their steady-state initial conditions, and zero-phase forward-backward `filtfilt` and
 * `sosfiltfilt`, plus `applyFilter` for any LTI system from the systems domain. Also the non-recursive smoothers:
 * median, Savitzky-Golay (local least-squares polynomials), moving average and exponentially weighted moving average.
 */

import { WhiteboxError } from '../../base/index.ts'
import { convolve } from '../../numerics/convolution/index.ts'
import { lstsq } from '../../numerics/linalg/index.ts'
import { type Lti, toSos, toTf } from '../../systems/index.ts'
import {
  add,
  concat,
  cumsum,
  div,
  flip,
  matmul,
  mul,
  pad,
  primalOf,
  reshape,
  sliceAxis,
  slidingWindowView,
  sort,
  stack,
  sub,
  sum,
  type TensorLike,
  take,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { constant, float, indexTensor, isLti, rowsOf } from '../_shared.ts'
import { convolve1d } from '../image/correlate.ts'
import { iir } from './_iir.ts'

// Applying filters as scipy.signal: lfilter and sosfilt (with initial and final states), forward-backward filtering,
// steady-state initial conditions, and the median, Savitzky-Golay and moving-average smoothers. All are composed
// from one differentiable filtering primitive, so every output is differentiable in the signal and the coefficients.

/** A filtered signal and the filter's final state. */
export interface FilterState {
  readonly y: Value
  readonly zf: Value
}

const ONE = constant([1])

function coefficient(v: TensorLike, name: string): Value {
  const c = float(v)
  if (c.shape.length !== 1 || (c.shape[0] ?? 0) === 0) throw new WhiteboxError('shape', `${name}: coefficients are 1-D`)
  return c
}

function padTo(v: Value, n: number): Value {
  const len = v.shape[0] ?? 0
  return len >= n ? v : pad(v, [[0, n - len]])
}

/** Filter rows `[batch, n]` with optional initial state rows `[batch, k]`; also the final state when asked. */
function filterRows(b: Value, a: Value, x: Value, zi: Value | null, state: boolean): { y: Value; zf: Value | null } {
  const k = Math.max(b.shape[0] ?? 1, a.shape[0] ?? 1) - 1
  if ((primalOf(a).data[0] ?? 0) === 0) throw new WhiteboxError('value', 'lfilter: a[0] must be non-zero')
  const a0 = reshape(sliceAxis(a, 0, 0, 1), [])
  const bn = div(padTo(b, k + 1), a0)
  const an = div(padTo(a, k + 1), a0)
  const [batch = 1, n = 0] = x.shape
  let y = iir(bn, an, x)
  if (zi !== null && k > 0 && n > 0) {
    const len = Math.min(k, n)
    y = add(
      y,
      iir(
        ONE,
        an,
        pad(sliceAxis(zi, 1, 0, len), [
          [0, 0],
          [0, n - len],
        ]),
      ),
    )
  }
  if (!state) return { y, zf: null }
  if (k === 0) return { y, zf: zeros([batch, 0]) }
  const len = Math.min(n, k)
  // zf_i = sum_m (b_(i+1+m) x[n-1-m] - a_(i+1+m) y[n-1-m]) + zi_(i+n).
  const xs = flip(sliceAxis(x, 1, n - len, n), 1)
  const ys = flip(sliceAxis(y, 1, n - len, n), 1)
  const idx: number[] = []
  for (let m = 0; m < len; m++) for (let i = 0; i < k; i++) idx.push(Math.min(i + 1 + m, k + 1))
  const tail = zeros([1])
  const bm = reshape(take(concat([bn, tail]), indexTensor(idx)), [len, k])
  const am = reshape(take(concat([an, tail]), indexTensor(idx)), [len, k])
  let zf = sub(matmul(xs, bm), matmul(ys, am))
  if (zi !== null && n < k)
    zf = add(
      zf,
      pad(sliceAxis(zi, 1, n, k), [
        [0, 0],
        [0, n],
      ]),
    )
  return { y, zf }
}

/** State rows of an initial condition given with the shape of x except `k` along the axis. */
function stateRows(zi: TensorLike, x: Value, axis: number, k: number, name: string): Value {
  const z = float(zi)
  const expected = x.shape.map((s, i) => (i === (axis < 0 ? axis + x.shape.length : axis) ? k : s))
  if (z.shape.length !== expected.length || z.shape.some((s, i) => s !== expected[i]))
    throw new WhiteboxError('shape', `${name}: zi must have shape [${expected.join(', ')}]`)
  return rowsOf(z, axis, name).rows
}

/** Options for {@link lfilter}. */
export interface LfilterOptions {
  /** Axis to filter along (default -1). */
  readonly axis?: number
}

/**
 * Filter a signal with a rational transfer function (scipy.signal.lfilter): a[0] y[n] = sum_k b[k] x[n - k] -
 * sum_(k >= 1) a[k] y[n - k], by the transposed direct form II. With an initial state `zi` (shape of x with
 * max(len(a), len(b)) - 1 along the axis) the final state is returned too. Differentiable in b, a, x and zi.
 *
 * @param b - Numerator coefficients (powers of z^-1).
 * @param a - Denominator coefficients (a[0] non-zero).
 * @param x - Signal.
 * @param options - Axis, and an initial state.
 * @returns The filtered signal, or the signal and the final state when `zi` is given.
 * @throws WhiteboxError with code `value` when a[0] is zero, `shape` for a mis-shaped state.
 * @concept digital-filters
 * @example
 * ```ts
 * import { lfilter } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(lfilter([1], [1, -0.5], [1, 0, 0, 0])) // => [1, 0.5, 0.25, 0.125]
 * toArray(lfilter([1, 1], [2], [1, 2, 3], { zi: [1] }).zf) // => [1.5]
 * ```
 */
export function lfilter(b: TensorLike, a: TensorLike, x: TensorLike, options?: LfilterOptions): Value
export function lfilter(
  b: TensorLike,
  a: TensorLike,
  x: TensorLike,
  options: LfilterOptions & { readonly zi: TensorLike },
): FilterState
export function lfilter(
  b: TensorLike,
  a: TensorLike,
  x: TensorLike,
  options: LfilterOptions & { readonly zi?: TensorLike } = {},
): Value | FilterState {
  const bv = coefficient(b, 'lfilter')
  const av = coefficient(a, 'lfilter')
  const v = float(x)
  const axis = options.axis ?? -1
  const r = rowsOf(v, axis, 'lfilter')
  if (options.zi === undefined) return r.restore(filterRows(bv, av, r.rows, null, false).y)
  const k = Math.max(bv.shape[0] ?? 1, av.shape[0] ?? 1) - 1
  const zi = stateRows(options.zi, v, axis, k, 'lfilter')
  const out = filterRows(bv, av, r.rows, zi, true)
  return { y: r.restore(out.y), zf: r.restore(out.zf ?? zeros([0])) }
}

/**
 * The steady-state initial condition of lfilter's step response (scipy.signal.lfilter_zi): with zi scaled by x[0],
 * a signal that starts at a constant level passes without a start-up transient. Differentiable in b and a.
 *
 * @param b - Numerator.
 * @param a - Denominator.
 * @returns zi `[max(len(a), len(b)) - 1]`.
 * @throws WhiteboxError with code `value` when a[0] is zero or sum(a) is zero (a pole at z = 1).
 * @concept digital-filters
 * @example
 * ```ts
 * import { lfilterZi } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(lfilterZi([0.5, 0.5], [1])) // => [0.5]
 * ```
 */
export function lfilterZi(b: TensorLike, a: TensorLike): Value {
  const bv = coefficient(b, 'lfilterZi')
  const av = coefficient(a, 'lfilterZi')
  if ((primalOf(av).data[0] ?? 0) === 0) throw new WhiteboxError('value', 'lfilterZi: a[0] must be non-zero')
  const n = Math.max(bv.shape[0] ?? 1, av.shape[0] ?? 1)
  const a0 = reshape(sliceAxis(av, 0, 0, 1), [])
  const bn = div(padTo(bv, n), a0)
  const an = div(padTo(av, n), a0)
  const sa = sum(an)
  let total = 0
  for (const v of primalOf(sa).data) total += v
  if (total === 0) throw new WhiteboxError('value', 'lfilterZi: sum(a) is zero (a pole at z = 1)')
  const yInf = div(sum(bn), sa)
  return sliceAxis(cumsum(sub(bn, mul(yInf, an)), 0, true), 0, 1, n)
}

/**
 * Initial conditions from past outputs and inputs (scipy.signal.lfiltic): the state that makes lfilter continue
 * a filtering whose last outputs were y[-1], y[-2], ... and last inputs x[-1], x[-2], ... (most recent first).
 *
 * @param b - Numerator.
 * @param a - Denominator.
 * @param y - Past outputs, most recent first.
 * @param x - Past inputs, most recent first (default zeros).
 * @returns zi `[max(len(a), len(b)) - 1]`.
 * @concept digital-filters
 * @example
 * ```ts
 * import { lfiltic } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(lfiltic([1], [1, -0.5], [2])) // => [1]
 * ```
 */
export function lfiltic(b: TensorLike, a: TensorLike, y: TensorLike, x?: TensorLike): Value {
  const bv = coefficient(b, 'lfiltic')
  const av = coefficient(a, 'lfiltic')
  const n = (av.shape[0] ?? 1) - 1
  const m = (bv.shape[0] ?? 1) - 1
  const k = Math.max(m, n)
  const fit = (v: Value, len: number): Value => {
    const l = v.shape[0] ?? 0
    return l >= len ? sliceAxis(v, 0, 0, len) : pad(v, [[0, len - l]])
  }
  const yv = fit(reshape(float(y), [-1]), n)
  const xv = fit(x === undefined ? zeros([m]) : reshape(float(x), [-1]), m)
  const terms: Value[] = []
  for (let i = 0; i < k; i++) {
    let zi: Value = zeros([])
    if (i < m) zi = add(zi, sum(mul(sliceAxis(bv, 0, i + 1, m + 1), sliceAxis(xv, 0, 0, m - i))))
    if (i < n) zi = sub(zi, sum(mul(sliceAxis(av, 0, i + 1, n + 1), sliceAxis(yv, 0, 0, n - i))))
    terms.push(zi)
  }
  if (terms.length === 0) return zeros([0])
  return div(stack(terms), reshape(sliceAxis(av, 0, 0, 1), []))
}

function sections(sos: TensorLike | Lti, name: string): Value {
  if (isLti(sos)) return toSos(sos).sections
  const s = float(sos as TensorLike)
  if (s.shape.length !== 2 || s.shape[1] !== 6) throw new WhiteboxError('shape', `${name}: sos must be [sections, 6]`)
  return s
}

function sectionCoefficients(s: Value, i: number): { b: Value; a: Value } {
  const row = reshape(sliceAxis(s, 0, i, i + 1), [6])
  return { b: sliceAxis(row, 0, 0, 3), a: sliceAxis(row, 0, 3, 6) }
}

/**
 * Filter with a cascade of second-order sections (scipy.signal.sosfilt), numerically safer than one high-order
 * lfilter. With an initial state `zi` `[sections, ..., 2, ...]` the final state is returned too. Differentiable.
 *
 * @param sos - Sections `[n, 6]` (rows b0 b1 b2 a0 a1 a2), or any LTI system (converted with toSos).
 * @param x - Signal.
 * @param options - Axis, and an initial state.
 * @returns The filtered signal, or the signal and the final state.
 * @concept second-order-sections
 * @example
 * ```ts
 * import { sosfilt } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sosfilt([[1, 0, 0, 1, -0.5, 0]], [1, 0, 0])) // => [1, 0.5, 0.25]
 * ```
 */
export function sosfilt(sos: TensorLike | Lti, x: TensorLike, options?: LfilterOptions): Value
export function sosfilt(
  sos: TensorLike | Lti,
  x: TensorLike,
  options: LfilterOptions & { readonly zi: TensorLike },
): FilterState
export function sosfilt(
  sos: TensorLike | Lti,
  x: TensorLike,
  options: LfilterOptions & { readonly zi?: TensorLike } = {},
): Value | FilterState {
  const s = sections(sos, 'sosfilt')
  const count = s.shape[0] ?? 0
  const v = float(x)
  const axis = options.axis ?? -1
  const r = rowsOf(v, axis, 'sosfilt')
  let rows = r.rows
  const finals: Value[] = []
  const zi = options.zi === undefined ? null : float(options.zi)
  if (zi !== null && zi.shape[0] !== count) throw new WhiteboxError('shape', 'sosfilt: zi needs one state per section')
  for (let i = 0; i < count; i++) {
    const { b, a } = sectionCoefficients(s, i)
    const z =
      zi === null ? null : stateRows(reshape(sliceAxis(zi, 0, i, i + 1), zi.shape.slice(1)), v, axis, 2, 'sosfilt')
    const out = filterRows(b, a, rows, z, zi !== null)
    rows = out.y
    if (out.zf !== null) finals.push(r.restore(out.zf))
  }
  if (zi === null) return r.restore(rows)
  return { y: r.restore(rows), zf: stack(finals) }
}

/**
 * Steady-state initial conditions of a cascade (scipy.signal.sosfilt_zi): each section's lfilter_zi scaled by the
 * DC gain of the sections before it.
 *
 * @param sos - Sections `[n, 6]` or an LTI system.
 * @returns zi `[n, 2]`.
 * @concept second-order-sections
 * @example
 * ```ts
 * import { sosfiltZi } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sosfiltZi([[0.5, 0.5, 0, 1, 0, 0]])) // => [[0.5, 0]]
 * ```
 */
export function sosfiltZi(sos: TensorLike | Lti): Value {
  const s = sections(sos, 'sosfiltZi')
  const rows: Value[] = []
  let scale: Value = constant([1], [])
  for (let i = 0; i < (s.shape[0] ?? 0); i++) {
    const { b, a } = sectionCoefficients(s, i)
    rows.push(mul(scale, lfilterZi(b, a)))
    scale = mul(scale, div(sum(b), sum(a)))
  }
  return stack(rows)
}

/** How filtfilt extends the signal before filtering. */
export type PadType = 'odd' | 'even' | 'constant' | 'none'

/** Options for {@link filtfilt} and {@link sosfiltfilt}. */
export interface FiltfiltOptions {
  readonly axis?: number
  /** Extension at both ends (default 'odd': point reflection about the end samples). */
  readonly padtype?: PadType
  /** Samples added at each end (default 3 max(len(a), len(b)), or 3 times the taps of the cascade). */
  readonly padlen?: number
  /** 'pad' (default) or Gustafsson's method, which chooses initial conditions so forward-backward equals
   * backward-forward filtering (lfilter only). */
  readonly method?: 'pad' | 'gust'
  /** For 'gust': the length of the impulse response kept (default all). */
  readonly irlen?: number
}

/** Extend rows `[batch, n]` by `edge` samples at each end (scipy's odd_ext, even_ext and const_ext). */
function extendRows(x: Value, edge: number, type: PadType): Value {
  if (edge === 0 || type === 'none') return x
  const n = x.shape[1] ?? 0
  const first = sliceAxis(x, 1, 0, 1)
  const last = sliceAxis(x, 1, n - 1, n)
  if (type === 'constant') {
    const idx = indexTensor(new Array<number>(edge).fill(0))
    return concat([take(first, idx, 1), x, take(last, idx, 1)], 1)
  }
  const left = flip(sliceAxis(x, 1, 1, edge + 1), 1)
  const right = flip(sliceAxis(x, 1, n - edge - 1, n - 1), 1)
  if (type === 'even') return concat([left, x, right], 1)
  return concat([sub(mul(first, 2), left), x, sub(mul(last, 2), right)], 1)
}

function edgeOf(options: FiltfiltOptions, ntaps: number, n: number, name: string): number {
  const type = options.padtype ?? 'odd'
  const edge = type === 'none' ? 0 : (options.padlen ?? 3 * ntaps)
  if (n <= edge) throw new WhiteboxError('value', `${name}: the signal must be longer than padlen (${edge})`)
  return edge
}

/** Gustafsson's forward-backward filtering of rows (scipy's _filtfilt_gust). */
function gust(b: Value, a: Value, x: Value, irlen: number | undefined): Value {
  const order = Math.max(b.shape[0] ?? 1, a.shape[0] ?? 1) - 1
  const n = x.shape[1] ?? 0
  if (order === 0) {
    const g = div(sliceAxis(b, 0, 0, 1), sliceAxis(a, 0, 0, 1))
    return mul(x, reshape(mul(g, g), []))
  }
  const m = irlen === undefined || n <= 2 * irlen ? n : irlen
  const e0 = constant(
    Float64Array.from({ length: order }, (_, i) => (i === 0 ? 1 : 0)),
    [1, order],
  )
  const impulse = reshape(filterRows(b, a, zeros([1, m]), e0, false).y, [m])
  const columns = Array.from({ length: order }, (_, k) =>
    k === 0 ? impulse : pad(sliceAxis(impulse, 0, 0, m - k), [[k, 0]]),
  )
  const obs = stack(columns, 1)
  const obsr = flip(obs, 0)
  const s = transpose(filterRows(b, a, transpose(obsr), null, false).y)
  const sr = flip(s, 0)
  const z = zeros([m, order])
  const mm =
    m === n
      ? concat([sub(sr, obs), sub(obsr, s)], 1)
      : concat([concat([sub(sr, obs), z], 1), concat([z, sub(obsr, s)], 1)], 0)
  const run = (v: Value): Value => filterRows(b, a, v, null, false).y
  const back = (v: Value): Value => flip(run(flip(v, 1)), 1)
  const yfb = back(run(x))
  const ybf = run(back(x))
  const d = sub(ybf, yfb)
  const delta = m === n ? d : concat([sliceAxis(d, 1, 0, m), sliceAxis(d, 1, n - m, n)], 1)
  const ic = transpose(lstsq(mm, transpose(delta)))
  const w = m === n ? concat([sr, obsr], 1) : concat([concat([sr, z], 1), concat([z, obsr], 1)], 0)
  const wic = matmul(ic, transpose(w))
  if (m === n) return add(yfb, wic)
  return add(
    add(
      yfb,
      pad(sliceAxis(wic, 1, 0, m), [
        [0, 0],
        [0, n - m],
      ]),
    ),
    pad(sliceAxis(wic, 1, m, 2 * m), [
      [0, 0],
      [n - m, 0],
    ]),
  )
}

/**
 * Zero-phase forward-backward filtering (scipy.signal.filtfilt): extend the signal, filter forwards from the
 * steady-state initial condition, filter the result backwards, and crop. The magnitude response is squared and the
 * phase cancels. Gustafsson's method instead chooses both end conditions by least squares. Differentiable.
 *
 * @param b - Numerator.
 * @param a - Denominator.
 * @param x - Signal (longer than padlen).
 * @param options - Axis, padding type and length, method.
 * @returns The filtered signal, the shape of x.
 * @throws WhiteboxError with code `value` when the signal is not longer than padlen.
 * @concept zero-phase-filtering
 * @example
 * ```ts
 * import { filtfilt } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(filtfilt([0.5, 0.5], [1], [2, 2, 2, 2, 2, 2, 2])).map((v) => Math.round(v * 1e12) / 1e12) // => [2, 2, 2, 2, 2, 2, 2]
 * ```
 */
export function filtfilt(b: TensorLike, a: TensorLike, x: TensorLike, options: FiltfiltOptions = {}): Value {
  const bv = coefficient(b, 'filtfilt')
  const av = coefficient(a, 'filtfilt')
  const r = rowsOf(float(x), options.axis ?? -1, 'filtfilt')
  if (options.method === 'gust') return r.restore(gust(bv, av, r.rows, options.irlen))
  const ntaps = Math.max(bv.shape[0] ?? 1, av.shape[0] ?? 1)
  const edge = edgeOf(options, ntaps, r.n, 'filtfilt')
  const ext = extendRows(r.rows, edge, options.padtype ?? 'odd')
  const zi = reshape(lfilterZi(bv, av), [1, ntaps - 1])
  const len = ext.shape[1] ?? 0
  const fwd = filterRows(bv, av, ext, mul(zi, sliceAxis(ext, 1, 0, 1)), false).y
  const rev = flip(fwd, 1)
  const bwd = flip(filterRows(bv, av, rev, mul(zi, sliceAxis(rev, 1, 0, 1)), false).y, 1)
  return r.restore(sliceAxis(bwd, 1, edge, len - edge))
}

/**
 * Zero-phase forward-backward filtering with second-order sections (scipy.signal.sosfiltfilt).
 *
 * @param sos - Sections `[n, 6]` or an LTI system.
 * @param x - Signal.
 * @param options - Axis, padding type and length.
 * @returns The filtered signal.
 * @concept zero-phase-filtering
 * @example
 * ```ts
 * import { sosfiltfilt } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sosfiltfilt([[0.5, 0.5, 0, 1, 0, 0]], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3])).map((v) => Math.round(v * 1e12) / 1e12) // => [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
 * ```
 */
export function sosfiltfilt(sos: TensorLike | Lti, x: TensorLike, options: FiltfiltOptions = {}): Value {
  const s = sections(sos, 'sosfiltfilt')
  const count = s.shape[0] ?? 0
  const data = primalOf(s).data
  let zb = 0
  let za = 0
  for (let i = 0; i < count; i++) {
    if (data[i * 6 + 2] === 0) zb++
    if (data[i * 6 + 5] === 0) za++
  }
  const ntaps = 2 * count + 1 - Math.min(zb, za)
  const r = rowsOf(float(x), options.axis ?? -1, 'sosfiltfilt')
  const edge = edgeOf(options, ntaps, r.n, 'sosfiltfilt')
  const ext = extendRows(r.rows, edge, options.padtype ?? 'odd')
  const zi = sosfiltZi(s)
  const len = ext.shape[1] ?? 0
  const cascade = (v: Value): Value => {
    const x0 = sliceAxis(v, 1, 0, 1)
    let out = v
    for (let i = 0; i < count; i++) {
      const { b, a } = sectionCoefficients(s, i)
      out = filterRows(b, a, out, mul(x0, reshape(sliceAxis(zi, 0, i, i + 1), [1, 2])), false).y
    }
    return out
  }
  const fwd = cascade(ext)
  const bwd = flip(cascade(flip(fwd, 1)), 1)
  return r.restore(sliceAxis(bwd, 1, edge, len - edge))
}

/**
 * Filter with any LTI system from the systems domain: a discrete transfer function through lfilter, and zeros,
 * poles and gain, state space or second-order sections through sosfilt.
 *
 * @param system - A discrete-time system (tf, zpk, ss or sos).
 * @param x - Signal.
 * @param options - Axis.
 * @returns The filtered signal.
 * @throws WhiteboxError with code `value` for a continuous-time system.
 * @concept digital-filters
 * @example
 * ```ts
 * import { applyFilter } from 'whitebox-engine/signal'
 * import { tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(applyFilter(tf([1, 0], [1, -0.5], { dt: 1 }), [1, 0, 0])) // => [1, 0.5, 0.25]
 * ```
 */
export function applyFilter(system: Lti, x: TensorLike, options: LfilterOptions = {}): Value {
  if (system.dt === 0) throw new WhiteboxError('value', 'applyFilter: the system must be discrete (dt > 0)')
  if (system.kind === 'tf' && system.num.shape.length === 1) {
    // Highest-degree-first polynomials in z become coefficients of z^-1 after left-aligning the degrees.
    const num = system.num
    const den = system.den
    const lead = (den.shape[0] ?? 1) - (num.shape[0] ?? 1)
    const b = lead > 0 ? pad(num, [[lead, 0]]) : num
    return lfilter(b, den, x, options)
  }
  if (system.kind === 'sos') return sosfilt(system, x, options)
  return system.kind === 'tf' ? sosfilt(toTf(system), x, options) : sosfilt(system, x, options)
}

/**
 * A median filter (scipy.signal.medfilt): the median over a window around each sample, with zero padding, in any
 * number of dimensions. Differentiable almost everywhere (through the sort).
 *
 * @param x - Input.
 * @param kernelSize - Odd window size, one per axis or shared (default 3).
 * @returns The filtered input.
 * @throws WhiteboxError with code `value` for an even kernel size.
 * @concept median-filter
 * @example
 * ```ts
 * import { medfilt } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(medfilt([1, 9, 2, 3, 8], 3)) // => [1, 2, 3, 3, 3]
 * ```
 */
export function medfilt(x: TensorLike, kernelSize: number | readonly number[] = 3): Value {
  const v = float(x)
  const rank = v.shape.length
  const k = typeof kernelSize === 'number' ? new Array<number>(rank).fill(kernelSize) : [...kernelSize]
  if (k.length !== rank) throw new WhiteboxError('shape', 'medfilt: one kernel size per axis')
  if (k.some((s) => s % 2 !== 1 || s < 1)) throw new WhiteboxError('value', 'medfilt: kernel sizes must be odd')
  const padded = pad(
    v,
    k.map((s): [number, number] => [(s - 1) / 2, (s - 1) / 2]),
  )
  const windows = slidingWindowView(padded, k, { axis: k.map((_, d) => d) })
  const total = k.reduce((p, s) => p * s, 1)
  const flat = reshape(windows, [...v.shape, total])
  const mid = (total - 1) / 2
  return reshape(sliceAxis(sort(flat, -1), rank, mid, mid + 1), v.shape)
}

/** Options for {@link savgolCoeffs}. */
export interface SavgolCoeffsOptions {
  /** Derivative order (default 0). */
  readonly deriv?: number
  /** Sample spacing for derivatives (default 1). */
  readonly delta?: number
  /** Evaluation point within the window (default the centre). */
  readonly pos?: number
  /** 'conv' (default, reversed for convolution) or 'dot'. */
  readonly use?: 'conv' | 'dot'
}

/** Least-squares polynomial fit matrix: rows of pinv(V) for V[t, p] = t^p. */
function polyfitMatrix(t: readonly number[], order: number): number[][] {
  const v = constant(
    t.flatMap((ti) => Array.from({ length: order + 1 }, (_, p) => ti ** p)),
    [t.length, order + 1],
  )
  const eye = constant(
    Array.from({ length: t.length * t.length }, (_, i) => (Math.floor(i / t.length) === i % t.length ? 1 : 0)),
    [t.length, t.length],
  )
  const p = primalOf(lstsq(v, eye)).data
  return Array.from({ length: order + 1 }, (_, r) =>
    Array.from({ length: t.length }, (_, c) => p[r * t.length + c] ?? 0),
  )
}

function factorial(n: number): number {
  let f = 1
  for (let i = 2; i <= n; i++) f *= i
  return f
}

/**
 * Savitzky-Golay filter coefficients (scipy.signal.savgol_coeffs): the weights that evaluate (a derivative of) the
 * least-squares polynomial of a given order through a window at one point.
 *
 * @param windowLength - Window length.
 * @param polyorder - Polynomial order (less than the window length).
 * @param options - Derivative, spacing, position and use.
 * @returns The coefficients `[windowLength]`.
 * @throws WhiteboxError with code `value` for an order at least the window length or a position outside it.
 * @concept savitzky-golay
 * @example
 * ```ts
 * import { savgolCoeffs } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(savgolCoeffs(5, 2)).map((v) => Math.round(v * 35 * 1e9) / 1e9) // => [-3, 12, 17, 12, -3]
 * ```
 */
export function savgolCoeffs(windowLength: number, polyorder: number, options: SavgolCoeffsOptions = {}): Value {
  if (polyorder >= windowLength)
    throw new WhiteboxError('value', 'savgolCoeffs: polyorder must be less than window_length')
  const deriv = options.deriv ?? 0
  const delta = options.delta ?? 1
  const half = Math.floor(windowLength / 2)
  const pos = options.pos ?? (windowLength % 2 === 0 ? half - 0.5 : half)
  if (!(pos >= 0 && pos < windowLength)) throw new WhiteboxError('value', 'savgolCoeffs: pos must be in the window')
  if (deriv > polyorder) return zeros([windowLength])
  let t = Array.from({ length: windowLength }, (_, i) => i - pos)
  if ((options.use ?? 'conv') === 'conv') t = t.reverse()
  const fit = polyfitMatrix(t, polyorder)
  const scale = factorial(deriv) / delta ** deriv
  return constant(Float64Array.from(fit[deriv] ?? [], (v) => v * scale))
}

/** Options for {@link savgolFilter}. */
export interface SavgolOptions {
  readonly deriv?: number
  readonly delta?: number
  readonly axis?: number
  /** 'interp' (default: fit the edge windows), or an ndimage border mode for the convolution. */
  readonly mode?: 'interp' | 'mirror' | 'constant' | 'nearest' | 'wrap'
  readonly cval?: number
}

/** The matrix mapping a window's samples to (a derivative of) its fitted polynomial at positions `at`. */
function edgeMatrix(
  windowLength: number,
  polyorder: number,
  deriv: number,
  delta: number,
  at: readonly number[],
): Value {
  const t = Array.from({ length: windowLength }, (_, i) => i)
  const fit = polyfitMatrix(t, polyorder)
  const rows = at.map((p) =>
    t.map((_, c) => {
      let s = 0
      for (let q = deriv; q <= polyorder; q++)
        s += ((fit[q]?.[c] ?? 0) * factorial(q) * p ** (q - deriv)) / factorial(q - deriv)
      return s / delta ** deriv
    }),
  )
  return constant(rows.flat(), [at.length, windowLength])
}

/**
 * The Savitzky-Golay smoothing (or differentiating) filter (scipy.signal.savgol_filter): convolve with
 * savgolCoeffs, and in 'interp' mode replace the half-windows at the ends by the polynomial fitted to the first and
 * last windows. Linear in x, hence differentiable.
 *
 * @param x - Signal.
 * @param windowLength - Window length.
 * @param polyorder - Polynomial order.
 * @param options - Derivative, spacing, axis, edge mode and constant.
 * @returns The filtered signal.
 * @throws WhiteboxError with code `value` when, in 'interp' mode, the window is longer than the signal.
 * @concept savitzky-golay
 * @example
 * ```ts
 * import { savgolFilter } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(savgolFilter([1, 4, 9, 16, 25, 36], 5, 2)).map((v) => Math.round(v * 1e9) / 1e9) // => [1, 4, 9, 16, 25, 36]
 * ```
 */
export function savgolFilter(
  x: TensorLike,
  windowLength: number,
  polyorder: number,
  options: SavgolOptions = {},
): Value {
  const deriv = options.deriv ?? 0
  const delta = options.delta ?? 1
  const mode = options.mode ?? 'interp'
  const coeffs = savgolCoeffs(windowLength, polyorder, { deriv, delta })
  const v = float(x)
  const axis = options.axis ?? -1
  if (mode !== 'interp') {
    return convolveAlong(v, coeffs, axis, mode, options.cval ?? 0)
  }
  const r = rowsOf(v, axis, 'savgolFilter')
  const n = r.n
  if (windowLength > n) throw new WhiteboxError('value', 'savgolFilter: window_length must not exceed the signal')
  const half = Math.floor(windowLength / 2)
  const middle = convolveAlong(r.rows, coeffs, 1, 'constant', 0)
  const left = matmul(
    sliceAxis(r.rows, 1, 0, windowLength),
    transpose(
      edgeMatrix(
        windowLength,
        polyorder,
        deriv,
        delta,
        Array.from({ length: half }, (_, i) => i),
      ),
    ),
  )
  const right = matmul(
    sliceAxis(r.rows, 1, n - windowLength, n),
    transpose(
      edgeMatrix(
        windowLength,
        polyorder,
        deriv,
        delta,
        Array.from({ length: half }, (_, i) => windowLength - half + i),
      ),
    ),
  )
  return r.restore(concat([left, sliceAxis(middle, 1, half, n - half), right], 1))
}

function convolveAlong(
  v: Value,
  w: Value,
  axis: number,
  mode: 'mirror' | 'constant' | 'nearest' | 'wrap',
  cval: number,
): Value {
  return convolve1d(v, w, axis, { mode, cval })
}

/**
 * A simple moving average: the convolution with n equal weights 1 / n (numpy.convolve with ones / n).
 *
 * @param x - Signal `[n]`.
 * @param window - Window length.
 * @param mode - 'valid' (default, complete windows only), 'same' or 'full'.
 * @returns The averages.
 * @concept moving-average
 * @example
 * ```ts
 * import { movingAverage } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(movingAverage([1, 2, 3, 4, 5], 2)) // => [1.5, 2.5, 3.5, 4.5]
 * ```
 */
export function movingAverage(x: TensorLike, window: number, mode: 'valid' | 'same' | 'full' = 'valid'): Value {
  if (!Number.isInteger(window) || window < 1)
    throw new WhiteboxError('value', 'movingAverage: the window must be >= 1')
  return convolve(float(x), constant(new Float64Array(window).fill(1 / window)), mode)
}

/**
 * An exponentially weighted moving average (pandas ewm(alpha).mean()): with `adjust` false the recursion y_t = (1 -
 * alpha) y_(t-1) + alpha x_t from y_0 = x_0; with `adjust` true (pandas' default) the weighted mean of all past
 * samples with weights (1 - alpha)^i. Both by lfilter, so differentiable.
 *
 * @param x - Signal (along the last axis).
 * @param alpha - Smoothing factor in (0, 1].
 * @param adjust - Normalise the finite-sum weights (default true).
 * @returns The averages.
 * @concept moving-average
 * @example
 * ```ts
 * import { exponentialMovingAverage } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(exponentialMovingAverage([2, 4, 4], 0.5, false)) // => [2, 3, 3.5]
 * ```
 */
export function exponentialMovingAverage(x: TensorLike, alpha: number, adjust = true): Value {
  if (!(alpha > 0 && alpha <= 1)) throw new WhiteboxError('value', 'exponentialMovingAverage: alpha must be in (0, 1]')
  const v = float(x)
  const r = rowsOf(v, -1, 'exponentialMovingAverage')
  const a = constant([1, alpha - 1])
  if (!adjust) {
    const zi = mul(sliceAxis(r.rows, 1, 0, 1), 1 - alpha)
    return r.restore(filterRows(constant([alpha]), a, r.rows, zi, false).y)
  }
  const num = filterRows(ONE, a, r.rows, null, false).y
  const den = filterRows(ONE, a, constant(new Float64Array(r.n).fill(1), [1, r.n]), null, false).y
  return r.restore(div(num, den))
}
