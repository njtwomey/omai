/**
 * State-space sequence models, the linear recurrences behind S4, Mamba and their relatives: discretising a continuous
 * model (zero-order hold or bilinear), the convolution kernel and recurrent views of the same linear system,
 * elementwise linear recurrences, Mamba's input-dependent selective scan, the delta rule of DeltaNet and the linear
 * recurrent unit (LRU) with its initialisation.
 */

import { OmaiError } from '../../base/index.ts'
import { expm, solve } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import {
  add,
  broadcastTo,
  type ComplexValue,
  concat,
  cos,
  cumsum,
  div,
  einsum,
  exp,
  expandDims,
  eye,
  fromStorage,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sin,
  sliceAxis,
  stack,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  take,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { silu } from './activations.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** A discretised linear state-space model x_{k+1} = Ab x_k + Bb u_k. */
export interface DiscreteSSM {
  /** The state transition `[N, N]` (or `[N]` for a diagonal model). */
  readonly a: Value
  /** The input map `[N, M]` (or `[N]`). */
  readonly b: Value
}

/** Discretisation methods: zero-order hold or the bilinear (Tustin) transform. */
export type Discretisation = 'zoh' | 'bilinear'

/**
 * Discretise a continuous state-space model x' = A x + B u with step dt (S4's first step): zero-order hold, Ab =
 * exp(dt A) and Bb = A^{-1}(Ab - I) B, computed as one matrix exponential of the block matrix [[A, B], [0, 0]] dt (so
 * a singular A is fine); or the bilinear transform, Ab = (I - dt A / 2)^{-1}(I + dt A / 2) and Bb =
 * (I - dt A / 2)^{-1} dt B. Differentiable in A, B and dt.
 *
 * @param a - `[N, N]`.
 * @param b - `[N, M]` (or `[N]` for one input).
 * @param dt - The step.
 * @param method - `zoh` (default) or `bilinear`.
 * @returns Ab and Bb (Bb with the shape of `b`).
 * @concept state-space-models
 * @example
 * ```ts
 * import { discretise } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(discretise([[0]], [1], 0.5).b) // => [0.5]
 * ```
 */
export function discretise(a: TensorLike, b: TensorLike, dt: number, method: Discretisation = 'zoh'): DiscreteSSM {
  const av = fl(a)
  const b0 = fl(b)
  const vector = b0.shape.length === 1
  const bv = vector ? reshape(b0, [-1, 1]) : b0
  const n = av.shape[0] ?? 0
  const m = bv.shape[1] ?? 0
  if (av.shape.length !== 2 || av.shape[1] !== n || bv.shape[0] !== n)
    throw new OmaiError('shape', 'discretise: A must be [N, N] and B [N, M]')
  let ab: Value
  let bb: Value
  if (method === 'zoh') {
    const top = concat([mul(av, dt), mul(bv, dt)], 1)
    const block = concat([top, zeros([m, n + m])], 0)
    const e = expm(block)
    ab = sliceAxis(sliceAxis(e, 0, 0, n), 1, 0, n)
    bb = sliceAxis(sliceAxis(e, 0, 0, n), 1, n, n + m)
  } else {
    const i = eye(n)
    const left = sub(i, mul(av, dt / 2))
    ab = solve(left, add(i, mul(av, dt / 2)))
    bb = solve(left, mul(bv, dt))
  }
  return { a: ab, b: vector ? reshape(bb, [n]) : bb }
}

/**
 * Discretise a diagonal state-space model (S4D, Mamba): with A = diag(lambda) the forms are elementwise: zero-order
 * hold Ab = exp(dt lambda), Bb = (exp(dt lambda) - 1) / lambda B (dt B where lambda = 0); bilinear Ab = (1 + dt
 * lambda / 2) / (1 - dt lambda / 2), Bb = dt / (1 - dt lambda / 2) B.
 *
 * @param lambda - The diagonal `[N]` (real; negative for stability).
 * @param b - `[N]` (or `[N, M]`).
 * @param dt - The step (a number, or `[N]` per state).
 * @param method - `zoh` (default) or `bilinear`.
 * @returns Ab `[N]` and Bb.
 * @concept state-space-models
 * @example
 * ```ts
 * import { discretiseDiagonal } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(discretiseDiagonal([0], [2], 0.5).b) // => [1]
 * ```
 */
export function discretiseDiagonal(
  lambda: TensorLike,
  b: TensorLike,
  dt: TensorLike,
  method: Discretisation = 'zoh',
): DiscreteSSM {
  const l = fl(lambda)
  const bv = fl(b)
  const d = fl(dt)
  const z = mul(l, d)
  const col = (v: Value): Value => (bv.shape.length === 2 ? expandDims(v, 1) : v)
  if (method === 'bilinear') {
    const den = sub(1, div(z, 2))
    return { a: div(add(1, div(z, 2)), den), b: mul(bv, col(div(d, den))) }
  }
  const ab = exp(z)
  // (e^z - 1) / lambda, with its limit dt at lambda = 0 (the safe denominator keeps the gradient finite there).
  const zero = fromStorage(
    Uint8Array.from(primalOf(l).data, (v) => (v === 0 ? 1 : 0)),
    l.shape,
    'bool',
  )
  const safe = where(zero, 1, l)
  const factor = where(zero, add(mul(l, 0), d), div(sub(ab, 1), safe))
  return { a: ab, b: mul(bv, col(factor)) }
}

/**
 * The convolution kernel of a discrete single-input single-output SSM (S4): K_k = C Ab^k Bb for k = 0 .. L - 1, so
 * running the recurrence over L steps is a causal convolution with K. For a diagonal Ab the kernel is a Vandermonde
 * product, K_k = sum_n C_n Bb_n Ab_n^k.
 *
 * @param ssm - The discrete model (Ab `[N, N]` or `[N]`, Bb `[N]`).
 * @param c - The output map `[N]`.
 * @param length - L.
 * @returns K `[L]`.
 * @concept state-space-models
 * @reference Gu, Goel and Re (2022), Efficiently modeling long sequences with structured state spaces.
 * @example
 * ```ts
 * import { ssmKernel } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(ssmKernel({ a: [0.5], b: [1] }, [1], 3)) // => [1, 0.5, 0.25]
 * ```
 */
export function ssmKernel(
  ssm: { readonly a: TensorLike; readonly b: TensorLike },
  c: TensorLike,
  length: number,
): Value {
  const a = fl(ssm.a)
  const b = fl(ssm.b)
  const cv = fl(c)
  if (a.shape.length === 1) {
    // Vandermonde: [L, N] powers Ab^k = exp(k log Ab) would fail for negative Ab, so build them by products.
    const powers: Value[] = []
    let p: Value = add(mul(a, 0), 1)
    for (let k = 0; k < length; k++) {
      powers.push(p)
      p = mul(p, a)
    }
    return sum(mul(stack(powers, 0), mul(cv, b)), 1)
  }
  const out: Value[] = []
  let v = reshape(b, [-1, 1])
  for (let k = 0; k < length; k++) {
    out.push(reshape(matmul(reshape(cv, [1, -1]), v), []))
    v = matmul(a, v)
  }
  return stack(out, 0)
}

/**
 * Causal convolution of sequences with a kernel: y_t = sum_{k <= t} K_k u_{t - k} + D u_t (the convolutional view of
 * an LTI state-space model). Built as a product with the Toeplitz matrix of K, so differentiable in u, K and D.
 *
 * @param u - Inputs `[..., L]`.
 * @param kernel - K `[L]` (or shorter: missing taps are zero).
 * @param d - The skip term D (default 0).
 * @returns `[..., L]`.
 * @example
 * ```ts
 * import { causalConvolution } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(causalConvolution([1, 0, 0], [1, 0.5, 0.25])) // => [1, 0.5, 0.25]
 * ```
 */
export function causalConvolution(u: TensorLike, kernel: TensorLike, d: TensorLike = 0): Value {
  const uv = fl(u)
  const k = fl(kernel)
  const l = uv.shape[uv.shape.length - 1] ?? 0
  const kl = k.shape[0] ?? 0
  // T[t, s] = K[t - s] for 0 <= t - s < kl.
  const idx = new Int32Array(l * l)
  const valid = new Uint8Array(l * l)
  for (let t = 0; t < l; t++) {
    for (let s = 0; s < l; s++) {
      const lag = t - s
      if (lag >= 0 && lag < kl) {
        idx[t * l + s] = lag
        valid[t * l + s] = 1
      }
    }
  }
  const toeplitz = where(
    fromStorage(valid, [l, l], 'bool'),
    reshape(take(k, fromStorage(idx, [l * l], 'int32'), 0), [l, l]),
    0,
  )
  return add(matmul(uv, swapaxes(toeplitz, 0, 1)), mul(uv, fl(d)))
}

/**
 * Causal convolution of multichannel sequences with one kernel per channel: y[b, t, h] = sum_{s <= t} K[h, t - s]
 * u[b, s, h] (the convolutional view of H independent state-space models, and the long convolution of Hyena). Built
 * from the Toeplitz matrices of the kernels, so differentiable in u and K.
 *
 * @param u - Inputs `[B, L, H]`.
 * @param kernel - Kernels `[H, K]` (K <= L taps; missing taps are zero).
 * @returns `[B, L, H]`.
 * @concept state-space-models
 * @example
 * ```ts
 * import { channelCausalConvolution } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(channelCausalConvolution([[[1], [0], [0]]], [[1, 0.5, 0.25]])) // => [[[1], [0.5], [0.25]]]
 * ```
 */
export function channelCausalConvolution(u: TensorLike, kernel: TensorLike): Value {
  const uv = fl(u)
  const k = fl(kernel)
  const [, l = 0, h = 1] = uv.shape
  const taps = k.shape[1] ?? 0
  const idx = new Int32Array(l * l)
  const valid = new Uint8Array(l * l)
  for (let t = 0; t < l; t++)
    for (let s = 0; s <= t; s++) {
      if (t - s < taps) {
        idx[t * l + s] = t - s
        valid[t * l + s] = 1
      }
    }
  const toeplitz = where(
    fromStorage(valid, [l, l], 'bool'),
    reshape(take(k, fromStorage(idx, [l * l], 'int32'), 1), [h, l, l]),
    0,
  )
  return einsum('hts,bsh->bth', toeplitz, uv)
}

/**
 * Run a discrete single-input single-output SSM as a recurrence: x_t = Ab x_{t-1} + Bb u_t, y_t = C x_t + D u_t
 * (from x_{-1} = 0); equal to {@link causalConvolution} with {@link ssmKernel}.
 *
 * @param ssm - Ab `[N, N]` or `[N]`, Bb `[N]`.
 * @param c - `[N]`.
 * @param u - Inputs `[L]`.
 * @param d - The skip term (default 0).
 * @returns `[L]`.
 * @example
 * ```ts
 * import { ssmRecurrence } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(ssmRecurrence({ a: [0.5], b: [1] }, [1], [1, 0, 0])) // => [1, 0.5, 0.25]
 * ```
 */
export function ssmRecurrence(
  ssm: { readonly a: TensorLike; readonly b: TensorLike },
  c: TensorLike,
  u: TensorLike,
  d: TensorLike = 0,
): Value {
  const a = fl(ssm.a)
  const b = fl(ssm.b)
  const cv = fl(c)
  const uv = fl(u)
  const l = uv.shape[0] ?? 0
  let x: Value = zeros([b.shape[0] ?? 0])
  const ys: Value[] = []
  for (let t = 0; t < l; t++) {
    const ut = reshape(sliceAxis(uv, 0, t, t + 1), [])
    const ax = a.shape.length === 1 ? mul(a, x) : reshape(matmul(a, reshape(x, [-1, 1])), [-1])
    x = add(ax, mul(b, ut))
    ys.push(add(sum(mul(cv, x)), mul(ut, fl(d))))
  }
  return stack(ys, 0)
}

/**
 * The linear recurrence h_t = a_t h_{t-1} + b_t, elementwise, along the leading (time) axis (the core of diagonal
 * SSMs, linear RNNs and gated linear attention).
 *
 * @param a - Decays `[T, ...]` broadcast against b (or `[1, ...]`: one decay for every step).
 * @param b - Inputs `[T, ...]`.
 * @param h0 - The initial state (default zeros).
 * @returns Every state `[T, ...]`.
 * @example
 * ```ts
 * import { linearRecurrence } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(linearRecurrence([[0.5], [0.5], [0.5]], [[1], [1], [1]])) // => [[1], [1.5], [1.75]]
 * ```
 */
export function linearRecurrence(a: TensorLike, b: TensorLike, h0?: TensorLike): Value {
  const av = fl(a)
  const bv = fl(b)
  const t = bv.shape[0] ?? 0
  let h: Value = h0 === undefined ? zeros(bv.shape.slice(1)) : fl(h0)
  const out: Value[] = []
  const at = (v: Value, i: number): Value => reshape(sliceAxis(v, 0, i, i + 1), v.shape.slice(1))
  const shared = av.shape[0] === 1
  for (let i = 0; i < t; i++) {
    h = add(mul(at(av, shared ? 0 : i), h), at(bv, i))
    out.push(h)
  }
  return stack(out, 0)
}

/** One doubling round of the Hillis-Steele scan of the affine maps h -> a h + b, along an axis. */
function scanRounds<T extends { readonly shape: readonly number[] }>(
  a: T,
  b: T,
  axis: number,
  ops: {
    readonly slice: (x: T, lo: number, hi: number) => T
    readonly cat: (parts: readonly T[]) => T
    readonly combine: (aTail: T, aPrev: T, bPrev: T, bTail: T) => { readonly a: T; readonly b: T }
  },
): T {
  const n = a.shape[axis] ?? 0
  let av = a
  let bv = b
  for (let d = 1; d < n; d *= 2) {
    const next = ops.combine(ops.slice(av, d, n), ops.slice(av, 0, n - d), ops.slice(bv, 0, n - d), ops.slice(bv, d, n))
    av = ops.cat([ops.slice(av, 0, d), next.a])
    bv = ops.cat([ops.slice(bv, 0, d), next.b])
  }
  return bv
}

/**
 * The linear recurrence h_t = a_t h_{t-1} + b_t as a parallel (associative) scan: ceil(log2 T) rounds in which every
 * step composes with the one d steps earlier, instead of T sequential steps. The result equals
 * {@link linearRecurrence} (the same sums in a different order); this is how S5 and Mamba's kernel run on parallel
 * hardware.
 *
 * @param a - Decays, broadcast against b.
 * @param b - Inputs.
 * @param axis - The time axis (default 0).
 * @param h0 - The initial state (the shape of b without the time axis; default zeros).
 * @returns Every state, with the shape of b.
 * @concept state-space-models
 * @example
 * ```ts
 * import { parallelScan } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(parallelScan([0.5, 0.5, 0.5], [1, 1, 1])) // => [1, 1.5, 1.75]
 * ```
 */
export function parallelScan(a: TensorLike, b: TensorLike, axis = 0, h0?: TensorLike): Value {
  const bv0 = fl(b)
  const ax = axis < 0 ? axis + bv0.shape.length : axis
  const av = broadcastTo(fl(a), bv0.shape)
  let bv = bv0
  if (h0 !== undefined) {
    // Fold the initial state into the first input: b_0 <- a_0 h0 + b_0.
    const first = (v: Value): Value => sliceAxis(v, ax, 0, 1)
    const start = add(mul(first(av), expandDims(fl(h0), ax)), first(bv))
    bv = concat([start, sliceAxis(bv, ax, 1, bv.shape[ax] ?? 1)], ax)
  }
  return scanRounds<Value>(av, bv, ax, {
    slice: (x, lo, hi) => sliceAxis(x, ax, lo, hi),
    cat: (parts) => concat(parts, ax),
    combine: (aTail, aPrev, bPrev, bTail) => ({ a: mul(aTail, aPrev), b: add(mul(aTail, bPrev), bTail) }),
  })
}

/**
 * {@link parallelScan} for a complex diagonal recurrence h_t = a_t h_{t-1} + b_t (S5, the LRU), with real and
 * imaginary parts as separate tensors.
 *
 * @param a - Complex decays, broadcast against b.
 * @param b - Complex inputs.
 * @param axis - The time axis (default 0).
 * @param h0 - The initial state (optional).
 * @returns Every state.
 * @concept state-space-models
 * @example
 * ```ts
 * import { parallelScanComplex } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(parallelScanComplex({ re: [0, 0], im: [1, 1] }, { re: [1, 0], im: [0, 0] }).re) // => [1, 0]
 * ```
 */
export function parallelScanComplex(
  a: { readonly re: TensorLike; readonly im: TensorLike },
  b: { readonly re: TensorLike; readonly im: TensorLike },
  axis = 0,
  h0?: { readonly re: TensorLike; readonly im: TensorLike },
): ComplexValue {
  const br = fl(b.re)
  const bi = fl(b.im)
  const ax = axis < 0 ? axis + br.shape.length : axis
  const ar = broadcastTo(fl(a.re), br.shape)
  const ai = broadcastTo(fl(a.im), br.shape)
  let cr = br
  let ci = bi
  if (h0 !== undefined) {
    const first = (v: Value): Value => sliceAxis(v, ax, 0, 1)
    const hr = expandDims(fl(h0.re), ax)
    const hi = expandDims(fl(h0.im), ax)
    const rest = (v: Value): Value => sliceAxis(v, ax, 1, v.shape[ax] ?? 1)
    cr = concat([add(sub(mul(first(ar), hr), mul(first(ai), hi)), first(br)), rest(br)], ax)
    ci = concat([add(add(mul(first(ar), hi), mul(first(ai), hr)), first(bi)), rest(bi)], ax)
  }
  // Pack the real and imaginary parts on a leading axis so that one scan carries both.
  const pack = (re: Value, im: Value): Value => stack([re, im], 0)
  const A = pack(ar, ai)
  const Bp = pack(cr, ci)
  const out = scanRounds<Value>(A, Bp, ax + 1, {
    slice: (x, lo, hi) => sliceAxis(x, ax + 1, lo, hi),
    cat: (parts) => concat(parts, ax + 1),
    combine: (aTail, aPrev, bPrev, bTail) => {
      const [tr, ti, pr, pi, qr, qi, sr, si] = [
        part2(aTail, 0),
        part2(aTail, 1),
        part2(aPrev, 0),
        part2(aPrev, 1),
        part2(bPrev, 0),
        part2(bPrev, 1),
        part2(bTail, 0),
        part2(bTail, 1),
      ]
      return {
        a: pack(sub(mul(tr, pr), mul(ti, pi)), add(mul(tr, pi), mul(ti, pr))),
        b: pack(add(sub(mul(tr, qr), mul(ti, qi)), sr), add(add(mul(tr, qi), mul(ti, qr)), si)),
      }
    },
  })
  return { re: part2(out, 0), im: part2(out, 1) }
}

/** Part k of a packed pair, keeping the other axes. */
function part2(x: Value, k: number): Value {
  return reshape(sliceAxis(x, 0, k, k + 1), x.shape.slice(1))
}

/** Inputs of {@link selectiveScan}. */
export interface SelectiveScanInput {
  /** Inputs `[B, L, D]`. */
  readonly x: TensorLike
  /** Positive steps `[B, L, D]` (after softplus). */
  readonly dt: TensorLike
  /** The diagonal state matrix `[D, N]` (negative entries; Mamba's -exp(A_log)). */
  readonly a: TensorLike
  /** Input-dependent input maps `[B, L, N]`. */
  readonly b: TensorLike
  /** Input-dependent output maps `[B, L, N]`. */
  readonly c: TensorLike
  /** The skip `[D]` (optional). */
  readonly d?: TensorLike
  /** A gate `[B, L, D]`: the output is multiplied by silu(z) (optional). */
  readonly z?: TensorLike
  /** The state before the first step `[B, D, N]` (default zeros): a scan resumed from an earlier one. */
  readonly h0?: TensorLike
}

/**
 * The selective scan of Mamba (Gu and Dao 2023): a diagonal SSM per channel whose step, input and output maps depend
 * on the input, h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t (zero-order hold of A, Euler of B, as the reference
 * implementation) and y_t = C_t h_t + D x_t, optionally gated by silu(z). A sequential scan; differentiable in
 * every input.
 *
 * @param input - x, dt, A, B, C, and optional D and z.
 * @returns y `[B, L, D]` and the final state `[B, D, N]`.
 * @concept state-space-models
 * @reference Gu and Dao (2023), Mamba: linear-time sequence modeling with selective state spaces.
 * @example
 * ```ts
 * import { selectiveScan } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(selectiveScan({ x: [[[1], [0]]], dt: [[[1], [1]]], a: [[-1]], b: [[[1], [1]]], c: [[[1], [1]]] }).y) // => [[[1], [0.36787944117144233]]]
 * ```
 */
export function selectiveScan(input: SelectiveScanInput): { readonly y: Value; readonly state: Value } {
  const x = fl(input.x)
  const dt = fl(input.dt)
  const a = fl(input.a)
  const b = fl(input.b)
  const c = fl(input.c)
  const [bs = 1, l = 0, dim = 1] = x.shape
  const n = a.shape[1] ?? 1
  let h: Value = input.h0 === undefined ? zeros([bs, dim, n]) : fl(input.h0)
  const ys: Value[] = []
  const step = (v: Value, t: number): Value => reshape(sliceAxis(v, 1, t, t + 1), [bs, -1])
  for (let t = 0; t < l; t++) {
    const dtt = expandDims(step(dt, t), 2) // [B, D, 1]
    const xt = expandDims(step(x, t), 2) // [B, D, 1]
    const bt = expandDims(step(b, t), 1) // [B, 1, N]
    const ct = expandDims(step(c, t), 1) // [B, 1, N]
    h = add(mul(exp(mul(dtt, a)), h), mul(mul(dtt, bt), xt))
    ys.push(sum(mul(h, ct), 2)) // [B, D]
  }
  let y = stack(ys, 1)
  if (input.d !== undefined) y = add(y, mul(x, fl(input.d)))
  if (input.z !== undefined) y = mul(y, silu(input.z))
  return { y, state: h }
}

/** Inputs of {@link ssd}. */
export interface SsdInput {
  /** Inputs `[B, L, H, P]`: H heads of width P. */
  readonly x: TensorLike
  /** Positive steps `[B, L, H]`. */
  readonly dt: TensorLike
  /** The scalar (negative) state matrix of each head `[H]`: A = a I. */
  readonly a: TensorLike
  /** Input maps `[B, L, N]`, shared by the heads. */
  readonly b: TensorLike
  /** Output maps `[B, L, N]`, shared by the heads. */
  readonly c: TensorLike
  /** The state before the first step `[B, H, P, N]` (default zeros). */
  readonly h0?: TensorLike
}

/** The three algorithms of structured state-space duality. */
export type SsdForm = 'recurrent' | 'quadratic' | 'chunked'

/**
 * Structured state-space duality (Mamba-2; Dao and Gu 2024): the selective SSM whose state matrix is a scalar times the
 * identity per head, `S_t = exp(dt_t a) S_{t-1} + dt_t x_t B_t^T` and `y_t = S_t C_t`, computed in one of three
 * equivalent ways. The `recurrent` form steps through time; the `quadratic` form is masked attention,
 * `y = (L . (C B^T)) (dt x)` with the decay mask `L[t, s] = exp(sum_{r in (s, t]} dt_r a)` (queries C, keys B, values
 * dt x); the `chunked` form applies the quadratic form inside chunks of a given length and the recurrence between
 * them. All three give the same outputs and final state.
 *
 * @param input - x, dt, a, b, c and optionally the initial state.
 * @param form - The algorithm (default `chunked`).
 * @param chunk - The chunk length of the chunked form (default 8).
 * @returns The outputs `[B, L, H, P]` and the final state `[B, H, P, N]`.
 * @concept state-space-models
 * @reference Dao and Gu (2024), Transformers are SSMs: generalized models and efficient algorithms through structured
 *   state space duality.
 * @example
 * ```ts
 * import { ssd } from 'omai-engine/nn'
 * import { toFlat } from 'omai-engine/tensor'
 * const input = { x: [[[[1]], [[0]]]], dt: [[[1], [1]]], a: [-1], b: [[[1], [1]]], c: [[[1], [1]]] }
 * toFlat(ssd(input, 'quadratic').y) // => [1, 0.36787944117144233]
 * ```
 */
export function ssd(
  input: SsdInput,
  form: SsdForm = 'chunked',
  chunk = 8,
): { readonly y: Value; readonly state: Value } {
  const x = fl(input.x)
  const dt = fl(input.dt)
  const a = fl(input.a)
  const b = fl(input.b)
  const c = fl(input.c)
  const [bs = 1, l = 0, heads = 1, width = 1] = x.shape
  const n = b.shape[2] ?? 1
  const h0 = input.h0 === undefined ? zeros([bs, heads, width, n]) : fl(input.h0)
  if (form === 'recurrent') {
    let s = h0
    const ys: Value[] = []
    for (let t = 0; t < l; t++) {
      const step = (v: Value, shape: number[]): Value => reshape(sliceAxis(v, 1, t, t + 1), shape)
      const dtt = step(dt, [bs, heads]) // [B, H]
      const decay = reshape(exp(mul(dtt, a)), [bs, heads, 1, 1])
      const xt = step(x, [bs, heads, width]) // [B, H, P]
      const write = mul(
        mul(expandDims(xt, 3), reshape(dtt, [bs, heads, 1, 1])),
        reshape(step(b, [bs, n]), [bs, 1, 1, n]),
      )
      s = add(mul(decay, s), write)
      ys.push(sum(mul(s, reshape(step(c, [bs, n]), [bs, 1, 1, n])), 3)) // [B, H, P]
    }
    return { y: stack(ys, 1), state: s }
  }
  if (form === 'chunked' && l > chunk) {
    let s = h0
    const ys: Value[] = []
    for (let lo = 0; lo < l; lo += chunk) {
      const hi = Math.min(l, lo + chunk)
      const cut = (v: Value): Value => sliceAxis(v, 1, lo, hi)
      const r = ssd({ x: cut(x), dt: cut(dt), a, b: cut(b), c: cut(c), h0: s }, 'quadratic')
      ys.push(r.y)
      s = r.state
    }
    return { y: concat(ys, 1), state: s }
  }
  // Quadratic: the decay mask from the cumulative log-decay (<= 0), cum[t] = sum_{r <= t} dt_r a.
  const logDecay = mul(dt, a) // [B, L, H]
  const cum = cumsum(logDecay, 1)
  const cumT = swapaxes(cum, 1, 2) // [B, H, L]
  const lower = new Uint8Array(l * l)
  for (let t = 0; t < l; t++) for (let s = 0; s <= t; s++) lower[t * l + s] = 1
  const mask = fromStorage(lower, [l, l], 'bool')
  const diff = sub(expandDims(cumT, 3), expandDims(cumT, 2)) // [B, H, T, S] = cum_t - cum_s
  const decay = where(mask, exp(where(mask, diff, 0)), 0)
  const scores = einsum('btn,bsn->bts', c, b) // C_t . B_s
  const xdt = mul(x, expandDims(dt, 3)) // [B, L, H, P]
  const mixed = einsum('bhts,bshp->bthp', mul(expandDims(scores, 1), decay), xdt)
  // The initial state, decayed to each step, and the state after the last step.
  const fromInitial = mul(einsum('bhpn,btn->bthp', h0, c), expandDims(exp(cum), 3))
  const last = sliceAxis(cum, 1, l - 1, l) // [B, 1, H]
  const weight = exp(sub(last, cum)) // [B, L, H]: decay from step s to the end
  const written = einsum('bsh,bshp,bsn->bhpn', weight, xdt, b)
  const carried = mul(reshape(exp(last), [bs, heads, 1, 1]), h0)
  return { y: add(mixed, fromInitial), state: add(carried, written) }
}

/**
 * The delta rule (DeltaNet; Schlag et al. 2021, Yang et al. 2024): a fast-weight memory S updated by
 * S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T and read by o_t = S_t q_t; linear attention that overwrites
 * rather than only adds.
 *
 * @param q - Queries `[..., L, dk]`.
 * @param k - Keys `[..., L, dk]` (usually L2-normalised).
 * @param v - Values `[..., L, dv]`.
 * @param beta - Write strengths `[..., L]` in (0, 1).
 * @param s0 - The initial memory `[..., dv, dk]` (default zeros).
 * @returns The outputs `[..., L, dv]` and the final memory.
 * @reference Yang, Wang, Zhang, Shen and Kim (2024), Parallelizing linear transformers with the delta rule over
 *   sequence length.
 * @example
 * ```ts
 * import { deltaRule } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(deltaRule([[1, 0]], [[1, 0]], [[3]], [1]).output) // => [[3]]
 * ```
 */
export function deltaRule(
  q: TensorLike,
  k: TensorLike,
  v: TensorLike,
  beta: TensorLike,
  s0?: TensorLike,
): { readonly output: Value; readonly state: Value } {
  const [qv, kv, vv, bv] = promoteFloat(q, k, v, beta) as [Value, Value, Value, Value]
  const rank = qv.shape.length
  const l = qv.shape[rank - 2] ?? 0
  const dk = qv.shape[rank - 1] ?? 0
  const dv = vv.shape[rank - 1] ?? 0
  const lead = qv.shape.slice(0, -2)
  let s: Value = s0 === undefined ? zeros([...lead, dv, dk]) : fl(s0)
  const row = (x: Value, t: number): Value => sliceAxis(x, rank - 2, t, t + 1) // [..., 1, d]
  const outs: Value[] = []
  for (let t = 0; t < l; t++) {
    const kt = row(kv, t)
    const vt = row(vv, t)
    const qt = row(qv, t)
    const bt = expandDims(sliceAxis(bv, rank - 2, t, t + 1), -1) // [..., 1, 1]
    // S k: [..., dv, 1]; the update (v - S k) k^T scaled by beta.
    const sk = matmul(s, swapaxes(kt, -1, -2))
    const err = sub(swapaxes(vt, -1, -2), sk)
    s = add(s, mul(bt, matmul(err, kt)))
    outs.push(swapaxes(matmul(s, swapaxes(qt, -1, -2)), -1, -2))
  }
  return { output: concat(outs, rank - 2), state: s }
}

/** The parameters of a linear recurrent unit (complex numbers as real and imaginary parts). */
export interface LRUParams {
  /** log(-log |lambda|) `[N]`. */
  readonly nuLog: Value
  /** log(arg lambda) `[N]`. */
  readonly thetaLog: Value
  /** log of the input normalisation gamma `[N]`. */
  readonly gammaLog: Value
  /** B `[N, H]`, real and imaginary. */
  readonly bRe: Value
  readonly bIm: Value
  /** C `[H, N]`, real and imaginary. */
  readonly cRe: Value
  readonly cIm: Value
  /** The skip `[H]`. */
  readonly d: Value
}

/**
 * Initialise a linear recurrent unit as the paper does: eigenvalues uniform on a ring r_min <= |lambda| <= r_max with
 * phases up to maxPhase, gamma = sqrt(1 - |lambda|^2), B and C Glorot-like complex normals (here uniform with the same
 * variance), D standard.
 *
 * @param key - The random key.
 * @param stateSize - N.
 * @param width - H (input and output width).
 * @param options - rMin (0), rMax (1), maxPhase (2 pi).
 * @returns The parameters.
 * @example
 * ```ts
 * import { lruInit } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * lruInit(key(0), 4, 2).bRe.shape // => [4, 2]
 * ```
 */
export function lruInit(
  key: Key,
  stateSize: number,
  width: number,
  options: { readonly rMin?: number; readonly rMax?: number; readonly maxPhase?: number } = {},
): LRUParams {
  const rMin = options.rMin ?? 0
  const rMax = options.rMax ?? 1
  const maxPhase = options.maxPhase ?? 2 * Math.PI
  const u1 = uniform(foldIn(key, 0), [stateSize]).data
  const u2 = uniform(foldIn(key, 1), [stateSize]).data
  const nu = Float64Array.from(u1, (u) => Math.log(-0.5 * Math.log(u * (rMax * rMax - rMin * rMin) + rMin * rMin)))
  const theta = Float64Array.from(u2, (u) => Math.log(Math.max(maxPhase * u, 1e-12)))
  const gamma = Float64Array.from(nu, (v) => Math.log(Math.sqrt(1 - Math.exp(-2 * Math.exp(v)))))
  // Complex normals with variance 1 / (2 fan-in) per part: uniform on +-sqrt(3 / (2 fan)).
  const u = (i: number, shape: number[], fan: number): Value => {
    const r = Math.sqrt(3 / (2 * fan))
    return uniform(foldIn(key, i), shape, { low: -r, high: r })
  }
  return {
    nuLog: fromStorage(nu, [stateSize], 'float64'),
    thetaLog: fromStorage(theta, [stateSize], 'float64'),
    gammaLog: fromStorage(gamma, [stateSize], 'float64'),
    bRe: u(2, [stateSize, width], width),
    bIm: u(3, [stateSize, width], width),
    cRe: u(4, [width, stateSize], stateSize),
    cIm: u(5, [width, stateSize], stateSize),
    d: uniform(foldIn(key, 6), [width], { low: -1, high: 1 }),
  }
}

/**
 * The eigenvalues of a linear recurrent unit, lambda = exp(-exp(nuLog) + i exp(thetaLog)), as modulus-phase parts:
 * |lambda| = exp(-exp(nuLog)) < 1 always, so the recurrence is stable by construction.
 *
 * @param params - The parameters.
 * @returns Real and imaginary parts `[N]`.
 * @example
 * ```ts
 * import { lruEigenvalues } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(lruEigenvalues({ nuLog: [0], thetaLog: [-100] }).re) // => [0.36787944117144233]
 * ```
 */
export function lruEigenvalues(params: { readonly nuLog: TensorLike; readonly thetaLog: TensorLike }): {
  readonly re: Value
  readonly im: Value
} {
  const mod = exp(mul(exp(fl(params.nuLog)), -1))
  const phase = exp(fl(params.thetaLog))
  return { re: mul(mod, cos(phase)), im: mul(mod, sin(phase)) }
}

/** The state of a linear recurrent unit: real and imaginary parts `[B, N]`. */
export interface LRUState {
  readonly re: Value
  readonly im: Value
}

/**
 * A linear recurrent unit (Orvieto et al. 2023): h_t = lambda h_{t-1} + gamma (B u_t) with complex diagonal lambda,
 * y_t = Re(C h_t) + D u_t, in real arithmetic on real and imaginary parts. Differentiable in the parameters and u.
 *
 * @param params - The parameters (see {@link lruInit}).
 * @param u - Inputs `[L, H]` (or `[B, L, H]`).
 * @returns Outputs with the shape of u.
 * @reference Orvieto et al. (2023), Resurrecting recurrent neural networks for long sequences.
 * @example
 * ```ts
 * import { lru, lruInit } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * import { zeros } from 'omai-engine/tensor'
 * lru(lruInit(key(0), 4, 2), zeros([5, 2])).shape // => [5, 2]
 * ```
 */
export function lru(params: LRUParams, u: TensorLike): Value {
  const u0 = fl(u)
  const batched = u0.shape.length === 3
  const y = lruScan(params, batched ? u0 : expandDims(u0, 0)).y
  return batched ? y : reshape(y, [y.shape[1] ?? 0, -1])
}

/**
 * {@link lru} from a given state, returning the final state too: the form that resumes a sequence (generation one
 * token at a time gives the same outputs as one pass over the whole sequence).
 *
 * @param params - The parameters.
 * @param u - Inputs `[B, L, H]`.
 * @param state - The state before the first step (default zeros).
 * @returns The outputs `[B, L, H]` and the state after the last step.
 * @example
 * ```ts
 * import { lruInit, lruScan } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * import { zeros } from 'omai-engine/tensor'
 * lruScan(lruInit(key(0), 4, 2), zeros([1, 5, 2])).state.re.shape // => [1, 4]
 * ```
 */
export function lruScan(
  params: LRUParams,
  u: TensorLike,
  state?: LRUState,
): { readonly y: Value; readonly state: LRUState } {
  const uv = fl(u)
  const [bs = 1, l = 0] = uv.shape
  const n = params.nuLog.shape[0] ?? 0
  const { re, im } = lruEigenvalues(params)
  const gamma = exp(params.gammaLog)
  // Bu for every step: [B, L, N].
  const bu = (w: Value): Value => mul(matmul(uv, swapaxes(w, 0, 1)), gamma)
  const xr = bu(params.bRe)
  const xi = bu(params.bIm)
  let hr: Value = state === undefined ? zeros([bs, n]) : state.re
  let hi: Value = state === undefined ? zeros([bs, n]) : state.im
  const ys: Value[] = []
  const at = (v: Value, t: number): Value => reshape(sliceAxis(v, 1, t, t + 1), [bs, n])
  for (let t = 0; t < l; t++) {
    const nr = add(sub(mul(re, hr), mul(im, hi)), at(xr, t))
    const ni = add(add(mul(re, hi), mul(im, hr)), at(xi, t))
    hr = nr
    hi = ni
    // Re(C h) = C_re h_re - C_im h_im.
    ys.push(sub(matmul(hr, swapaxes(params.cRe, 0, 1)), matmul(hi, swapaxes(params.cIm, 0, 1))))
  }
  return { y: add(stack(ys, 1), mul(uv, params.d)), state: { re: hr, im: hi } }
}
