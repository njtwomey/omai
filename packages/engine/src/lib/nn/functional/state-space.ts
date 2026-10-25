/**
 * State-space sequence models, the linear recurrences behind S4, Mamba and their relatives: discretising a continuous
 * model (zero-order hold or bilinear), the convolution kernel and recurrent views of the same linear system,
 * elementwise linear recurrences, Mamba's input-dependent selective scan, the delta rule of DeltaNet and the linear
 * recurrent unit (LRU) with its initialisation.
 */

import { WhiteboxError } from '../../base/index.ts'
import { expm, solve } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import {
  add,
  concat,
  cos,
  div,
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
 * import { discretise } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
    throw new WhiteboxError('shape', 'discretise: A must be [N, N] and B [N, M]')
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
 * import { discretiseDiagonal } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { ssmKernel } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { causalConvolution } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { ssmRecurrence } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { linearRecurrence } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { selectiveScan } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
  let h: Value = zeros([bs, dim, n])
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
 * import { deltaRule } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { lruInit } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
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
 * import { lruEigenvalues } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { lru, lruInit } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import { zeros } from 'whitebox-engine/tensor'
 * lru(lruInit(key(0), 4, 2), zeros([5, 2])).shape // => [5, 2]
 * ```
 */
export function lru(params: LRUParams, u: TensorLike): Value {
  const u0 = fl(u)
  const batched = u0.shape.length === 3
  const uv = batched ? u0 : expandDims(u0, 0)
  const [bs = 1, l = 0] = uv.shape
  const n = params.nuLog.shape[0] ?? 0
  const { re, im } = lruEigenvalues(params)
  const gamma = exp(params.gammaLog)
  // Bu for every step: [B, L, N].
  const bu = (w: Value): Value => mul(matmul(uv, swapaxes(w, 0, 1)), gamma)
  const xr = bu(params.bRe)
  const xi = bu(params.bIm)
  let hr: Value = zeros([bs, n])
  let hi: Value = zeros([bs, n])
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
  const y = add(stack(ys, 1), mul(uv, params.d))
  return batched ? y : reshape(y, [l, -1])
}
