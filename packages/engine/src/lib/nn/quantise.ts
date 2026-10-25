/**
 * Quantisation, which stores weights as low-bit integers with a scale and zero point: quantisation parameters per
 * tensor or per channel, quantise, dequantise and fake quantisation with a straight-through gradient for
 * quantisation-aware training, error measures, the GPTQ and AWQ post-training algorithms for linear layers, and
 * `quantiseModule` for round-to-nearest quantisation of a whole parameter tree.
 */

import { customVjp } from '../autodiff/index.ts'
import { maxOf, minOf, WhiteboxError } from '../base/index.ts'
import { cholesky, inv } from '../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  cast,
  clip,
  div,
  fromStorage,
  greaterEqual,
  lessEqual,
  logicalAnd,
  matmul,
  mul,
  primalOf,
  reshape,
  round,
  sub,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  treeFlatten,
  treePaths,
  treeUnflatten,
  type Value,
} from '../tensor/index.ts'
import type { Params } from './module/module.ts'

/** How a tensor maps to b-bit integers: a scale and zero point per tensor or per channel along an axis. */
export interface QuantParams {
  readonly bits: number
  /** Symmetric (zero point 0, signed range) or affine (unsigned range with a zero point). */
  readonly symmetric: boolean
  readonly qmin: number
  readonly qmax: number
  /** One scale per tensor, or per index of `axis`. */
  readonly scale: readonly number[]
  readonly zeroPoint: readonly number[]
  /** The channel axis of per-channel parameters (undefined: per tensor). */
  readonly axis?: number
}

/** Options of {@link quantParams}. */
export interface QuantOptions {
  /** Bit width (default 8). */
  readonly bits?: number
  /** Symmetric (default) or affine. */
  readonly symmetric?: boolean
  /** Per-channel parameters along this axis (default: per tensor). */
  readonly axis?: number
}

const normAxis = (axis: number, rank: number): number => (axis < 0 ? axis + rank : axis)

/** The values of each channel along an axis (or one group of everything). */
function channels(x: Tensor, axis: number | undefined): number[][] {
  const data = toFlat(x)
  if (axis === undefined) return [data]
  const a = normAxis(axis, x.shape.length)
  const n = x.shape[a] ?? 1
  const inner = x.shape.slice(a + 1).reduce((p, c) => p * c, 1)
  const groups: number[][] = Array.from({ length: n }, () => [])
  for (const [i, v] of data.entries()) groups[Math.floor(i / inner) % n]?.push(v)
  return groups
}

/**
 * Quantisation parameters from a tensor's range (PyTorch's MinMax observer): symmetric scale max|x| / qmax with the
 * signed range [-2^(b-1), 2^(b-1) - 1], or affine scale (max - min) / (2^b - 1) with the range widened to include 0 and
 * a zero point, per tensor or per channel.
 *
 * @param x - The tensor (traced values use their primal).
 * @param options - Bits (8), symmetric (true), per-channel axis.
 * @returns The parameters.
 * @concept quantisation
 * @example
 * ```ts
 * import { quantParams } from 'whitebox-engine/nn'
 * quantParams([-1, 0.5, 2], { bits: 8, symmetric: false }).zeroPoint // => [85]
 * ```
 */
export function quantParams(x: TensorLike, options: QuantOptions = {}): QuantParams {
  const t = primalOf(asTensor(x))
  const bits = options.bits ?? 8
  if (!(bits >= 2 && bits <= 16)) throw new WhiteboxError('value', `quantParams: bits must be in [2, 16], got ${bits}`)
  const symmetric = options.symmetric ?? true
  const qmin = symmetric ? -(2 ** (bits - 1)) : 0
  const qmax = symmetric ? 2 ** (bits - 1) - 1 : 2 ** bits - 1
  const scale: number[] = []
  const zeroPoint: number[] = []
  for (const g of channels(t, options.axis)) {
    if (symmetric) {
      const m = maxOf(g.map(Math.abs))
      scale.push(m > 0 ? m / qmax : 1)
      zeroPoint.push(0)
    } else {
      const lo = Math.min(minOf(g), 0)
      const hi = Math.max(maxOf(g), 0)
      const s = hi > lo ? (hi - lo) / (qmax - qmin) : 1
      scale.push(s)
      zeroPoint.push(Math.min(Math.max(qmin - Math.round(lo / s), qmin), qmax))
    }
  }
  return {
    bits,
    symmetric,
    qmin,
    qmax,
    scale,
    zeroPoint,
    ...(options.axis === undefined ? {} : { axis: options.axis }),
  }
}

/** The scale and zero point as tensors broadcastable against x. */
function broadcastParams(x: Value, qp: QuantParams): { scale: Value; zero: Value } {
  if (qp.axis === undefined) return { scale: asTensor(qp.scale[0] ?? 1), zero: asTensor(qp.zeroPoint[0] ?? 0) }
  const a = normAxis(qp.axis, x.shape.length)
  const shape = x.shape.map((n, i) => (i === a ? n : 1))
  return {
    scale: fromStorage(Float64Array.from(qp.scale), shape, 'float64'),
    zero: fromStorage(Float64Array.from(qp.zeroPoint), shape, 'float64'),
  }
}

/**
 * Quantise to integers: clamp(round(x / scale) + zeroPoint, qmin, qmax), rounding half to even.
 *
 * @param x - The tensor.
 * @param qp - The parameters.
 * @returns int32 codes.
 * @example
 * ```ts
 * import { quantise, quantParams } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(quantise([-1, 0.5, 1], quantParams([-1, 0.5, 1], { bits: 4 }))) // => [-7, 4, 7]
 * ```
 */
export function quantise(x: TensorLike, qp: QuantParams): Tensor<'int32'> {
  const v = primalOf(asTensor(x))
  const { scale, zero } = broadcastParams(v, qp)
  return primalOf(cast(clip(add(round(div(v, scale)), zero), qp.qmin, qp.qmax), 'int32')) as Tensor<'int32'>
}

/**
 * Back to real values: (q - zeroPoint) scale.
 *
 * @param q - Integer codes.
 * @param qp - The parameters.
 * @returns The dequantised tensor.
 * @example
 * ```ts
 * import { dequantise, quantParams } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(dequantise([-7, 7], quantParams([-1, 1], { bits: 4 }))) // => [-1, 1]
 * ```
 */
export function dequantise(q: TensorLike, qp: QuantParams): Tensor {
  const v = asTensor(q, 'float64')
  const { scale, zero } = broadcastParams(v, qp)
  return primalOf(mul(sub(v, zero), scale))
}

/**
 * Fake quantisation for quantisation-aware training: the forward pass is dequantise(quantise(x)); the backward pass
 * is the straight-through estimator, the identity inside the representable range [(qmin - z) s, (qmax - z) s] and zero
 * outside (PyTorch's fake_quantize_per_tensor_affine).
 *
 * @param x - The value (traced values are fine).
 * @param qp - The parameters (default: from x's own range, 8-bit symmetric).
 * @returns The fake-quantised value.
 * @concept quantisation-aware-training
 * @example
 * ```ts
 * import { fakeQuantise, quantParams } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fakeQuantise([0.3, 1], quantParams([0.3, 1], { bits: 2 }))) // => [0, 1]
 * ```
 */
export function fakeQuantise(x: TensorLike, qp?: QuantParams): Value {
  const v = asTensor(x)
  const p = qp ?? quantParams(v)
  const inRange = (u: Value): Value => {
    const { scale, zero } = broadcastParams(u, p)
    const lo = mul(sub(p.qmin, zero), scale)
    const hi = mul(sub(p.qmax, zero), scale)
    return cast(logicalAnd(greaterEqual(u, lo), lessEqual(u, hi)), 'float64')
  }
  const f = customVjp({
    name: 'fakeQuantise',
    forward: (u) => dequantise(quantise(u, p), p),
    vjp: (g, [u]) => [mul(g, inRange(u as Value))],
    jvp: ([u], [t]) => (t === null || t === undefined ? null : mul(t, inRange(u as Value))),
  })
  return f(v)
}

/** Quantisation error of a tensor against its quantised copy. */
export interface QuantError {
  readonly mse: number
  readonly maxAbs: number
  /** Signal-to-quantisation-noise ratio, 10 log10(sum x^2 / sum (x - xq)^2), in dB. */
  readonly sqnr: number
}

/**
 * How far a quantised copy is from the original: mean squared error, largest absolute error and SQNR in dB.
 *
 * @param x - The original.
 * @param xq - The (dequantised) copy.
 * @returns The errors.
 * @example
 * ```ts
 * import { quantisationError } from 'whitebox-engine/nn'
 * quantisationError([1, 2], [1, 2]).sqnr // => Infinity
 * ```
 */
export function quantisationError(x: TensorLike, xq: TensorLike): QuantError {
  const a = toFlat(primalOf(asTensor(x)))
  const b = toFlat(primalOf(asTensor(xq)))
  let noise = 0
  let signal = 0
  let maxAbs = 0
  for (const [i, v] of a.entries()) {
    const e = v - (b[i] ?? 0)
    noise += e * e
    signal += v * v
    maxAbs = Math.max(maxAbs, Math.abs(e))
  }
  return {
    mse: noise / Math.max(1, a.length),
    maxAbs,
    sqnr: noise === 0 ? Number.POSITIVE_INFINITY : 10 * Math.log10(signal / noise),
  }
}

/**
 * The output error of a quantised linear layer on calibration inputs: mean ||x W^T - x Wq^T||^2 per sample.
 *
 * @param weight - `[out, in]`.
 * @param quantised - `[out, in]`, dequantised.
 * @param inputs - Calibration inputs `[n, in]`.
 * @returns The mean squared output error per sample.
 * @example
 * ```ts
 * import { outputError } from 'whitebox-engine/nn'
 * outputError([[1, 2]], [[1, 1]], [[1, 1]]) // => 1
 * ```
 */
export function outputError(weight: TensorLike, quantised: TensorLike, inputs: TensorLike): number {
  const x = asTensor(inputs)
  const d = toFlat(matmul(x, swapaxes(sub(asTensor(weight), asTensor(quantised)), 0, 1)))
  const n = x.shape[0] ?? 1
  return d.reduce((s, v) => s + v * v, 0) / n
}

/** Options of {@link gptq}. */
export interface GptqOptions {
  readonly bits?: number
  readonly symmetric?: boolean
  /** Dampening as a fraction of the mean diagonal of H (default 0.01). */
  readonly percDamp?: number
  /** Quantise columns in order of decreasing Hessian diagonal (GPTQ's act-order; default false). */
  readonly actOrder?: boolean
}

/** Result of {@link gptq} and {@link awq}. */
export interface QuantisedWeight {
  /** The dequantised weight `[out, in]`. */
  readonly weight: Tensor
  /** The integer codes `[out, in]`. */
  readonly codes: Tensor<'int32'>
  /** Per-output-row parameters. */
  readonly params: QuantParams
}

/**
 * GPTQ (Frantar et al. 2022): quantise a linear layer's weight column by column on a per-row grid, spreading each
 * column's rounding error over the remaining columns through the Cholesky factor of the inverse Hessian
 * H = 2 X^T X / n + damping of the calibration inputs, so the layer's output on those inputs changes least.
 *
 * @param weight - `[out, in]`.
 * @param inputs - Calibration inputs `[n, in]`.
 * @param options - Bits (4), symmetric (false), dampening and act-order.
 * @returns The quantised weight, codes and parameters.
 * @reference Frantar, Ashkboos, Hoefler and Alistarh (2022), GPTQ: accurate post-training quantization for generative
 *   pre-trained transformers (Algorithm 1).
 * @example
 * ```ts
 * import { gptq } from 'whitebox-engine/nn'
 * gptq([[0.5, -1], [2, 0.1]], [[1, 0], [0, 1], [1, 1]], { bits: 4 }).weight.shape // => [2, 2]
 * ```
 */
export function gptq(weight: TensorLike, inputs: TensorLike, options: GptqOptions = {}): QuantisedWeight {
  const w0 = primalOf(asTensor(weight))
  const x = primalOf(asTensor(inputs))
  const [rows = 0, cols = 0] = w0.shape
  if (x.shape[1] !== cols) throw new WhiteboxError('shape', 'gptq: inputs must be [n, in] with in the weight width')
  const n = x.shape[0] ?? 1
  const h = toFlat(mul(matmul(swapaxes(x, 0, 1), x), 2 / n))
  const w = toFlat(w0)
  // Dead inputs: no information about their columns, so zero them (as the reference implementation).
  for (let j = 0; j < cols; j++) {
    if ((h[j * cols + j] ?? 0) === 0) {
      h[j * cols + j] = 1
      for (let r = 0; r < rows; r++) w[r * cols + j] = 0
    }
  }
  let mean = 0
  for (let j = 0; j < cols; j++) mean += (h[j * cols + j] ?? 0) / cols
  const damp = (options.percDamp ?? 0.01) * mean
  for (let j = 0; j < cols; j++) h[j * cols + j] = (h[j * cols + j] ?? 0) + damp
  const order = Array.from({ length: cols }, (_, j) => j)
  if (options.actOrder === true) order.sort((a, b) => (h[b * cols + b] ?? 0) - (h[a * cols + a] ?? 0))
  // Permute H and W to the quantisation order.
  const hp = new Float64Array(cols * cols)
  for (const [i, a] of order.entries()) for (const [j, b] of order.entries()) hp[i * cols + j] = h[a * cols + b] ?? 0
  const wp = new Float64Array(rows * cols)
  for (let r = 0; r < rows; r++) for (const [j, b] of order.entries()) wp[r * cols + j] = w[r * cols + b] ?? 0
  // U upper triangular with H^-1 = U^T U.
  const hinv = inv(fromStorage(hp, [cols, cols], 'float64'))
  const u = toFlat(swapaxes(cholesky(hinv), 0, 1))
  const qp = quantParams(fromStorage(Float64Array.from(wp), [rows, cols], 'float64'), {
    bits: options.bits ?? 4,
    symmetric: options.symmetric ?? false,
    axis: 0,
  })
  const qcols = new Float64Array(rows * cols)
  const codes = new Int32Array(rows * cols)
  for (let j = 0; j < cols; j++) {
    const col = fromStorage(
      Float64Array.from({ length: rows }, (_, r) => wp[r * cols + j] ?? 0),
      [rows],
      'float64',
    )
    const colParams: QuantParams = { ...qp, axis: 0 }
    const q = quantise(col, colParams)
    const dq = toFlat(dequantise(q, colParams))
    const d = u[j * cols + j] ?? 1
    const qv = toFlat(q)
    for (let r = 0; r < rows; r++) {
      const err = ((wp[r * cols + j] ?? 0) - (dq[r] ?? 0)) / d
      qcols[r * cols + j] = dq[r] ?? 0
      codes[r * cols + j] = qv[r] ?? 0
      for (let k = j + 1; k < cols; k++) wp[r * cols + k] = (wp[r * cols + k] ?? 0) - err * (u[j * cols + k] ?? 0)
    }
  }
  // Undo the permutation.
  const out = new Float64Array(rows * cols)
  const outCodes = new Int32Array(rows * cols)
  for (let r = 0; r < rows; r++) {
    for (const [j, b] of order.entries()) {
      out[r * cols + b] = qcols[r * cols + j] ?? 0
      outCodes[r * cols + b] = codes[r * cols + j] ?? 0
    }
  }
  return {
    weight: fromStorage(out, [rows, cols], 'float64'),
    codes: fromStorage(outCodes, [rows, cols], 'int32'),
    params: qp,
  }
}

/** Options of {@link awq}. */
export interface AwqOptions {
  readonly bits?: number
  readonly symmetric?: boolean
  /** Grid points for alpha in [0, 1] (default 20). */
  readonly gridSize?: number
}

/** Result of {@link awq}. */
export interface AwqResult extends QuantisedWeight {
  /** The chosen exponent. */
  readonly alpha: number
  /** The per-input-channel scales s. */
  readonly scales: readonly number[]
  /** The calibration output error at the chosen alpha. */
  readonly error: number
}

/** Round-to-nearest per-row quantisation of W diag(s), divided back by s. */
function scaledRtn(w: Tensor, s: readonly number[], bits: number, symmetric: boolean): QuantisedWeight {
  const sv = fromStorage(Float64Array.from(s), [1, s.length], 'float64')
  const scaled = primalOf(mul(w, sv))
  const qp = quantParams(scaled, { bits, symmetric, axis: 0 })
  const codes = quantise(scaled, qp)
  return { weight: primalOf(div(dequantise(codes, qp), sv)), codes, params: qp }
}

/**
 * AWQ (Lin et al. 2023): activation-aware weight quantisation. Input channels with large activations matter more, so
 * the weight's columns are scaled by s_j = mean|x_j|^alpha before round-to-nearest quantisation and divided back
 * after; alpha is chosen on a grid in [0, 1] to minimise the layer's output error on calibration inputs (alpha = 0 is
 * plain round-to-nearest, so AWQ is never worse on the calibration data).
 *
 * @param weight - `[out, in]`.
 * @param inputs - Calibration inputs `[n, in]`.
 * @param options - Bits (4), symmetric (false), grid size.
 * @returns The quantised weight, codes, parameters, alpha, scales and error.
 * @reference Lin, Tang, Tang, Yang, Dang and Han (2023), AWQ: activation-aware weight quantization for LLM compression
 *   and acceleration.
 * @example
 * ```ts
 * import { awq } from 'whitebox-engine/nn'
 * awq([[0.5, -1], [2, 0.1]], [[1, 0], [0, 3]], { bits: 3 }).scales.length // => 2
 * ```
 */
export function awq(weight: TensorLike, inputs: TensorLike, options: AwqOptions = {}): AwqResult {
  const w = primalOf(asTensor(weight))
  const x = primalOf(asTensor(inputs))
  const cols = w.shape[1] ?? 0
  const bits = options.bits ?? 4
  const symmetric = options.symmetric ?? false
  const grid = options.gridSize ?? 20
  const xv = toFlat(x)
  const n = x.shape[0] ?? 1
  const meanAbs = Array.from({ length: cols }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += Math.abs(xv[i * cols + j] ?? 0)
    return s / n
  })
  let best: AwqResult | undefined
  for (let g = 0; g <= grid; g++) {
    const alpha = g / grid
    const raw = meanAbs.map((m) => Math.max(m, 1e-8) ** alpha)
    // Normalise as the reference implementation: s / sqrt(max s min s).
    const norm = Math.sqrt(maxOf(raw) * minOf(raw))
    const scales = raw.map((v) => v / norm)
    const q = scaledRtn(w, scales, bits, symmetric)
    const error = outputError(w, q.weight, x)
    if (best === undefined || error < best.error) best = { ...q, alpha, scales, error }
  }
  return best as AwqResult
}

/** Options of {@link quantiseModule}. */
export interface QuantiseModuleOptions {
  readonly bits?: number
  readonly symmetric?: boolean
  /** Per-output-channel scales (default true). */
  readonly perChannel?: boolean
  /** Which parameters to quantise (default: leaves named `weight` with at least two axes). */
  readonly filter?: (path: string, value: Value) => boolean
}

/** Result of {@link quantiseModule}. */
export interface QuantisedParams {
  /** The parameter tree with quantised (dequantised float) weights. */
  readonly params: Params
  /** The error of each quantised tensor, by path. */
  readonly report: Readonly<Record<string, QuantError>>
}

/**
 * Post-training round-to-nearest quantisation of a parameter tree: every Linear and convolution weight (by default) is
 * replaced by its quantised copy (per output channel by default); the rest is unchanged. Load the result with
 * `module.setParams` to evaluate the quantised model.
 *
 * @param params - The parameter tree (a module's `params`).
 * @param options - Bits (8), symmetric (true), per-channel (true), a filter.
 * @returns The quantised tree and an error report.
 * @concept quantisation
 * @example
 * ```ts
 * import { quantiseModule, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * Object.keys(quantiseModule(new Linear(4, 2, { key: key(0) }).params, { bits: 4 }).report) // => ['weight']
 * ```
 */
export function quantiseModule(params: Params, options: QuantiseModuleOptions = {}): QuantisedParams {
  const filter =
    options.filter ??
    ((path: string, v: Value) => (path === 'weight' || path.endsWith('.weight')) && v.shape.length >= 2)
  const paths = treePaths(params)
  const { def } = treeFlatten(params)
  const report: Record<string, QuantError> = {}
  const leaves = Object.entries(paths).map(([path, v]) => {
    if (!filter(path, v)) return v
    const qp = quantParams(v, {
      bits: options.bits ?? 8,
      symmetric: options.symmetric ?? true,
      ...((options.perChannel ?? true) ? { axis: 0 } : {}),
    })
    const q = dequantise(quantise(v, qp), qp)
    report[path] = quantisationError(v, q)
    return reshape(q, v.shape)
  })
  return { params: treeUnflatten(def, leaves) as Params, report }
}
