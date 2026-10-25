/**
 * Gradient attributions for differentiable models (captum's family): saliency, input times gradient, integrated
 * gradients with their quadrature rules, SmoothGrad and noise tunnels (VarGrad, SmoothGrad squared) over any method,
 * expected gradients and GradientSHAP, DeepLIFT's rescale rule, guided backpropagation and deconvolution, Grad-CAM with
 * map interpolation, and layer-wise relevance propagation (epsilon, gamma and alpha-beta rules) for dense networks.
 */
import { vjp } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { keyFor } from '../learn/_shared.ts'
import { matrixView } from '../learn/validation.ts'
import { gaussLegendre } from '../numerics/quadrature/index.ts'
import { foldIn, type Key, normal, randint, uniform } from '../random/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  isTensor,
  mul,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'

/** A differentiable model: inputs `[n, d]` to outputs `[n, k]` (or `[n]`), rows independent. */
export type DifferentiableModel = (x: Value) => Value

/** Which output to explain: one column for every row, or one per row. */
export type Target = number | readonly number[]

function concrete(x: TensorLike, who: string): Tensor {
  const t = asTensor(x)
  if (!isTensor(t)) throw new WhiteboxError('value', `${who}: expected concrete inputs`)
  return t.dtype === 'float64' ? t : fromStorage(Float64Array.from(toFlat(t)), t.shape, 'float64')
}

/**
 * The gradient of the target output of every row with respect to its inputs, `[n, d]`, by one reverse pass.
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param target - The output column (or one per row; default 0).
 * @returns The gradients.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * toArray(explain.targetGradient((x) => matmul(x, [[1, 0], [0, 5]]), [[1, 1]], 1)) // => [[0, 5]]
 * ```
 */
export function targetGradient(model: DifferentiableModel, x: TensorLike, target: Target = 0): Tensor {
  const xt = concrete(x, 'targetGradient')
  const { out, pullback } = vjp(model, xt as Value)
  const o = out as Tensor
  const n = o.shape[0] ?? 0
  const k = o.shape.length === 2 ? (o.shape[1] ?? 1) : 1
  const cot = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    const t = typeof target === 'number' ? target : (target[i] ?? 0)
    if (!(Number.isInteger(t) && t >= 0 && t < k))
      throw new WhiteboxError('value', `targetGradient: target ${t} outside [0, ${k})`)
    cot[i * k + t] = 1
  }
  return pullback(fromStorage(cot, o.shape, 'float64')) as Tensor
}

/**
 * Saliency (Simonyan, Vedaldi and Zisserman 2014; captum Saliency): the gradient of the target output with respect to
 * the input, by default in absolute value.
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param options - target (default 0) and abs (default true).
 * @returns Attributions `[n, d]`.
 * @concept attribution
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * const linear = (x) => matmul(x, [[2], [-3]])
 * toArray(explain.saliency(linear, [[1, 1]])) // => [[2, 3]]
 * ```
 */
export function saliency(
  model: DifferentiableModel,
  x: TensorLike,
  options: { readonly target?: Target; readonly abs?: boolean } = {},
): Tensor {
  const g = targetGradient(model, x, options.target ?? 0)
  return options.abs === false
    ? g
    : fromStorage(Float64Array.from(g.data as Float64Array, Math.abs), g.shape, 'float64')
}

/**
 * Input times gradient (Shrikumar et al. 2016; captum InputXGradient): the elementwise product of the input and the
 * gradient of the target output, exact for linear models.
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param options - target (default 0).
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * toArray(explain.inputXGradient((x) => matmul(x, [[2], [-3]]), [[1, 2]])) // => [[2, -6]]
 * ```
 */
export function inputXGradient(
  model: DifferentiableModel,
  x: TensorLike,
  options: { readonly target?: Target } = {},
): Tensor {
  const xt = concrete(x, 'inputXGradient')
  return mul(xt, targetGradient(model, xt, options.target ?? 0)) as Tensor
}

/** The quadrature rule of {@link integratedGradients} (captum's names). */
export type PathMethod = 'gausslegendre' | 'riemann_left' | 'riemann_right' | 'riemann_middle' | 'riemann_trapezoid'

/**
 * Nodes in [0, 1] and weights of a path-integral rule with n points (captum's approximation_methods).
 *
 * @param method - The rule.
 * @param n - The number of points.
 * @returns The path positions and their weights (summing to 1).
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.pathRule('riemann_trapezoid', 3) // => { alphas: [0, 0.5, 1], weights: [1 / 6, 1 / 3, 1 / 6] }
 * ```
 */
export function pathRule(method: PathMethod, n: number): { readonly alphas: number[]; readonly weights: number[] } {
  if (!(Number.isInteger(n) && n >= (method === 'gausslegendre' ? 1 : 2)))
    throw new WhiteboxError('value', `pathRule: bad number of steps ${n}`)
  if (method === 'gausslegendre') {
    const r = gaussLegendre(n)
    return { alphas: toFlat(r.nodes).map((v) => 0.5 * (1 + v)), weights: toFlat(r.weights).map((w) => 0.5 * w) }
  }
  const weights = new Array<number>(n).fill(1 / n)
  if (method === 'riemann_trapezoid') {
    weights[0] = 0.5 / n
    weights[n - 1] = 0.5 / n
  }
  const lin = (a: number, b: number): number[] => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
  const alphas =
    method === 'riemann_trapezoid'
      ? lin(0, 1)
      : method === 'riemann_left'
        ? lin(0, 1 - 1 / n)
        : method === 'riemann_right'
          ? lin(1 / n, 1)
          : lin(1 / (2 * n), 1 - 1 / (2 * n))
  return { alphas, weights }
}

/** Options of {@link integratedGradients}. */
export interface IntegratedGradientsOptions {
  readonly target?: Target
  /** The baseline (default zeros): a row `[d]` or one per input `[n, d]`. */
  readonly baseline?: TensorLike
  /** Path points (default 50). */
  readonly nSteps?: number
  /** The quadrature (default `'gausslegendre'`, captum's). */
  readonly method?: PathMethod
}

function baselineOf(xt: Tensor, baseline: TensorLike | undefined, who: string): Tensor {
  if (baseline === undefined) return fromStorage(new Float64Array(xt.data.length), xt.shape, 'float64')
  const b = concrete(baseline, who)
  if (b.data.length === xt.data.length)
    return fromStorage(Float64Array.from(b.data as Float64Array), xt.shape, 'float64')
  const d = xt.shape[1] ?? 1
  if (b.data.length !== d) throw new WhiteboxError('shape', `${who}: the baseline must be [d] or the shape of x`)
  const n = xt.shape[0] ?? 1
  return fromStorage(
    Float64Array.from({ length: n * d }, (_, i) => (b.data as Float64Array)[i % d] ?? 0),
    xt.shape,
    'float64',
  )
}

/**
 * Integrated gradients (Sundararajan, Taly and Yan 2017; captum IntegratedGradients): (x - x') times the average
 * gradient along the straight path from a baseline x' to x, by Gauss-Legendre or Riemann quadrature. The attributions
 * sum to f(x) - f(x') up to the quadrature error (completeness), returned as `delta`.
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param options - target, baseline, nSteps and method.
 * @returns Attributions `[n, d]` and the convergence delta per row.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { square, sum, toArray } from 'whitebox-engine/tensor'
 * const r = explain.integratedGradients((x) => sum(square(x), 1), [[1, 2]], { nSteps: 4 })
 * toArray(r.attributions).map((row) => row.map((v) => Math.round(v * 1e12) / 1e12)) // => [[1, 4]]
 * ```
 */
export function integratedGradients(
  model: DifferentiableModel,
  x: TensorLike,
  options: IntegratedGradientsOptions = {},
): { readonly attributions: Tensor; readonly delta: number[] } {
  const xt = concrete(x, 'integratedGradients')
  const base = baselineOf(xt, options.baseline, 'integratedGradients')
  const diff = sub(xt, base) as Tensor
  const { alphas, weights } = pathRule(options.method ?? 'gausslegendre', options.nSteps ?? 50)
  const total = new Float64Array(xt.data.length)
  alphas.forEach((a, s) => {
    const g = targetGradient(model, add(base, mul(diff, a)) as Tensor, options.target ?? 0)
    const gd = g.data as Float64Array
    for (let i = 0; i < total.length; i++) total[i] = (total[i] ?? 0) + (weights[s] ?? 0) * (gd[i] ?? 0)
  })
  const attributions = mul(diff, fromStorage(total, xt.shape, 'float64')) as Tensor
  const n = xt.shape[0] ?? 1
  const d = xt.shape[1] ?? 1
  const at = (v: Tensor): Float64Array => {
    const o = model(v) as Tensor
    const k = o.shape.length === 2 ? (o.shape[1] ?? 1) : 1
    const od = toFlat(o)
    return Float64Array.from(
      { length: n },
      (_, i) =>
        od[i * k + (typeof options.target === 'object' ? (options.target[i] ?? 0) : (options.target ?? 0))] ?? 0,
    )
  }
  const fx = at(xt)
  const fb = at(base)
  const ad = attributions.data as Float64Array
  const delta = Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < d; j++) s += ad[i * d + j] ?? 0
    return s - ((fx[i] ?? 0) - (fb[i] ?? 0))
  })
  return { attributions, delta }
}

/** Options of {@link smoothGrad}. */
export interface SmoothGradOptions {
  readonly target?: Target
  /** Noisy copies averaged (default 50). */
  readonly nSamples?: number
  /** The noise standard deviation, absolute (default 0.15). */
  readonly stdev?: number
  /** Average the squared gradients (SmoothGrad-squared) instead (default false). */
  readonly squared?: boolean
  readonly key?: Key
}

/**
 * SmoothGrad (Smilkov et al. 2017; captum NoiseTunnel over Saliency): the average gradient over Gaussian-perturbed
 * copies of the input, which removes the high-frequency noise of raw gradients.
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param options - target, nSamples, stdev, squared and key.
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * // A linear model's gradient does not change under noise.
 * toArray(explain.smoothGrad((x) => matmul(x, [[2], [-3]]), [[1, 1]], { nSamples: 5 })) // => [[2, -3]]
 * ```
 */
export function smoothGrad(model: DifferentiableModel, x: TensorLike, options: SmoothGradOptions = {}): Tensor {
  const xt = concrete(x, 'smoothGrad')
  const ns = options.nSamples ?? 50
  if (!(Number.isInteger(ns) && ns >= 1)) throw new WhiteboxError('value', 'smoothGrad: nSamples must be >= 1')
  const key = options.key ?? keyFor({}, 'smoothGrad')
  const sd = options.stdev ?? 0.15
  const total = new Float64Array(xt.data.length)
  for (let s = 0; s < ns; s++) {
    const noisy = add(xt, mul(normal(foldIn(key, s), xt.shape), sd)) as Tensor
    const g = targetGradient(model, noisy, options.target ?? 0).data as Float64Array
    for (let i = 0; i < total.length; i++)
      total[i] = (total[i] ?? 0) + (options.squared === true ? (g[i] ?? 0) ** 2 : (g[i] ?? 0)) / ns
  }
  return fromStorage(total, xt.shape, 'float64')
}

/** Options of {@link expectedGradients}. */
export interface ExpectedGradientsOptions {
  readonly target?: Target
  /** Samples of (baseline, path position) per input (default 100). */
  readonly nSamples?: number
  readonly key?: Key
}

/**
 * Expected gradients (Erion et al. 2021; captum GradientShap without input noise): integrated gradients averaged over
 * baselines drawn from a background distribution, estimated by sampling a background row and a uniform path position
 * for each sample. The attributions sum, in expectation, to f(x) - E f(x').
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param background - Baseline rows `[m, d]`.
 * @param options - target, nSamples and key.
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * // For a linear model: w (x - mean background) exactly, whatever the samples.
 * toArray(explain.expectedGradients((x) => matmul(x, [[2], [1]]), [[3, 3]], [[1, 1], [1, 1]], { nSamples: 8 })) // => [[4, 2]]
 * ```
 */
export function expectedGradients(
  model: DifferentiableModel,
  x: TensorLike,
  background: TensorLike,
  options: ExpectedGradientsOptions = {},
): Tensor {
  const xt = concrete(x, 'expectedGradients')
  const bg = matrixView(background, 'expectedGradients')
  const n = xt.shape[0] ?? 1
  const d = xt.shape[1] ?? 1
  if (bg.d !== d) throw new WhiteboxError('shape', 'expectedGradients: the background must have the features of x')
  const ns = options.nSamples ?? 100
  const key = options.key ?? keyFor({}, 'expectedGradients')
  const xd = xt.data as Float64Array
  const total = new Float64Array(n * d)
  for (let s = 0; s < ns; s++) {
    const rowsIdx = toFlat(randint(foldIn(key, 2 * s), [n], 0, bg.n))
    const alpha = toFlat(uniform(foldIn(key, 2 * s + 1), [n]))
    const base = Float64Array.from(
      { length: n * d },
      (_, i) => bg.data[(rowsIdx[Math.floor(i / d)] ?? 0) * d + (i % d)] ?? 0,
    )
    const point = Float64Array.from(base, (b, i) => b + (alpha[Math.floor(i / d)] ?? 0) * ((xd[i] ?? 0) - b))
    const g = targetGradient(model, fromStorage(point, xt.shape, 'float64'), options.target ?? 0).data as Float64Array
    for (let i = 0; i < total.length; i++)
      total[i] = (total[i] ?? 0) + (((xd[i] ?? 0) - (base[i] ?? 0)) * (g[i] ?? 0)) / ns
  }
  return fromStorage(total, xt.shape, 'float64')
}

/** A layer of a sequential network for {@link deepLift}: an affine map, or an elementwise nonlinearity. */
export interface DeepLiftLayer {
  readonly fn: (v: Value) => Value
  /** True for elementwise nonlinearities (ReLU, tanh, sigmoid), which use the rescale rule. */
  readonly nonlinear?: boolean
}

/**
 * DeepLIFT with the rescale rule (Shrikumar, Greenside and Kundaje 2017; captum DeepLift) for sequential networks of
 * affine layers and elementwise nonlinearities: multipliers propagate backwards, through affine layers by their
 * Jacobian and through nonlinearities as the secant slope (f(x) - f(x')) / (x - x') (the gradient where x ~ x').
 * Attributions are (x - x') times the input multipliers and sum exactly to f(x) - f(x').
 *
 * @param layers - The network's layers in order.
 * @param x - Inputs `[n, d]`.
 * @param options - target (default 0) and baseline (default zeros).
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { add, matmul, relu, toArray } from 'whitebox-engine/tensor'
 * const layers = [{ fn: (v) => add(matmul(v, [[1], [1]]), -1) }, { fn: relu, nonlinear: true }]
 * // f(x) = relu(x1 + x2 - 1): from the zero baseline (f = 0) to (2, 1) (f = 2), shared as 4/3 and 2/3.
 * toArray(explain.deepLift(layers, [[2, 1]])).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[1.333333333333, 0.666666666667]]
 * ```
 */
export function deepLift(
  layers: readonly DeepLiftLayer[],
  x: TensorLike,
  options: { readonly target?: Target; readonly baseline?: TensorLike } = {},
): Tensor {
  const xt = concrete(x, 'deepLift')
  const base = baselineOf(xt, options.baseline, 'deepLift')
  const inputs: { readonly x: Tensor; readonly b: Tensor }[] = []
  let cx: Tensor = xt
  let cb: Tensor = base
  for (const layer of layers) {
    inputs.push({ x: cx, b: cb })
    cx = concrete(layer.fn(cx), 'deepLift')
    cb = concrete(layer.fn(cb), 'deepLift')
  }
  const n = cx.shape[0] ?? 1
  const k = cx.shape.length === 2 ? (cx.shape[1] ?? 1) : 1
  const m = new Float64Array(n * k)
  for (let i = 0; i < n; i++)
    m[i * k + (typeof options.target === 'object' ? (options.target[i] ?? 0) : (options.target ?? 0))] = 1
  let mult: Tensor = fromStorage(m, cx.shape, 'float64')
  for (let l = layers.length - 1; l >= 0; l--) {
    const layer = layers[l] as DeepLiftLayer
    const io = inputs[l] as { readonly x: Tensor; readonly b: Tensor }
    if (layer.nonlinear === true) {
      const yx = toFlat(layer.fn(io.x))
      const yb = toFlat(layer.fn(io.b))
      const grad = toFlat(vjp(layer.fn, io.x as Value).pullback(mult) as Tensor)
      const md = mult.data as Float64Array
      const xd = io.x.data as Float64Array
      const bd = io.b.data as Float64Array
      mult = fromStorage(
        Float64Array.from(md, (v, i) => {
          const dIn = (xd[i] ?? 0) - (bd[i] ?? 0)
          return Math.abs(dIn) < 1e-10 ? (grad[i] ?? 0) : (v * ((yx[i] ?? 0) - (yb[i] ?? 0))) / dIn
        }),
        io.x.shape,
        'float64',
      )
    } else {
      mult = vjp(layer.fn, io.x as Value).pullback(mult) as Tensor
    }
  }
  return mul(sub(xt, base), mult) as Tensor
}

/** Options of {@link gradientShap}. */
export interface GradientShapOptions {
  readonly target?: Target
  /** Samples per input (default 50). */
  readonly nSamples?: number
  /** The standard deviation of the Gaussian noise added to the inputs (default 0, captum's). */
  readonly stdev?: number
  readonly key?: Key
}

/**
 * GradientSHAP (Lundberg and Lee 2017; captum GradientShap): expected gradients with noise on the inputs. Each sample
 * draws a baseline row from the background, a path position alpha in [0, 1] and Gaussian noise on the input, and
 * contributes (x + noise - baseline) times the gradient at baseline + alpha (x + noise - baseline); the attributions
 * approximate SHAP values of a model that is nearly linear between the background and the input.
 *
 * @param model - The differentiable model.
 * @param x - Inputs `[n, d]`.
 * @param background - Baseline rows `[m, d]`.
 * @param options - target, nSamples, stdev and key.
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * // For a linear model: w (x - baseline) whatever the samples, when the baselines agree.
 * toArray(explain.gradientShap((x) => matmul(x, [[2], [1]]), [[3, 3]], [[1, 1]], { nSamples: 4 })) // => [[4, 2]]
 * ```
 */
export function gradientShap(
  model: DifferentiableModel,
  x: TensorLike,
  background: TensorLike,
  options: GradientShapOptions = {},
): Tensor {
  const xt = concrete(x, 'gradientShap')
  const bg = matrixView(background, 'gradientShap')
  const n = xt.shape[0] ?? 1
  const d = xt.shape[1] ?? 1
  if (bg.d !== d) throw new WhiteboxError('shape', 'gradientShap: the background must have the features of x')
  const ns = options.nSamples ?? 50
  if (!(Number.isInteger(ns) && ns >= 1)) throw new WhiteboxError('value', 'gradientShap: nSamples must be >= 1')
  const key = options.key ?? keyFor({}, 'gradientShap')
  const sd = options.stdev ?? 0
  const xd = xt.data as Float64Array
  const total = new Float64Array(n * d)
  for (let s = 0; s < ns; s++) {
    const rowsIdx = toFlat(randint(foldIn(key, 3 * s), [n], 0, bg.n))
    const alpha = toFlat(uniform(foldIn(key, 3 * s + 1), [n]))
    const noise = sd === 0 ? undefined : toFlat(normal(foldIn(key, 3 * s + 2), [n, d]))
    const noisy = Float64Array.from(xd, (v, i) => v + (noise === undefined ? 0 : sd * (noise[i] ?? 0)))
    const base = Float64Array.from(
      { length: n * d },
      (_, i) => bg.data[(rowsIdx[Math.floor(i / d)] ?? 0) * d + (i % d)] ?? 0,
    )
    const point = Float64Array.from(base, (b, i) => b + (alpha[Math.floor(i / d)] ?? 0) * ((noisy[i] ?? 0) - b))
    const g = targetGradient(model, fromStorage(point, xt.shape, 'float64'), options.target ?? 0).data as Float64Array
    for (let i = 0; i < total.length; i++)
      total[i] = (total[i] ?? 0) + (((noisy[i] ?? 0) - (base[i] ?? 0)) * (g[i] ?? 0)) / ns
  }
  return fromStorage(total, xt.shape, 'float64')
}

/** The aggregation of a noise tunnel (captum's NoiseTunnel types). */
export type NoiseTunnelType = 'smoothgrad' | 'smoothgrad_sq' | 'vargrad'

/** Options of {@link noiseTunnel}. */
export interface NoiseTunnelOptions {
  /** How the noisy attributions combine (default `smoothgrad`, their mean). */
  readonly type?: NoiseTunnelType
  /** Noisy copies (default 25). */
  readonly nSamples?: number
  /** The noise standard deviation, absolute (default 0.15). */
  readonly stdev?: number
  readonly key?: Key
}

/**
 * A noise tunnel over any attribution method (Smilkov et al. 2017; Adebayo et al. 2018; captum NoiseTunnel): compute
 * the attributions of Gaussian-perturbed copies of the input and combine them by their mean (SmoothGrad), the mean
 * of their squares (SmoothGrad squared) or their variance (VarGrad).
 *
 * @param attribute - An attribution method on inputs `[n, d]` (saliency, integrated gradients, ...).
 * @param x - Inputs `[n, d]`.
 * @param options - type, nSamples, stdev and key.
 * @returns Attributions `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, toArray } from 'whitebox-engine/tensor'
 * // A linear model's gradient does not change under noise, so its VarGrad is zero.
 * const f = (x) => matmul(x, [[2], [-3]])
 * toArray(explain.noiseTunnel((v) => explain.saliency(f, v, { abs: false }), [[1, 1]], { type: 'vargrad', nSamples: 5 })) // => [[0, 0]]
 * ```
 */
export function noiseTunnel(attribute: (x: Tensor) => Tensor, x: TensorLike, options: NoiseTunnelOptions = {}): Tensor {
  const xt = concrete(x, 'noiseTunnel')
  const ns = options.nSamples ?? 25
  if (!(Number.isInteger(ns) && ns >= 1)) throw new WhiteboxError('value', 'noiseTunnel: nSamples must be >= 1')
  const key = options.key ?? keyFor({}, 'noiseTunnel')
  const sd = options.stdev ?? 0.15
  const type = options.type ?? 'smoothgrad'
  const sum = new Float64Array(xt.data.length)
  const sumSq = new Float64Array(xt.data.length)
  for (let s = 0; s < ns; s++) {
    const noisy = add(xt, mul(normal(foldIn(key, s), xt.shape), sd)) as Tensor
    const a = toFlat(attribute(noisy))
    for (let i = 0; i < sum.length; i++) {
      sum[i] = (sum[i] ?? 0) + (a[i] ?? 0)
      sumSq[i] = (sumSq[i] ?? 0) + (a[i] ?? 0) ** 2
    }
  }
  const out = Float64Array.from(sum, (v, i) => {
    const mean = v / ns
    const sq = (sumSq[i] ?? 0) / ns
    return type === 'smoothgrad' ? mean : type === 'smoothgrad_sq' ? sq : Math.max(0, sq - mean * mean)
  })
  return fromStorage(out, xt.shape, 'float64')
}

/** How a modified backward pass treats the nonlinear layers. */
type BackwardRule = 'guided' | 'deconvolution'

function modifiedBackward(
  layers: readonly DeepLiftLayer[],
  x: TensorLike,
  target: Target,
  rule: BackwardRule,
  who: string,
): Tensor {
  const xt = concrete(x, who)
  const inputs: Tensor[] = []
  let h: Tensor = xt
  for (const layer of layers) {
    inputs.push(h)
    h = concrete(layer.fn(h), who)
  }
  const n = h.shape[0] ?? 1
  const k = h.shape.length >= 2 ? h.data.length / n : 1
  const cot = new Float64Array(h.data.length)
  for (let i = 0; i < n; i++) cot[i * k + (typeof target === 'number' ? target : (target[i] ?? 0))] = 1
  let g: Tensor = fromStorage(cot, h.shape, 'float64')
  for (let l = layers.length - 1; l >= 0; l--) {
    const layer = layers[l] as DeepLiftLayer
    const input = inputs[l] as Tensor
    if (layer.nonlinear === true) {
      // Guided backpropagation passes the positive part of the ordinary gradient; deconvolution the positive part of
      // the incoming signal, ignoring the forward pass.
      const passed = rule === 'guided' ? toFlat(vjp(layer.fn, input as Value).pullback(g) as Tensor) : toFlat(g)
      g = fromStorage(
        Float64Array.from(passed, (v) => Math.max(0, v)),
        input.shape,
        'float64',
      )
    } else g = vjp(layer.fn, input as Value).pullback(g) as Tensor
  }
  return g
}

/**
 * Guided backpropagation (Springenberg et al. 2015; captum GuidedBackprop): the gradient of the target output through
 * a sequential network, except that at every ReLU only the positive part of the backward signal passes (on top of the
 * ReLU's own gate), which leaves sharp, image-like maps.
 *
 * @param layers - The network's layers in order, its ReLUs marked `nonlinear`.
 * @param x - Inputs (`[n, d]`, or images the first layer reads).
 * @param options - target (default 0).
 * @returns Attributions shaped like x.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, relu, toArray } from 'whitebox-engine/tensor'
 * const layers = [{ fn: (v) => matmul(v, [[1, -1], [1, 1]]) }, { fn: relu, nonlinear: true }, { fn: (v) => matmul(v, [[1], [-1]]) }]
 * // The second hidden unit's negative weight is blocked on the way back.
 * toArray(explain.guidedBackprop(layers, [[1, 2]])) // => [[1, 1]]
 * ```
 */
export function guidedBackprop(
  layers: readonly DeepLiftLayer[],
  x: TensorLike,
  options: { readonly target?: Target } = {},
): Tensor {
  return modifiedBackward(layers, x, options.target ?? 0, 'guided', 'guidedBackprop')
}

/**
 * Deconvolution (Zeiler and Fergus 2014; captum Deconvolution): like {@link guidedBackprop} but at every ReLU the
 * backward signal is rectified without the forward pass's gate.
 *
 * @param layers - The network's layers in order, its ReLUs marked `nonlinear`.
 * @param x - Inputs.
 * @param options - target (default 0).
 * @returns Attributions shaped like x.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, relu, toArray } from 'whitebox-engine/tensor'
 * const layers = [{ fn: (v) => matmul(v, [[1, -1], [1, 1]]) }, { fn: relu, nonlinear: true }, { fn: (v) => matmul(v, [[1], [1]]) }]
 * // Both hidden units pass a signal back, though the first is inactive at x.
 * toArray(explain.deconvolution(layers, [[-2, 1]])) // => [[0, 2]]
 * ```
 */
export function deconvolution(
  layers: readonly DeepLiftLayer[],
  x: TensorLike,
  options: { readonly target?: Target } = {},
): Tensor {
  return modifiedBackward(layers, x, options.target ?? 0, 'deconvolution', 'deconvolution')
}

/**
 * Grad-CAM (Selvaraju et al. 2017; captum LayerGradCam): weight each channel of a convolutional layer's activations by
 * the spatial mean of the target output's gradient with respect to it, sum over channels and keep the positive part:
 * a coarse map of where in the layer's grid the evidence for the target lies.
 *
 * @param head - The network up to the layer: inputs to activations `[n, c, h, w]` (or `[n, c, l]`).
 * @param tail - The rest: activations to outputs `[n, k]`.
 * @param x - Inputs.
 * @param options - target (default 0) and relu (default true; false keeps negative evidence).
 * @returns The maps `[n, h, w]` (or `[n, 1, l]`) as nested arrays.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { matmul, reshape } from 'whitebox-engine/tensor'
 * // One channel on a 1 x 2 grid, the output its sum: every cell weighs in with its activation.
 * const head = (x) => reshape(x, [1, 1, 1, 2])
 * const tail = (a) => matmul(reshape(a, [1, 2]), [[1], [1]])
 * explain.gradCam(head, tail, [[3, 1]]) // => [[[3, 1]]]
 * ```
 */
export function gradCam(
  head: DifferentiableModel,
  tail: DifferentiableModel,
  x: TensorLike,
  options: { readonly target?: Target; readonly relu?: boolean } = {},
): number[][][] {
  const xt = concrete(x, 'gradCam')
  const a = concrete(head(xt), 'gradCam')
  if (a.shape.length < 3) throw new WhiteboxError('shape', 'gradCam: the layer must have channels and a grid')
  const [n = 1, c = 1] = a.shape
  const grid = a.shape.slice(2)
  const h = grid.length === 1 ? 1 : (grid[0] ?? 1)
  const w = grid.length === 1 ? (grid[0] ?? 1) : (grid[1] ?? 1)
  const cells = h * w
  const g = toFlat(targetGradient(tail, a, options.target ?? 0))
  const av = toFlat(a)
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: h }, (_, r) =>
      Array.from({ length: w }, (_, q) => {
        let s = 0
        for (let ch = 0; ch < c; ch++) {
          let alpha = 0
          const off = (i * c + ch) * cells
          for (let t = 0; t < cells; t++) alpha += g[off + t] ?? 0
          s += (alpha / cells) * (av[off + r * w + q] ?? 0)
        }
        return options.relu === false ? s : Math.max(0, s)
      }),
    ),
  )
}

/**
 * Resize a map to another grid (torch.nn.functional.interpolate on one channel): `nearest` (default, captum's for
 * Grad-CAM) takes each output cell's source floor(i in / out); `bilinear` interpolates with half-pixel centres
 * (align_corners false).
 *
 * @param map - The map `[h][w]`.
 * @param size - The output `[H, W]`.
 * @param mode - `nearest` or `bilinear`.
 * @returns The resized map `[H][W]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.interpolateMap([[1, 2]], [2, 4]) // => [[1, 1, 2, 2], [1, 1, 2, 2]]
 * ```
 */
export function interpolateMap(
  map: readonly (readonly number[])[],
  size: readonly [number, number],
  mode: 'nearest' | 'bilinear' = 'nearest',
): number[][] {
  const h = map.length
  const w = map[0]?.length ?? 0
  const [H, W] = size
  if (h === 0 || w === 0) throw new WhiteboxError('shape', 'interpolateMap: an empty map')
  const at = (r: number, q: number): number =>
    map[Math.min(h - 1, Math.max(0, r))]?.[Math.min(w - 1, Math.max(0, q))] ?? 0
  return Array.from({ length: H }, (_, i) =>
    Array.from({ length: W }, (_, j) => {
      if (mode === 'nearest') return at(Math.floor((i * h) / H), Math.floor((j * w) / W))
      const y = Math.max(0, ((i + 0.5) * h) / H - 0.5)
      const x = Math.max(0, ((j + 0.5) * w) / W - 0.5)
      const y0 = Math.floor(y)
      const x0 = Math.floor(x)
      const fy = y - y0
      const fx = x - x0
      return (
        (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) +
        fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1))
      )
    }),
  )
}

/** A layer of a network for {@link lrp}: a dense layer by its weights `[in, out]` and bias, or an elementwise one. */
export type LrpLayer =
  | { readonly kind: 'linear'; readonly weight: TensorLike; readonly bias?: TensorLike; readonly rule?: LrpRule }
  | { readonly kind: 'elementwise'; readonly fn: (v: Value) => Value }

/** A layer-wise relevance propagation rule. */
export type LrpRule =
  | { readonly name: 'epsilon'; readonly epsilon?: number }
  | { readonly name: 'gamma'; readonly gamma?: number }
  | { readonly name: 'alphaBeta'; readonly alpha?: number; readonly beta?: number }

/**
 * Layer-wise relevance propagation (Bach et al. 2015; Montavon et al. 2019; captum LRP) for dense networks: the
 * target output's value is the relevance at the top; each dense layer passes relevance down in proportion to each
 * input's contribution a_j w_jk to z_k, stabilised (the epsilon rule, z + epsilon sign z, default epsilon 1e-9),
 * favouring positive weights (the gamma rule, w + gamma w+, default 0.25), or splitting positive and negative parts
 * (alpha-beta, alpha z+ / sum z+ - beta z- / sum z-, default alpha 1, beta 0, without the bias); elementwise layers
 * (activations) pass it through unchanged. Relevance is nearly conserved, summing to the output minus what the biases
 * absorb.
 *
 * @param layers - The network's layers in order.
 * @param x - Inputs `[n, d]`.
 * @param options - target (default 0) and the rule of dense layers without their own (default epsilon).
 * @returns Relevances `[n, d]`.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { toArray } from 'whitebox-engine/tensor'
 * // One dense layer without bias: relevance is each input's contribution w x.
 * toArray(explain.lrp([{ kind: 'linear', weight: [[2], [-1]] }], [[1, 3]])).map((r) => r.map((v) => Math.round(v * 1e6) / 1e6)) // => [[2, -3]]
 * ```
 */
export function lrp(
  layers: readonly LrpLayer[],
  x: TensorLike,
  options: { readonly target?: Target; readonly rule?: LrpRule } = {},
): Tensor {
  const xt = concrete(x, 'lrp')
  const n = xt.shape[0] ?? 1
  const acts: Float64Array[] = []
  const widths: number[] = []
  let a = Float64Array.from(xt.data as Float64Array)
  let width = xt.shape[1] ?? 1
  const dense = layers.map((l) => {
    if (l.kind !== 'linear') return undefined
    const wt = concrete(l.weight, 'lrp')
    const [inW = 1, outW = 1] = wt.shape
    const b = l.bias === undefined ? new Float64Array(outW) : Float64Array.from(toFlat(asTensor(l.bias)))
    return { w: Float64Array.from(wt.data as Float64Array), b, inW, outW }
  })
  for (const [li, l] of layers.entries()) {
    acts.push(a)
    widths.push(width)
    const dl = dense[li]
    if (l.kind === 'linear' && dl !== undefined) {
      if (dl.inW !== width) throw new WhiteboxError('shape', `lrp: layer ${li} expects ${dl.inW} inputs, got ${width}`)
      const out = new Float64Array(n * dl.outW)
      for (let i = 0; i < n; i++)
        for (let k = 0; k < dl.outW; k++) {
          let s = dl.b[k] ?? 0
          for (let j = 0; j < width; j++) s += (a[i * width + j] ?? 0) * (dl.w[j * dl.outW + k] ?? 0)
          out[i * dl.outW + k] = s
        }
      a = out
      width = dl.outW
    } else if (l.kind === 'elementwise') {
      a = Float64Array.from(toFlat(concrete(l.fn(fromStorage(a, [n, width], 'float64')), 'lrp')))
    }
  }
  const k0 = width
  let rel = new Float64Array(n * k0)
  for (let i = 0; i < n; i++) {
    const t = typeof options.target === 'object' ? (options.target[i] ?? 0) : (options.target ?? 0)
    rel[i * k0 + t] = a[i * k0 + t] ?? 0
  }
  for (let li = layers.length - 1; li >= 0; li--) {
    const l = layers[li] as LrpLayer
    const dl = dense[li]
    if (l.kind !== 'linear' || dl === undefined) continue
    const input = acts[li] as Float64Array
    const inW = widths[li] ?? 1
    const rule = l.rule ?? options.rule ?? { name: 'epsilon' }
    const next = new Float64Array(n * inW)
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < dl.outW; k++) {
        const r = rel[i * dl.outW + k] ?? 0
        if (r === 0) continue
        if (rule.name === 'alphaBeta') {
          const alpha = rule.alpha ?? 1
          const beta = rule.beta ?? 0
          let zp = 0
          let zn = 0
          for (let j = 0; j < inW; j++) {
            const z = (input[i * inW + j] ?? 0) * (dl.w[j * dl.outW + k] ?? 0)
            if (z > 0) zp += z
            else zn += z
          }
          for (let j = 0; j < inW; j++) {
            const z = (input[i * inW + j] ?? 0) * (dl.w[j * dl.outW + k] ?? 0)
            const share =
              (z > 0 && zp > 0 ? (alpha * z) / (zp + 1e-9) : 0) - (z < 0 && zn < 0 ? (beta * z) / (zn - 1e-9) : 0)
            next[i * inW + j] = (next[i * inW + j] ?? 0) + share * r
          }
          continue
        }
        const gamma = rule.name === 'gamma' ? (rule.gamma ?? 0.25) : 0
        const eps = rule.name === 'epsilon' ? (rule.epsilon ?? 1e-9) : 1e-9
        const wOf = (j: number): number => {
          const v = dl.w[j * dl.outW + k] ?? 0
          return v + gamma * Math.max(0, v)
        }
        let z = dl.b[k] ?? 0
        for (let j = 0; j < inW; j++) z += (input[i * inW + j] ?? 0) * wOf(j)
        const denom = z + eps * (z >= 0 ? 1 : -1)
        for (let j = 0; j < inW; j++)
          next[i * inW + j] = (next[i * inW + j] ?? 0) + ((input[i * inW + j] ?? 0) * wOf(j) * r) / denom
      }
    }
    rel = next
  }
  return fromStorage(rel, [n, xt.shape[1] ?? 1], 'float64')
}
