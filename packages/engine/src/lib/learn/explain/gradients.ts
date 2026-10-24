import { vjp } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { gaussLegendre } from '../../numerics/quadrature/index.ts'
import { foldIn, type Key, normal, randint, uniform } from '../../random/index.ts'
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
} from '../../tensor/index.ts'
import { keyFor } from '../_shared.ts'
import { matrixView } from '../validation.ts'

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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
 * import { explain } from 'whitebox-engine/learn'
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
