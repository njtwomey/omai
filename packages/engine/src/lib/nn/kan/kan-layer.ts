/**
 * `KanLayer`: one layer of a Kolmogorov-Arnold network (Liu et al. 2024). Where an MLP layer puts a learnable weight on
 * each edge and a fixed activation at each node, a KAN layer puts a learnable univariate function on each of its
 * `in x out` edges, `phi(x) = w_b b(x) + w_s spline(x)` (a residual base function `b`, SiLU in the paper, plus a
 * weighted spline), and its nodes only sum. The splines are `SplineLayer`s (so the knots are the `splineKnots` slot and
 * may train), or any family of `kanBases`. Initialisation is the paper's (pykan's): small noise on the spline
 * coefficients, `w_s = 1 / sqrt(in)`, `w_b` uniform in `+-1 / sqrt(in)`. An input outside the grid stays finite and
 * differentiable: the spline continues with its boundary tangent (or is held), and the base function is defined
 * everywhere. A mask buffer switches edges off (pruning).
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { dict, type OptionTable, optionTable, type SpaceValue } from '../../space/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  mul,
  ones,
  relu as reluOp,
  reshape,
  softplus as softplusOp,
  sum,
  type Tensor,
  type TensorLike,
  tanh,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { silu } from '../functional/activations.ts'
import { type ApplyContext, assemble, childScope, initKey, Module, type Params } from '../module/module.ts'
import { type EdgeFunctions, kanBases } from './edge-bases.ts'

/** The residual base function `b` of an edge: a tensor function and the same function on numbers. */
export interface KanBaseFunction {
  readonly name: string
  /** The function on tensors; `undefined` for `none` (no base term at all). */
  readonly apply: ((x: Value) => Value) | undefined
  /** The function on numbers (for fitting and printing edges). */
  readonly value: (x: number) => number
}

/** What a base function is, for validity rules and selectors. */
export interface KanBaseProperties {
  /** Whether the edge has no base term. */
  readonly absent: boolean
}

const baseFunction = (
  absent: boolean,
  description: string,
  apply: ((x: Value) => Value) | undefined,
  value: (x: number) => number,
  name: string,
): {
  readonly description: string
  readonly space: ReturnType<typeof dict>
  readonly properties: KanBaseProperties
  readonly make: () => KanBaseFunction
} => ({ description, space: dict({}), properties: { absent }, make: () => ({ name, apply, value }) })

/**
 * The base function of a Kolmogorov-Arnold edge as one slot: `silu` (the paper's default), `none` (splines only),
 * `identity` (a linear skip term), `relu`, `tanh` and `softplus`.
 *
 * @example
 * ```ts
 * import { kanBaseFunctions } from 'mdml-engine/nn'
 * kanBaseFunctions.make({ kind: 'silu' }).value(0) // => 0
 * kanBaseFunctions.make({ kind: 'none' }).apply // => undefined
 * ```
 */
export const kanBaseFunctions: OptionTable<KanBaseFunction, KanBaseProperties> = optionTable({
  silu: baseFunction(
    false,
    'x sigmoid(x), the paper’s residual activation',
    (x) => silu(x),
    (x) => x / (1 + Math.exp(-x)),
    'silu',
  ),
  none: baseFunction(true, 'no base term: the edge is its spline alone', undefined, () => 0, 'none'),
  identity: baseFunction(
    false,
    'x, a linear skip term',
    (x) => x,
    (x) => x,
    'identity',
  ),
  relu: baseFunction(
    false,
    'max(x, 0)',
    (x) => reluOp(x),
    (x) => Math.max(x, 0),
    'relu',
  ),
  tanh: baseFunction(false, 'tanh(x)', (x) => tanh(x), Math.tanh, 'tanh'),
  softplus: baseFunction(
    false,
    'log(1 + exp(x))',
    (x) => softplusOp(x),
    (x) => (x > 30 ? x : Math.log1p(Math.exp(x))),
    'softplus',
  ),
})

/** Options of {@link KanLayer}. */
export interface KanLayerOptions {
  readonly inFeatures: number
  readonly outFeatures: number
  /** The grid size `G` (default 5): intervals of the spline. */
  readonly grid?: number
  /** The spline degree (default 3). */
  readonly degree?: number
  /** The `kanBases` option: the family of the edge functions (default `{ kind: 'bspline' }`). */
  readonly basis?: SpaceValue
  /** The `kanBaseFunctions` option: the residual base function (default `{ kind: 'silu' }`). */
  readonly base?: SpaceValue
  /** The `splineKnots` option of the spline family (default evenly spaced). */
  readonly knots?: SpaceValue
  /** Beyond the grid: continue with the boundary tangent (default) or hold the boundary value. */
  readonly outside?: 'clamp' | 'linear'
  /** The range of the inputs, one pair for all or one per input (default the range of `sample`, else `[-1, 1]`). */
  readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
  /** A sample of the inputs `[n, in]`: sets the range (and the quantile knots). */
  readonly sample?: TensorLike
  /** A bias per output node (default false: the paper has none; a spline can represent a constant). */
  readonly bias?: boolean
  /** The scale of the initial noise on the spline values (pykan's `noise_scale`, default 0.3). */
  readonly noise?: number
  /** The mean and spread of `w_b` before dividing by `sqrt(in)` (pykan's `scale_base_mu` 0 and `scale_base_sigma` 1). */
  readonly scaleBaseMu?: number
  readonly scaleBaseSigma?: number
  /** The initial `w_s` before dividing by `sqrt(in)` (pykan's `scale_sp`, default 1). */
  readonly scaleSpline?: number
  readonly key?: Key
}

/**
 * A layer of `in x out` learnable univariate functions on the edges, summed at the output nodes: `y_o = b_o + sum_i
 * m_{o,i} phi_{o,i}(x_i)` with `phi_{o,i}(x) = w_b b(x) + w_s sum_k c_{o,i,k} B_k(x)`. Parameters are `edge` (the
 * coefficients, and the free knots when the spline's are), `wBase [out, in]`, `wSpline [out, in]` and optionally
 * `bias [out]`; the buffers are the edge's (ranges, fixed knots) and `mask [out, in]`.
 *
 * @concept splines
 * @example
 * ```ts
 * import { KanLayer } from 'mdml-engine/nn'
 * import { asTensor } from 'mdml-engine/tensor'
 * const layer = new KanLayer({ inFeatures: 2, outFeatures: 3, grid: 5, degree: 3 })
 * layer.forward([[0.2, -0.4]]).shape // => [1, 3]
 * layer.activations(layer.params, asTensor([[0.2, -0.4]])).shape // => [1, 2, 3]
 * ```
 */
export class KanLayer extends Module {
  readonly kind = 'KanLayer'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly grid: number
  readonly degree: number
  readonly edge: EdgeFunctions
  readonly baseFunction: KanBaseFunction
  readonly hasBias: boolean
  private readonly scaleBaseMu: number
  private readonly scaleBaseSigma: number
  private readonly scaleSpline: number

  constructor(options: KanLayerOptions) {
    super()
    this.inFeatures = options.inFeatures
    this.outFeatures = options.outFeatures
    this.grid = options.grid ?? 5
    this.degree = options.degree ?? 3
    if (!(Number.isInteger(this.grid) && this.grid >= 2))
      throw new MdmlError('value', 'KanLayer: grid must be an integer >= 2')
    this.baseFunction = kanBaseFunctions.make(options.base)
    this.hasBias = options.bias ?? false
    this.scaleBaseMu = options.scaleBaseMu ?? 0
    this.scaleBaseSigma = options.scaleBaseSigma ?? 1
    this.scaleSpline = options.scaleSpline ?? 1
    const key = initKey('KanLayer', options.key)
    const noise = options.noise ?? 0.3
    this.edge = kanBases.make(options.basis).create({
      inFeatures: this.inFeatures,
      outFeatures: this.outFeatures,
      grid: this.grid,
      degree: this.degree,
      knots: options.knots ?? { kind: 'uniform' },
      outside: options.outside ?? 'linear',
      range: options.range ?? (options.sample === undefined ? [-1, 1] : undefined),
      ...(options.sample === undefined ? {} : { sample: options.sample }),
      initScale: noise / (this.grid * Math.sqrt(12)),
      key: foldIn(key, 0),
    } as Parameters<ReturnType<typeof kanBases.make>['create']>[0])
    this.params = this.initParams(key)
    this.buffers = this.initBuffers()
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return [['edge', this.edge]]
  }

  override initParams(key: Key): Params {
    const shape = [this.outFeatures, this.inFeatures]
    const root = Math.sqrt(this.inFeatures)
    const u = uniform(foldIn(key, 1), shape, { low: -1, high: 1 })
    const own: Record<string, unknown> = {
      wSpline: fromStorage(
        new Float64Array(this.outFeatures * this.inFeatures).fill(this.scaleSpline / root),
        shape,
        'float64',
      ),
    }
    if (this.baseFunction.apply !== undefined)
      own['wBase'] = fromStorage(
        Float64Array.from(u.data, (v) => (this.scaleBaseMu + this.scaleBaseSigma * v) / root),
        shape,
        'float64',
      )
    if (this.hasBias) own['bias'] = zeros([this.outFeatures])
    return assemble([['edge', this.edge.initParams(foldIn(key, 0))]], own)
  }

  override initBuffers(): Params {
    return assemble([['edge', this.edge.initBuffers()]], { mask: ones([this.outFeatures, this.inFeatures]) })
  }

  /** The inputs as `[n, in]`. */
  private inputs(x: Value): Value {
    if (x.shape.length === 1 && this.inFeatures === 1) return reshape(x, [x.shape[0] ?? 0, 1])
    if (x.shape.length !== 2 || x.shape[1] !== this.inFeatures)
      throw new MdmlError('shape', `KanLayer: expected [n, ${this.inFeatures}], got [${x.shape.join(', ')}]`)
    return x
  }

  /**
   * The activation of every edge `[n, in, out]`: `m_{o,i} phi_{o,i}(x_i)` before the sum at the nodes.
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, in]`.
   * @param buffers - The buffers (default the layer's).
   * @returns The activations.
   */
  activations(params: Params, x: Value, buffers?: Params): Value {
    const b = buffers ?? this.buffers
    const xs = this.inputs(x)
    let phi = mul(
      this.edge.terms(params['edge'] as Params, xs, b['edge'] as Params | undefined),
      transpose(params['wSpline'] as Value),
    )
    if (this.baseFunction.apply !== undefined) {
      const n = xs.shape[0] ?? 0
      phi = add(
        phi,
        mul(reshape(this.baseFunction.apply(xs), [n, this.inFeatures, 1]), transpose(params['wBase'] as Value)),
      )
    }
    return mul(phi, transpose(b['mask'] as Value))
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const edgeContext = scope.child('edge', 0)
    const b = ctx.buffers ?? this.buffers
    const y = sum(this.activations(params, x, { ...b, edge: edgeContext.buffers ?? (b['edge'] as Params) }), 1)
    return this.hasBias ? add(y, params['bias'] as Value) : y
  }

  /**
   * One edge on a grid: `phi_{o,i}` for input `feature` and output `output` (the mask applied, the bias not). The spline
   * part is the edge function's own `curve`.
   *
   * @param params - The parameters.
   * @param grid - The points.
   * @param feature - The input node `i` (default 0).
   * @param output - The output node `o` (default 0).
   * @param buffers - The buffers (default the layer's).
   * @returns The values at the grid, `[grid.length]`.
   */
  edgeCurve(params: Params, grid: ArrayLike<number>, feature = 0, output = 0, buffers?: Params): Tensor {
    const b = buffers ?? this.buffers
    const at = (name: string): number => toFlat(params[name] as Value)[output * this.inFeatures + feature] as number
    const m = toFlat(b['mask'] as Value)[output * this.inFeatures + feature] as number
    const spline = toFlat(
      this.edge.curve(params['edge'] as Params, grid, feature, output, b['edge'] as Params | undefined),
    )
    const wb = this.baseFunction.apply === undefined ? 0 : at('wBase')
    const ws = at('wSpline')
    return asTensor(
      Array.from(grid, (g, k) => m * (ws * (spline[k] as number) + wb * this.baseFunction.value(g))),
    ) as Tensor
  }

  /** The input range of every node `[in]` as `[lo, hi]` pairs, from the buffers. */
  inputRange(buffers?: Params): [number, number][] {
    const range = toFlat(
      (((buffers ?? this.buffers)['edge'] as Params)['range'] ?? this.edge.buffers['range']) as Value,
    )
    return Array.from({ length: this.inFeatures }, (_, i) => [range[2 * i] as number, range[2 * i + 1] as number])
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in=${this.inFeatures}, out=${this.outFeatures}, grid=${this.grid}, degree=${this.degree}, edge=${this.edge.kind}, base=${this.baseFunction.name}`
  }
}
