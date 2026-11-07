/**
 * `SplineLayer`: a layer of learnable univariate splines, the building block of additive networks and of
 * Kolmogorov-Arnold networks. Edge `(o, i)` carries a B-spline `phi_{o,i}(x_i) = sum_k c_{o,i,k} B_k(x_i)` on the knots of
 * input `i`, and output `o` is the sum of its edges plus a bias, `y_o = b_o + sum_i phi_{o,i}(x_i)`. With one output it
 * is an additive model of the inputs; with more it is a KAN layer. The knots are a slot (`splineKnots`: fixed evenly
 * or at quantiles, or free and trained with the coefficients through the differentiable basis), the coefficient shape
 * is a slot (`splineShapes`: monotone, convex, bounded), and the P-spline difference penalty on the coefficients is a
 * method, so a trainer adds it to the loss.
 */

import { MdmlError, maxOf, minOf } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import type { SpaceValue } from '../../space/index.ts'
import {
  add,
  asTensor,
  einsum,
  fromStorage,
  reshape,
  sliceAxis,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { initKey, Module, type Params } from '../module/module.ts'
import {
  type BasisOptions,
  type Boundary,
  bsplineBasis,
  bsplineBasisDerivative,
  extendKnots,
  grevilleAbscissae,
  knotLogits,
  knotsFromLogits,
  type Outside,
} from './bspline.ts'
import { type KnotPlacement, placeInterior, type ShapeTransform, splineKnots, splineShapes } from './knots.ts'

/** Options of {@link SplineLayer}. */
export interface SplineLayerOptions {
  /** The number of input features (one spline per feature and output). */
  readonly inFeatures: number
  /** The number of outputs (1 for an additive model; more for a Kolmogorov-Arnold layer). */
  readonly outFeatures: number
  /** The number of interior knots (default 8): the layer has `interiorKnots + degree + 1` coefficients per spline. */
  readonly interiorKnots?: number
  /** The degree (default 3, cubic). */
  readonly degree?: number
  /** The `splineKnots` option: `{ kind: 'uniform' }` (default), `{ kind: 'quantile' }` or `{ kind: 'learnable', init, minGap }`. */
  readonly knots?: SpaceValue
  /** The `splineShapes` option for the coefficients (default `{ kind: 'none' }`). */
  readonly shape?: SpaceValue
  /** The range of the inputs, one pair for all or one per feature (default the range of `sample`, else `[0, 1]`). */
  readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
  /** A sample of the inputs `[n, inFeatures]`: sets the range and the quantile knots. */
  readonly sample?: TensorLike
  /** Add a bias per output (default true). */
  readonly bias?: boolean
  /** Beyond the range: hold the boundary value (default) or continue with the boundary tangent. */
  readonly outside?: Outside
  /**
   * How the knot vector ends: `clamped` (default, boundary knots repeated) or `extended` (P-splines: equally spaced knots
   * beyond the range, the basis of `gam`'s `ps` smooth when the interior knots are even).
   */
  readonly boundary?: Boundary
  /** The standard deviation of the raw coefficients at initialisation (default 0.1). */
  readonly initScale?: number
  readonly key?: Key
}

/**
 * The sum of squared `order`th differences of the coefficients along the last axis: Eilers and Marx's P-spline penalty
 * `beta' D' D beta`, for every spline of a coefficient array at once.
 *
 * @param coefficients - The coefficients `[..., K]`.
 * @param order - The difference order (default 2; at most `K - 1`).
 * @returns The scalar penalty.
 * @throws MdmlError `value` when the order is not in `[0, K)`.
 * @concept splines
 * @example
 * ```ts
 * import { differenceRoughness } from 'mdml-engine/nn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(differenceRoughness([[1, 2, 4, 8]], 2))[0] // => 5
 * ```
 */
export function differenceRoughness(coefficients: TensorLike, order = 2): Value {
  let c = asTensor(coefficients)
  const axis = c.shape.length - 1
  const k = c.shape[axis] ?? 0
  if (!(Number.isInteger(order) && order >= 0 && order < k))
    throw new MdmlError('value', 'differenceRoughness: order must be in [0, K)')
  for (let o = 0; o < order; o++) {
    const n = c.shape[axis] ?? 0
    c = sub(sliceAxis(c, axis, 1, n), sliceAxis(c, axis, 0, n - 1))
  }
  return sum(square(c))
}

/**
 * A layer of learnable univariate B-splines, summed per output: `y_o = b_o + sum_i sum_k c_{o,i,k} B_k(x_i)`.
 * Parameters are `coef [out, in, K]`, `bias [out]` and, for free knots, `knotLogits [in, interior + 1]`; the ranges
 * and fixed knots are buffers (so they travel with a saved model). Input is `[n, in]` (or `[n]` for one feature).
 *
 * @concept splines
 * @example
 * ```ts
 * import { SplineLayer } from 'mdml-engine/nn'
 * const layer = new SplineLayer({ inFeatures: 2, outFeatures: 1, interiorKnots: 4 })
 * layer.forward([[0.2, 0.7], [0.5, 0.1]]).shape // => [2, 1]
 * layer.basisSize // => 8
 * ```
 */
export class SplineLayer extends Module {
  readonly kind = 'SplineLayer'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly degree: number
  readonly interiorKnots: number
  /** The number of coefficients per spline, `interiorKnots + degree + 1`. */
  readonly basisSize: number
  readonly placement: KnotPlacement
  readonly shapeTransform: ShapeTransform
  readonly hasBias: boolean
  readonly outside: Outside
  readonly boundary: Boundary
  readonly initScale: number
  private readonly lo: number[]
  private readonly hi: number[]
  private readonly start: number[][]

  constructor(options: SplineLayerOptions) {
    super()
    this.inFeatures = options.inFeatures
    this.outFeatures = options.outFeatures
    this.degree = options.degree ?? 3
    this.interiorKnots = options.interiorKnots ?? 8
    if (!(Number.isInteger(this.interiorKnots) && this.interiorKnots >= 1))
      throw new MdmlError('value', 'SplineLayer: interiorKnots must be an integer >= 1')
    if (!(Number.isInteger(this.degree) && this.degree >= 0))
      throw new MdmlError('value', 'SplineLayer: degree must be an integer >= 0 (0 is piecewise constant, 1 linear)')
    this.basisSize = this.interiorKnots + this.degree + 1
    this.placement = splineKnots.make(options.knots)
    this.shapeTransform = splineShapes.make(options.shape)
    this.hasBias = options.bias ?? true
    this.outside = options.outside ?? 'clamp'
    this.boundary = options.boundary ?? 'clamped'
    this.initScale = options.initScale ?? 0.1
    const d = this.inFeatures
    const sample = options.sample === undefined ? undefined : asTensor(options.sample)
    if (sample !== undefined && (sample.shape.length !== 2 || sample.shape[1] !== d))
      throw new MdmlError('shape', `SplineLayer: sample must be [n, ${d}]`)
    const values = sample === undefined ? undefined : toFlat(sample)
    const rows = sample === undefined ? 0 : (sample.shape[0] ?? 0)
    const column = (i: number): number[] | undefined =>
      values === undefined ? undefined : Array.from({ length: rows }, (_, r) => values[r * d + i] as number)
    const given = (i: number): readonly [number, number] | undefined => {
      const r = options.range
      if (r === undefined) return undefined
      return (typeof r[0] === 'number' ? r : r[i]) as readonly [number, number] | undefined
    }
    this.lo = []
    this.hi = []
    this.start = []
    for (let i = 0; i < d; i++) {
      const col = column(i)
      let [lo, hi] = given(i) ?? (col === undefined ? [0, 1] : [minOf(col), maxOf(col)])
      if (!(hi > lo)) [lo, hi] = [lo - 0.5, lo + 0.5]
      this.lo.push(lo)
      this.hi.push(hi)
      // Without a sample (a model rebuilt to load a saved state) the knots start even; the buffers then replace them.
      this.start.push(
        placeInterior(
          col === undefined ? 'uniform' : this.placement.rule,
          this.interiorKnots,
          lo,
          hi,
          col,
          this.placement.minGap,
        ),
      )
    }
    const k = initKey('SplineLayer', options.key)
    this.params = this.initParams(k)
    this.buffers = this.initBuffers()
  }

  override initParams(key: Key): Params {
    const [o, d, k] = [this.outFeatures, this.inFeatures, this.basisSize]
    const raw = Float64Array.from(normal(foldIn(key, 0), [o, d, k], { scale: this.initScale }).data)
    for (let j = 0; j < raw.length; j++)
      if (j % k >= this.shapeTransform.smallFrom) raw[j] = (raw[j] ?? 0) + this.shapeTransform.smallShift
    const out: Record<string, unknown> = { coef: fromStorage(raw, [o, d, k], 'float64') }
    if (this.hasBias) out['bias'] = fromStorage(new Float64Array(o), [o], 'float64')
    if (this.placement.learnable) {
      const logits = this.start.map((inner, i) =>
        toFlat(knotLogits(inner, this.lo[i] as number, this.hi[i] as number, this.placement.minGap)),
      )
      out['knotLogits'] = fromStorage(Float64Array.from(logits.flat()), [d, this.interiorKnots + 1], 'float64')
    }
    return out
  }

  override initBuffers(): Params {
    const d = this.inFeatures
    const out: Record<string, unknown> = {
      range: fromStorage(Float64Array.from(this.lo.flatMap((lo, i) => [lo, this.hi[i] as number])), [d, 2], 'float64'),
    }
    if (!this.placement.learnable)
      out['interior'] = fromStorage(Float64Array.from(this.start.flat()), [d, this.interiorKnots], 'float64')
    return out
  }

  /** The inputs as `[n, in]`. */
  private inputs(x: Value): Value {
    if (x.shape.length === 1 && this.inFeatures === 1) return reshape(x, [x.shape[0] ?? 0, 1])
    if (x.shape.length !== 2 || x.shape[1] !== this.inFeatures)
      throw new MdmlError('shape', `SplineLayer: expected [n, ${this.inFeatures}], got [${x.shape.join(', ')}]`)
    return x
  }

  private bounds(buffers: Params | undefined): { lo: number[]; hi: number[] } {
    const range = (buffers ?? this.buffers)['range']
    if (range === undefined) return { lo: this.lo, hi: this.hi }
    const r = toFlat(range as Value)
    return { lo: this.lo.map((_, i) => r[2 * i] as number), hi: this.hi.map((_, i) => r[2 * i + 1] as number) }
  }

  /**
   * The interior knots `[in, interiorKnots]`: the buffer for fixed knots, the softmax-of-gaps map of the logits (traced
   * when the parameters are) for free ones.
   *
   * @param params - The parameters.
   * @param buffers - The buffers (default the module's).
   * @returns The knots.
   */
  interior(params: Params, buffers?: Params): Value {
    const b = buffers ?? this.buffers
    if (!this.placement.learnable) return b['interior'] as Value
    const { lo, hi } = this.bounds(b)
    return knotsFromLogits(params['knotLogits'] as Value, lo, hi, this.placement.minGap)
  }

  /**
   * The full knot vectors `[in, interiorKnots + 2 (degree + 1)]`, boundary knots repeated.
   *
   * @param params - The parameters.
   * @param buffers - The buffers (default the module's).
   * @returns The knot vectors.
   */
  knotVectors(params: Params, buffers?: Params): Value {
    const { lo, hi } = this.bounds(buffers)
    return extendKnots(this.interior(params, buffers), lo, hi, this.degree, this.boundary)
  }

  /**
   * The effective coefficients `[out, in, K]`: the raw ones through the shape constraint.
   *
   * @param params - The parameters.
   * @param buffers - The buffers (default the module's).
   * @returns The coefficients.
   */
  coefficients(params: Params, buffers?: Params): Value {
    const raw = params['coef'] as Value
    if (this.shapeTransform.name === 'none') return raw
    return this.shapeTransform.apply(raw, grevilleAbscissae(this.knotVectors(params, buffers), this.degree))
  }

  /**
   * The basis of every input `[n, in, K]` at the layer's knots.
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, in]`.
   * @param buffers - The buffers (default the module's).
   * @returns The basis.
   */
  basis(params: Params, x: Value, buffers?: Params): Value {
    const options: BasisOptions = { outside: this.outside, boundary: this.boundary }
    return bsplineBasis(this.inputs(x), this.knotVectors(params, buffers), this.degree, options)
  }

  /**
   * The contribution of every edge `[n, in, out]` (before the sum over inputs and the bias): the partial effects of an
   * additive model, the activations of a Kolmogorov-Arnold layer.
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, in]`.
   * @param buffers - The buffers (default the module's).
   * @returns The contributions.
   */
  terms(params: Params, x: Value, buffers?: Params): Value {
    return einsum('nik,oik->nio', this.basis(params, x, buffers), this.coefficients(params, buffers))
  }

  apply(params: Params, x: Value, ctx: { readonly buffers?: Params } = {}): Value {
    const y = einsum('nik,oik->no', this.basis(params, x, ctx.buffers), this.coefficients(params, ctx.buffers))
    return this.hasBias ? add(y, params['bias'] as Value) : y
  }

  /**
   * The derivative of every edge with respect to its input `[n, in, out]` (the slope of each spline at the points).
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, in]`.
   * @param order - The derivative order (default 1).
   * @param buffers - The buffers (default the module's).
   * @returns The derivatives.
   */
  termDerivatives(params: Params, x: Value, order = 1, buffers?: Params): Value {
    const d = bsplineBasisDerivative(this.inputs(x), this.knotVectors(params, buffers), this.degree, order, {
      outside: this.outside,
      boundary: this.boundary,
    })
    return einsum('nik,oik->nio', d, this.coefficients(params, buffers))
  }

  /**
   * The basis functions of one spline on a grid with their coefficients: row `g` of `basis` is the basis at `grid[g]`,
   * and `basis[g][k] * coefficients[k]` summed over `k` is {@link curve}. The weighted functions are the pieces the
   * curve is built from.
   *
   * @param params - The parameters.
   * @param grid - The points.
   * @param feature - The input (default 0).
   * @param output - The output (default 0).
   * @param buffers - The buffers (default the module's).
   * @returns The basis `[grid.length][K]` and the coefficients `[K]`.
   * @example
   * ```ts
   * import { SplineLayer } from 'mdml-engine/nn'
   * const layer = new SplineLayer({ inFeatures: 1, outFeatures: 1, interiorKnots: 3 })
   * const { basis, coefficients } = layer.weightedBasis(layer.params, [0.5])
   * [basis.length, basis[0]?.length === coefficients.length] // => [1, true]
   * ```
   */
  weightedBasis(
    params: Params,
    grid: ArrayLike<number>,
    feature = 0,
    output = 0,
    buffers?: Params,
  ): { readonly basis: number[][]; readonly coefficients: number[] } {
    const knots = reshape(sliceAxis(this.knotVectors(params, buffers), 0, feature, feature + 1), [-1])
    const coef = toFlat(
      sliceAxis(sliceAxis(this.coefficients(params, buffers), 0, output, output + 1), 1, feature, feature + 1),
    )
    const b = toFlat(
      bsplineBasis(Array.from(grid), knots, this.degree, { outside: this.outside, boundary: this.boundary }),
    )
    const k = this.basisSize
    return {
      basis: Array.from({ length: grid.length }, (_, g) => Array.from(b.slice(g * k, (g + 1) * k))),
      coefficients: Array.from(coef),
    }
  }

  /**
   * One spline on a grid, without the bias: the shape function of input `feature` towards output `output`.
   *
   * @param params - The parameters.
   * @param grid - The points.
   * @param feature - The input (default 0).
   * @param output - The output (default 0).
   * @param buffers - The buffers (default the module's).
   * @returns The values at the grid, `[grid.length]`.
   */
  curve(params: Params, grid: ArrayLike<number>, feature = 0, output = 0, buffers?: Params): Tensor {
    const knots = reshape(sliceAxis(this.knotVectors(params, buffers), 0, feature, feature + 1), [-1])
    const coef = reshape(
      sliceAxis(sliceAxis(this.coefficients(params, buffers), 0, output, output + 1), 1, feature, feature + 1),
      [this.basisSize, 1],
    )
    const values = reshape(
      einsum(
        'nk,kj->nj',
        bsplineBasis(Array.from(grid), knots, this.degree, { outside: this.outside, boundary: this.boundary }),
        coef,
      ),
      [grid.length],
    )
    return asTensor(toFlat(values)) as Tensor
  }

  /**
   * The P-spline penalty of the layer: the sum over splines of squared `order`th differences of the effective
   * coefficients, `sum beta' D' D beta` (the same quantity `lib/gam`'s `ps` penalty measures, one smoothing parameter
   * for all splines).
   *
   * @param params - The parameters.
   * @param order - The difference order (default 2).
   * @param buffers - The buffers (default the module's).
   * @returns The scalar penalty.
   */
  penalty(params: Params, order = 2, buffers?: Params): Value {
    return differenceRoughness(this.coefficients(params, buffers), order)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in=${this.inFeatures}, out=${this.outFeatures}, degree=${this.degree}, knots=${this.interiorKnots} (${this.placement.learnable ? 'learnable' : this.placement.rule}), shape=${this.shapeTransform.name}`
  }
}
