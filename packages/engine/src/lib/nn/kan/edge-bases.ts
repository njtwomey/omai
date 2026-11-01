/**
 * The function families an edge of a Kolmogorov-Arnold network can be drawn from, as one slot (`kanBases`): the cubic
 * B-spline of the paper (a `SplineLayer`, so its knots can sit at quantiles or train), Gaussian radial basis functions
 * (FastKAN), Chebyshev polynomials of the squashed input (ChebyKAN) and a Fourier series. Every family is a linear
 * combination `sum_k c_{o,i,k} B_k(x_i)` of fixed basis functions of the (standardised) input, so a layer of any of them
 * has the interface of `SplineLayer` (`basis`, `coefficients`, `terms`, `curve`) and the layers above and the training
 * machinery (grid updates, refinement by least squares, pruning, symbolic fitting) do not care which one it is. The
 * three cheaper families keep the input range in a buffer, as the spline does, so a range can be refitted to data.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import { choice, dict, type OptionTable, optionTable, type SpaceValue } from '../../space/index.ts'
import {
  add,
  asTensor,
  cos,
  div,
  einsum,
  exp,
  fromStorage,
  mul,
  reshape,
  sin,
  sliceAxis,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tanh,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { SplineLayer } from '../additive/spline-layer.ts'
import { initKey, Module, type Params } from '../module/module.ts'

/** The interface of a layer of learnable univariate functions `phi_{o,i}(x_i) = sum_k c_{o,i,k} B_k(x_i)`. */
export interface EdgeFunctions extends Module {
  readonly inFeatures: number
  readonly outFeatures: number
  /** The number of coefficients per edge. */
  readonly basisSize: number
  /** The basis of every input `[n, in, K]`. */
  basis(params: Params, x: Value, buffers?: Params): Value
  /** The effective coefficients `[out, in, K]`. */
  coefficients(params: Params, buffers?: Params): Value
  /** The value of every edge `[n, in, out]`. */
  terms(params: Params, x: Value, buffers?: Params): Value
  /** One edge on a grid. */
  curve(params: Params, grid: ArrayLike<number>, feature?: number, output?: number, buffers?: Params): Tensor
}

/** What an edge family needs to be built. */
export interface EdgeBasisOptions {
  readonly inFeatures: number
  readonly outFeatures: number
  /** The grid size `G`: intervals of the spline, centres of the radial basis minus one, harmonics of the Fourier series. */
  readonly grid: number
  /** The spline degree, and the degree of the Chebyshev polynomials. */
  readonly degree: number
  /** The `splineKnots` option (only the spline family can place knots anywhere but evenly). */
  readonly knots: SpaceValue
  /** Beyond the range: hold the boundary value or continue with the tangent (splines). */
  readonly outside: 'clamp' | 'linear'
  readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
  readonly sample?: TensorLike
  /** The standard deviation of the raw coefficients at initialisation. */
  readonly initScale: number
  readonly key?: Key
}

/** A family of edge functions: how many coefficients an edge has and how to build a layer of them. */
export interface EdgeBasis {
  readonly name: string
  /** The number of coefficients per edge for a grid size and a degree. */
  size(grid: number, degree: number): number
  create(options: EdgeBasisOptions): EdgeFunctions
}

/** What an edge family is, for validity rules and selectors. */
export interface EdgeBasisProperties {
  /** Whether the family places and can train knots (only the spline). */
  readonly knotted: boolean
  /** Whether going from one grid size to a larger one contains the smaller function space (so refinement is exact). */
  readonly nested: boolean
}

const family = (
  properties: EdgeBasisProperties,
  description: string,
  space: ReturnType<typeof dict>,
  make: (k: Readonly<Record<string, SpaceValue>>) => EdgeBasis,
): {
  readonly description: string
  readonly space: ReturnType<typeof dict>
  readonly properties: EdgeBasisProperties
  readonly make: (k: Readonly<Record<string, SpaceValue>>) => EdgeBasis
} => ({ description, space, properties, make })

type Expansion = 'rbf' | 'chebyshev' | 'fourier'

const expansionSize = (kind: Expansion, grid: number, degree: number): number =>
  kind === 'rbf' ? grid + 1 : kind === 'chebyshev' ? degree + 1 : 2 * grid + 1

/**
 * A layer of edge functions from a fixed non-spline basis of the standardised input `t = (x - mid) / half` (so `t` runs
 * over `[-1, 1]` on the input range): Gaussian bumps `exp(-((t - c_k) / h)^2)` on `grid + 1` centres (FastKAN),
 * Chebyshev polynomials `T_0 .. T_degree` of `tanh(t)` (ChebyKAN), or the constant and `grid` cosine and sine harmonics
 * of `pi (t + 1) / 2` (a Fourier series, period twice the range). Parameters are `coef [out, in, K]`; the range is a
 * buffer.
 *
 * @concept splines
 * @example
 * ```ts
 * import { ExpansionLayer } from 'mdml-engine/nn'
 * const layer = new ExpansionLayer({ inFeatures: 2, outFeatures: 3, expansion: 'chebyshev', degree: 4 })
 * layer.basisSize // => 5
 * layer.forward([[0.1, -0.2]]).shape // => [1, 3]
 * ```
 */
export class ExpansionLayer extends Module implements EdgeFunctions {
  readonly kind = 'ExpansionLayer'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly expansion: Expansion
  readonly grid: number
  readonly degree: number
  readonly basisSize: number
  private readonly lo: number[]
  private readonly hi: number[]
  private readonly initScale: number

  constructor(options: {
    readonly inFeatures: number
    readonly outFeatures: number
    readonly expansion: Expansion
    readonly grid?: number
    readonly degree?: number
    readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
    readonly sample?: TensorLike
    readonly initScale?: number
    readonly key?: Key
  }) {
    super()
    this.inFeatures = options.inFeatures
    this.outFeatures = options.outFeatures
    this.expansion = options.expansion
    this.grid = options.grid ?? 5
    this.degree = options.degree ?? 3
    if (!(Number.isInteger(this.grid) && this.grid >= 1))
      throw new MdmlError('value', 'ExpansionLayer: grid must be >= 1')
    if (!(Number.isInteger(this.degree) && this.degree >= 0))
      throw new MdmlError('value', 'ExpansionLayer: degree must be an integer >= 0')
    this.basisSize = expansionSize(this.expansion, this.grid, this.degree)
    this.initScale = options.initScale ?? 0.1
    const d = this.inFeatures
    const sample = options.sample === undefined ? undefined : asTensor(options.sample)
    if (sample !== undefined && (sample.shape.length !== 2 || sample.shape[1] !== d))
      throw new MdmlError('shape', `ExpansionLayer: sample must be [n, ${d}]`)
    const values = sample === undefined ? undefined : toFlat(sample)
    const rows = sample === undefined ? 0 : (sample.shape[0] ?? 0)
    this.lo = []
    this.hi = []
    for (let i = 0; i < d; i++) {
      const r = options.range
      const given = r === undefined ? undefined : ((typeof r[0] === 'number' ? r : r[i]) as readonly [number, number])
      let lo = given?.[0] ?? 0
      let hi = given?.[1] ?? 1
      if (given === undefined && values !== undefined && rows > 0) {
        lo = Number.POSITIVE_INFINITY
        hi = Number.NEGATIVE_INFINITY
        for (let r2 = 0; r2 < rows; r2++) {
          const v = values[r2 * d + i] as number
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
      }
      if (!(hi > lo)) [lo, hi] = [lo - 0.5, lo + 0.5]
      this.lo.push(lo)
      this.hi.push(hi)
    }
    this.params = this.initParams(initKey('ExpansionLayer', options.key))
    this.buffers = this.initBuffers()
  }

  override initParams(key: Key): Params {
    const shape = [this.outFeatures, this.inFeatures, this.basisSize]
    return {
      coef: fromStorage(
        Float64Array.from(normal(foldIn(key, 0), shape, { scale: this.initScale }).data),
        shape,
        'float64',
      ),
    }
  }

  override initBuffers(): Params {
    return {
      range: fromStorage(
        Float64Array.from(this.lo.flatMap((lo, i) => [lo, this.hi[i] as number])),
        [this.inFeatures, 2],
        'float64',
      ),
    }
  }

  /** The inputs as `[n, in]`. */
  private inputs(x: Value): Value {
    if (x.shape.length === 1 && this.inFeatures === 1) return reshape(x, [x.shape[0] ?? 0, 1])
    if (x.shape.length !== 2 || x.shape[1] !== this.inFeatures)
      throw new MdmlError('shape', `ExpansionLayer: expected [n, ${this.inFeatures}], got [${x.shape.join(', ')}]`)
    return x
  }

  /** The standardised input `t` `[n, in]`: `-1` at the lower end of the range and `1` at the upper. */
  private standardise(x: Value, buffers: Params | undefined): Value {
    const range = (buffers ?? this.buffers)['range']
    const r = range === undefined ? undefined : toFlat(range as Value)
    const mid = this.lo.map((lo, i) =>
      r === undefined ? (lo + (this.hi[i] as number)) / 2 : ((r[2 * i] as number) + (r[2 * i + 1] as number)) / 2,
    )
    const half = this.lo.map((lo, i) =>
      r === undefined ? ((this.hi[i] as number) - lo) / 2 : ((r[2 * i + 1] as number) - (r[2 * i] as number)) / 2,
    )
    return div(sub(x, asTensor(mid)), asTensor(half))
  }

  basis(_params: Params, x: Value, buffers?: Params): Value {
    const t = this.standardise(this.inputs(x), buffers)
    const k = this.basisSize
    if (this.expansion === 'rbf') {
      const centres = asTensor(Array.from({ length: k }, (_, j) => -1 + (2 * j) / this.grid))
      const h = 2 / this.grid
      return exp(mul(square(div(sub(reshape(t, [...t.shape, 1]), centres), h)), -1))
    }
    if (this.expansion === 'chebyshev') {
      const u = tanh(t)
      const terms: Value[] = [add(mul(u, 0), 1)]
      if (k > 1) terms.push(u)
      for (let j = 2; j < k; j++) terms.push(sub(mul(mul(u, terms[j - 1] as Value), 2), terms[j - 2] as Value))
      return stack(terms, 2)
    }
    const theta = mul(add(t, 1), Math.PI / 2)
    const terms: Value[] = [add(mul(theta, 0), 1)]
    for (let h = 1; h <= this.grid; h++) terms.push(cos(mul(theta, h)), sin(mul(theta, h)))
    return stack(terms, 2)
  }

  coefficients(params: Params): Value {
    return params['coef'] as Value
  }

  terms(params: Params, x: Value, buffers?: Params): Value {
    return einsum('nik,oik->nio', this.basis(params, x, buffers), this.coefficients(params))
  }

  apply(params: Params, x: Value, ctx: { readonly buffers?: Params } = {}): Value {
    return einsum('nik,oik->no', this.basis(params, x, ctx.buffers), this.coefficients(params))
  }

  curve(params: Params, grid: ArrayLike<number>, feature = 0, output = 0, buffers?: Params): Tensor {
    const g = Array.from(grid)
    const x = fromStorage(
      Float64Array.from(g.flatMap((v) => Array.from({ length: this.inFeatures }, () => v))),
      [g.length, this.inFeatures],
      'float64',
    )
    const b = sliceAxis(this.basis(params, x, buffers), 1, feature, feature + 1)
    const c = sliceAxis(sliceAxis(this.coefficients(params), 0, output, output + 1), 1, feature, feature + 1)
    return asTensor(
      toFlat(reshape(sum(mul(reshape(b, [g.length, this.basisSize]), reshape(c, [1, this.basisSize])), 1), [g.length])),
    ) as Tensor
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in=${this.inFeatures}, out=${this.outFeatures}, ${this.expansion}, size=${this.basisSize}`
  }
}

const noKnots = (who: string, knots: SpaceValue): void => {
  const kind =
    typeof knots === 'object' && knots !== null && !Array.isArray(knots)
      ? (knots as { kind?: unknown }).kind
      : undefined
  if (kind !== undefined && kind !== 'uniform')
    throw new MdmlError('value', `${who}: only the spline family places knots (got ${String(kind)})`)
}

const expansionFamily = (kind: Expansion): EdgeBasis => ({
  name: kind,
  size: (grid, degree) => expansionSize(kind, grid, degree),
  create: (o) => {
    noKnots(kind, o.knots)
    return new ExpansionLayer({
      inFeatures: o.inFeatures,
      outFeatures: o.outFeatures,
      expansion: kind,
      grid: o.grid,
      degree: o.degree,
      ...(o.range === undefined ? {} : { range: o.range }),
      ...(o.sample === undefined ? {} : { sample: o.sample }),
      initScale: o.initScale,
      ...(o.key === undefined ? {} : { key: o.key }),
    })
  },
})

/**
 * The function family of a Kolmogorov-Arnold edge as one slot: `bspline` (the paper: `grid + degree` coefficients, the
 * knots a slot of their own), `rbf` (Gaussian radial basis functions, FastKAN: `grid + 1` centres), `chebyshev`
 * (Chebyshev polynomials of `tanh` of the standardised input, ChebyKAN: `degree + 1` terms) and `fourier` (a constant
 * and `grid` cosine and sine harmonics).
 *
 * @example
 * ```ts
 * import { kanBases } from 'mdml-engine/nn'
 * kanBases.make({ kind: 'bspline' }).size(5, 3) // => 8
 * kanBases.make({ kind: 'fourier' }).size(4, 3) // => 9
 * kanBases.properties({ kind: 'rbf' }).knotted // => false
 * ```
 */
export const kanBases: OptionTable<EdgeBasis, EdgeBasisProperties> = optionTable({
  bspline: family(
    { knotted: true, nested: true },
    'B-splines on a grid of knots (the paper); the knots are a slot and can be trained',
    dict({ boundary: choice(['clamped', 'extended'], { default: 'extended', label: 'knot vector ends' }) }),
    (k) => ({
      name: 'bspline',
      size: (grid, degree) => grid + degree,
      create: (o) =>
        new SplineLayer({
          inFeatures: o.inFeatures,
          outFeatures: o.outFeatures,
          interiorKnots: o.grid - 1,
          degree: o.degree,
          knots: o.knots,
          bias: false,
          outside: o.outside,
          boundary: (k['boundary'] ?? 'extended') as 'clamped' | 'extended',
          initScale: o.initScale,
          ...(o.range === undefined ? {} : { range: o.range }),
          ...(o.sample === undefined ? {} : { sample: o.sample }),
          ...(o.key === undefined ? {} : { key: o.key }),
        }),
    }),
  ),
  rbf: family(
    { knotted: false, nested: false },
    'Gaussian radial basis functions on evenly spaced centres (FastKAN)',
    dict({}),
    () => expansionFamily('rbf'),
  ),
  chebyshev: family(
    { knotted: false, nested: true },
    'Chebyshev polynomials of the squashed standardised input (ChebyKAN)',
    dict({}),
    () => expansionFamily('chebyshev'),
  ),
  fourier: family(
    { knotted: false, nested: true },
    'a constant and cosine and sine harmonics of the standardised input (Fourier series)',
    dict({}),
    () => expansionFamily('fourier'),
  ),
})
