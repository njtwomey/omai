/**
 * The terms of an additive network, in the manner of `gam/terms.ts`: a list of specifications (`splineTerm`, `netTerm`,
 * `linearTerm`, `interactionTerm`, `subnetTerm`) that the `additiveTerms` table turns into modules. Each term maps the
 * columns of its features, `[n, k]`, to one contribution `[n, 1]`: a learnable spline of one feature, a small network of
 * one feature (a neural additive model's feature net, ReLU or ExU units, with dropout and an output penalty), a linear
 * term, a tensor-product spline of two features, or a network over any subset (GA2M and GAMI-Net interactions,
 * arbitrary sub-networks). A built term also says how it is regularised: the P-spline smoothness penalty of splines,
 * the output penalty and weight decay of networks.
 */

import { MdmlError } from '../../base/index.ts'
import { dgesv } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  choice,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../../space/index.ts'
import { differencePenalty } from '../../stats/smoothing/index.ts'
import {
  add,
  asTensor,
  clip,
  einsum,
  exp,
  fromStorage,
  matmul,
  mean,
  mul,
  reshape,
  sliceAxis,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  transpose,
  treePaths,
  type Value,
} from '../../tensor/index.ts'
import type { ActivationName } from '../functional/activations.ts'
import { activationModule } from '../module/activations.ts'
import { Sequential } from '../module/containers.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, Module, type Params } from '../module/module.ts'
import { Dropout } from '../module/regularisation.ts'
import { bsplineBasis } from './bspline.ts'
import { splineKnots, splineShapes } from './knots.ts'
import { differenceRoughness, SplineLayer } from './spline-layer.ts'

/**
 * Exp-centred units (Agarwal et al. 2021): `h = relu_n((x - b) e^W)`, a layer whose units can switch on and off within
 * a very short interval of the input, so a feature net can fit jagged shape functions with few units. The activation is
 * the ReLU clipped at `n` (default 1). Parameters are `weight [in, out]` (initialised around `initMean`, 3 by default)
 * and the input offsets `bias [in]`.
 *
 * @concept additive-models
 * @example
 * ```ts
 * import { ExULayer } from 'mdml-engine/nn'
 * new ExULayer(1, 16).forward([[0.2], [0.8]]).shape // => [2, 16]
 * ```
 */
export class ExULayer extends Module {
  readonly kind = 'ExU'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly clipAt: number
  readonly initMean: number

  constructor(
    inFeatures: number,
    outFeatures: number,
    options: { readonly clipAt?: number; readonly initMean?: number; readonly key?: Key } = {},
  ) {
    super()
    this.inFeatures = inFeatures
    this.outFeatures = outFeatures
    this.clipAt = options.clipAt ?? 1
    this.initMean = options.initMean ?? 3
    this.params = this.initParams(initKey('ExU', options.key))
  }

  override initParams(key: Key): Params {
    return {
      weight: normal(foldIn(key, 0), [this.inFeatures, this.outFeatures], { loc: this.initMean, scale: 0.5 }),
      bias: normal(foldIn(key, 1), [this.inFeatures], { scale: 0.5 }),
    }
  }

  apply(params: Params, x: Value): Value {
    const centred = sub(x, params['bias'] as Value)
    return clip(matmul(centred, exp(params['weight'] as Value)), 0, this.clipAt)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outFeatures]
  }
}

/**
 * A tensor-product spline of two features: `f(x, z) = sum_{a,b} C_{ab} B_a(x) B_b(z)`, with the knots of each margin a
 * slot as in {@link SplineLayer} (fixed or free) and a P-spline penalty on the rows and columns of `C`. Parameters are
 * `coef [K, K]` and the margins' `knotLogits`; the margins' ranges and fixed knots are buffers.
 *
 * @concept additive-models
 * @example
 * ```ts
 * import { TensorSpline2d } from 'mdml-engine/nn'
 * new TensorSpline2d({ interiorKnots: 3 }).forward([[0.2, 0.9], [0.5, 0.5]]).shape // => [2, 1]
 * ```
 */
export class TensorSpline2d extends Module {
  readonly kind = 'TensorSpline2d'
  readonly margins: SplineLayer

  constructor(options: {
    readonly interiorKnots?: number
    readonly degree?: number
    readonly knots?: SpaceValue
    readonly range?: readonly [number, number] | readonly (readonly [number, number])[]
    readonly sample?: TensorLike
    readonly initScale?: number
    readonly key?: Key
  }) {
    super()
    this.margins = new SplineLayer({
      inFeatures: 2,
      outFeatures: 1,
      bias: false,
      ...(options.interiorKnots === undefined ? {} : { interiorKnots: options.interiorKnots }),
      ...(options.degree === undefined ? {} : { degree: options.degree }),
      ...(options.knots === undefined ? {} : { knots: options.knots }),
      ...(options.range === undefined ? {} : { range: options.range }),
      ...(options.sample === undefined ? {} : { sample: options.sample }),
      ...(options.key === undefined ? {} : { key: options.key }),
    })
    this.initScale = options.initScale ?? 0.1
    this.params = this.initParams(initKey('TensorSpline2d', options.key))
    this.buffers = this.initBuffers()
  }

  private readonly initScale: number

  override initParams(key: Key): Params {
    const k = this.margins.basisSize
    const own = this.margins.params
    return {
      coef: normal(foldIn(key, 0), [k, k], { scale: this.initScale }),
      ...(own['knotLogits'] === undefined ? {} : { knotLogits: own['knotLogits'] }),
    }
  }

  override initBuffers(): Params {
    return this.margins.initBuffers()
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const basis = this.margins.basis({ knotLogits: params['knotLogits'] }, x, ctx.buffers ?? this.buffers)
    const f = einsum('na,nb,ab->n', margin(basis, 0), margin(basis, 1), params['coef'] as Value)
    return reshape(f, [f.shape[0] ?? 0, 1])
  }

  /**
   * The surface on a grid: row `a` is `x = gridX[a]`, column `b` is `z = gridZ[b]`, without any centring.
   *
   * @param params - The parameters.
   * @param gridX - Values of the first feature.
   * @param gridZ - Values of the second feature.
   * @param buffers - The buffers (default the module's).
   * @returns The values `[gridX.length, gridZ.length]`.
   */
  surface(params: Params, gridX: ArrayLike<number>, gridZ: ArrayLike<number>, buffers?: Params): Value {
    const b = buffers ?? this.buffers
    const knots = this.margins.knotVectors({ knotLogits: params['knotLogits'] }, b)
    const rows = (j: number, grid: ArrayLike<number>): Value =>
      bsplineBasis(Array.from(grid), reshape(sliceAxis(knots, 0, j, j + 1), [-1]), this.margins.degree, {
        outside: this.margins.outside,
        boundary: this.margins.boundary,
      })
    return einsum('ga,ab,hb->gh', rows(0, gridX), params['coef'] as Value, rows(1, gridZ))
  }

  /**
   * The P-spline penalty: squared `order`th differences along both axes of the coefficient matrix.
   *
   * @param params - The parameters.
   * @param order - The difference order (default 2).
   * @returns The scalar.
   */
  penalty(params: Params, order = 2): Value {
    const c = params['coef'] as Value
    return add(differenceRoughness(c, order), differenceRoughness(transpose(c), order))
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), 1]
  }
}

/** The basis of one margin, `[n, K]`, from a `[n, 2, K]` basis. */
const margin = (basis: Value, j: number): Value =>
  reshape(sliceAxis(basis, 1, j, j + 1), [basis.shape[0] ?? 0, basis.shape[2] ?? 0])

/** What a built term is for the model around it: its module, its features and label, and how it is regularised. */
export interface BuiltTerm {
  /** Maps the term's feature columns `[n, k]` to its contribution `[n, 1]`. */
  readonly module: Module
  readonly kind: string
  readonly features: readonly number[]
  readonly label: string
  /** The share of rows whose whole contribution is dropped in training (a neural additive model's feature dropout). */
  readonly featureDropout: number
  /** Smoothness penalty in the `n * loss` convention of `gam` (a penalised likelihood), or undefined for none. */
  smoothness(params: Params, buffers: Params): Value | undefined
  /** Penalties in the mean-loss convention (output penalty, weight decay) on a reference sample `[r, k]`. */
  regularisation(params: Params, buffers: Params, rows: Value): Value | undefined
  /** The effective degrees of freedom of a penalised term for unit weights, on rows `[n, k]`, or undefined. */
  edf(params: Params, buffers: Params, rows: Value): number | undefined
}

/** What the model hands a builder: the data's sample, the term's key and the penalty to use where a term gives none. */
export interface TermContext {
  /** The whole feature matrix `[n, d]` (for quantile knots and ranges), when there is one. */
  readonly sample?: Value
  readonly key: Key
  /** Smoothness `lambda` for spline terms that give none (default 0). */
  readonly lambda: number
}

/** A term builder: the term's features, label, context and knobs to the built term. */
export type TermBuilder = (
  features: readonly number[],
  label: string,
  context: TermContext,
  knobs: Readonly<Record<string, SpaceValue>>,
) => BuiltTerm

/** What a kind of term is, for validity rules and selectors. */
export interface TermProperties {
  /** How many features it takes. */
  readonly arity: 'one' | 'two' | 'any'
  /** Whether it is a neural network (as against a spline or a linear map). */
  readonly neural: boolean
  /** Whether it has a smoothness penalty. */
  readonly penalised: boolean
}

const columnsOf = (sample: Value | undefined, features: readonly number[]): Value | undefined => {
  if (sample === undefined) return undefined
  const d = sample.shape[1] ?? 0
  const rows = sample.shape[0] ?? 0
  const flat = toFlat(sample)
  const k = features.length
  return fromStorage(
    Float64Array.from(
      { length: rows * k },
      (_, i) => flat[Math.floor(i / k) * d + (features[i % k] as number)] as number,
    ),
    [rows, k],
    'float64',
  )
}

const num = (knobs: Readonly<Record<string, SpaceValue>>, name: string, fallback: number): number => {
  const v = knobs[name]
  return typeof v === 'number' ? v : fallback
}

/** Penalised least squares degrees of freedom `tr((B'B + lambda S)^-1 B'B)` of a basis `[n, K]` and a penalty `[K, K]`. */
function penalisedEdf(basis: Float64Array, n: number, k: number, penalty: Float64Array, lambda: number): number {
  const gram = new Float64Array(k * k)
  for (let r = 0; r < n; r++)
    for (let a = 0; a < k; a++) {
      const ba = basis[r * k + a] as number
      if (ba !== 0)
        for (let b = 0; b < k; b++) gram[a * k + b] = (gram[a * k + b] as number) + ba * (basis[r * k + b] as number)
    }
  const system = Float64Array.from(gram, (v, i) => v + lambda * (penalty[i] as number) + (i % (k + 1) === 0 ? 1e-9 : 0))
  let total = 0
  for (let c = 0; c < k; c++) {
    const col = dgesv(
      system,
      Float64Array.from({ length: k }, (_, r) => gram[r * k + c] as number),
      k,
    )
    total += col[c] as number
  }
  return total
}

const NONE_SMOOTHNESS: BuiltTerm['smoothness'] = () => undefined
const NONE_REGULARISATION: BuiltTerm['regularisation'] = () => undefined
const NONE_EDF: BuiltTerm['edf'] = () => undefined

const kind = (
  properties: TermProperties,
  description: string,
  space: Option<TermBuilder, TermProperties>['space'],
  make: (knobs: Readonly<Record<string, SpaceValue>>) => TermBuilder,
): Option<TermBuilder, TermProperties> => ({ description, space, properties, make })

/** A feed-forward net over `k` inputs to one output, with ReLU/ExU/other units and dropout between the layers. */
function featureNet(
  k: number,
  width: number,
  depth: number,
  unit: string,
  dropout: number,
  initMean: number,
  key: Key,
): Sequential {
  const layers: Module[] = []
  const act = (): Module => activationModule((unit === 'exu' ? 'relu' : unit) as ActivationName)
  let into = k
  for (let l = 0; l < depth; l++) {
    const k0 = foldIn(key, l)
    if (l === 0 && unit === 'exu') layers.push(new ExULayer(into, width, { initMean, key: k0 }))
    else {
      layers.push(new Linear(into, width, { key: k0 }), act())
    }
    if (dropout > 0) layers.push(new Dropout(dropout))
    into = width
  }
  layers.push(new Linear(into, 1, { key: foldIn(key, depth) }))
  return new Sequential(...layers)
}

/**
 * The kinds of term of an additive network as one slot: `spline` (a learnable spline of one feature, with the knots and
 * shape slots), `net` (a feed-forward network over one or several features: a neural additive model's feature net with
 * ReLU or ExU units, a pairwise interaction net, any sub-network), `linear` and `tensorSpline` (a tensor-product spline
 * of two features, with the P-spline penalty along both axes).
 *
 * @example
 * ```ts
 * import { additiveTerms } from 'mdml-engine/nn'
 * additiveTerms.names // => ['spline', 'net', 'linear', 'tensorSpline']
 * additiveTerms.properties({ kind: 'net' }).neural // => true
 * ```
 */
export const additiveTerms: OptionTable<TermBuilder, TermProperties> = optionTable({
  spline: kind(
    { arity: 'one', neural: false, penalised: true },
    'a learnable B-spline of one feature (free or fixed knots, optional shape constraint, P-spline penalty)',
    dict({
      interiorKnots: int(1, 64, { default: 8, label: 'interior knots' }),
      degree: int(1, 5, { default: 3, label: 'degree' }),
      knots: splineKnots.space,
      shape: splineShapes.space,
      boundary: choice(['clamped', 'extended'], { default: 'clamped', label: 'boundary knots' }),
      lambda: real(0, 1e8, { default: 0, label: 'smoothness λ' }),
      order: int(0, 4, { default: 2, label: 'difference order' }),
    }),
    (k) => (features, label, ctx) => {
      if (features.length !== 1) throw new MdmlError('value', `spline term ${label}: takes one feature`)
      const sample = columnsOf(ctx.sample, features)
      const layer = new SplineLayer({
        inFeatures: 1,
        outFeatures: 1,
        bias: false,
        interiorKnots: num(k, 'interiorKnots', 8),
        degree: num(k, 'degree', 3),
        key: ctx.key,
        ...(k['knots'] === undefined ? {} : { knots: k['knots'] }),
        ...(k['shape'] === undefined ? {} : { shape: k['shape'] }),
        ...(k['boundary'] === 'extended' ? { boundary: 'extended' as const } : {}),
        ...(sample === undefined ? {} : { sample }),
      })
      const lambda = num(k, 'lambda', ctx.lambda)
      const order = num(k, 'order', 2)
      return {
        module: layer,
        kind: 'spline',
        features,
        label,
        featureDropout: 0,
        smoothness: (params, buffers) => (lambda > 0 ? mul(layer.penalty(params, order, buffers), lambda) : undefined),
        regularisation: NONE_REGULARISATION,
        edf: (params, buffers, rows) => {
          if (layer.shapeTransform.name !== 'none') return undefined
          const b = layer.basis(params, rows, buffers)
          const kk = layer.basisSize
          return penalisedEdf(
            Float64Array.from(toFlat(b)),
            rows.shape[0] ?? 0,
            kk,
            Float64Array.from(toFlat(differencePenalty(kk, order))),
            lambda,
          )
        },
      }
    },
  ),
  net: kind(
    { arity: 'any', neural: true, penalised: false },
    'a feed-forward net of one or several features (feature net, interaction net, sub-network), ReLU or ExU units',
    dict({
      width: int(1, 512, { default: 16, label: 'width' }),
      depth: int(1, 6, { default: 2, label: 'hidden layers' }),
      unit: choice(['relu', 'exu', 'gelu', 'silu', 'tanh'], { default: 'relu', label: 'hidden unit' }),
      dropout: real(0, 0.9, { default: 0, label: 'dropout' }),
      featureDropout: real(0, 0.9, { default: 0, label: 'feature dropout' }),
      outputPenalty: real(0, 100, { default: 0, label: 'output penalty' }),
      weightDecay: real(0, 10, { default: 0, label: 'weight decay' }),
      initMean: real(-2, 8, { default: 3, label: 'ExU weight mean' }),
    }),
    (k) => (features, label, ctx) => {
      const net = featureNet(
        features.length,
        num(k, 'width', 16),
        num(k, 'depth', 2),
        typeof k['unit'] === 'string' ? k['unit'] : 'relu',
        num(k, 'dropout', 0),
        num(k, 'initMean', 3),
        ctx.key,
      )
      const outputPenalty = num(k, 'outputPenalty', 0)
      const weightDecay = num(k, 'weightDecay', 0)
      return {
        module: net,
        kind: 'net',
        features,
        label,
        featureDropout: num(k, 'featureDropout', 0),
        smoothness: NONE_SMOOTHNESS,
        regularisation:
          outputPenalty === 0 && weightDecay === 0
            ? NONE_REGULARISATION
            : (params, buffers, rows) => {
                let total: Value = asTensor(0)
                if (outputPenalty > 0)
                  total = add(
                    total,
                    mul(mean(square(net.apply(params, rows, { train: false, buffers }))), outputPenalty),
                  )
                if (weightDecay > 0) {
                  let sq: Value = asTensor(0)
                  for (const [name, leaf] of Object.entries(treePaths(params)))
                    if (name.endsWith('weight')) sq = add(sq, sum(square(leaf)))
                  total = add(total, mul(sq, weightDecay))
                }
                return total
              },
        edf: NONE_EDF,
      }
    },
  ),
  linear: kind(
    { arity: 'any', neural: false, penalised: false },
    'a linear term in one or several features',
    dict({}),
    () => (features, label, ctx) => ({
      module: new Linear(features.length, 1, { bias: false, key: ctx.key }),
      kind: 'linear',
      features,
      label,
      featureDropout: 0,
      smoothness: NONE_SMOOTHNESS,
      regularisation: NONE_REGULARISATION,
      edf: () => features.length,
    }),
  ),
  tensorSpline: kind(
    { arity: 'two', neural: false, penalised: true },
    'a tensor-product spline of two features (free or fixed knots on each margin, P-spline penalty on both axes)',
    dict({
      interiorKnots: int(1, 32, { default: 5, label: 'interior knots per margin' }),
      degree: int(1, 5, { default: 3, label: 'degree' }),
      knots: splineKnots.space,
      lambda: real(0, 1e8, { default: 0, label: 'smoothness λ' }),
      order: int(0, 4, { default: 2, label: 'difference order' }),
    }),
    (k) => (features, label, ctx) => {
      if (features.length !== 2) throw new MdmlError('value', `tensorSpline term ${label}: takes two features`)
      const sample = columnsOf(ctx.sample, features)
      const surface = new TensorSpline2d({
        interiorKnots: num(k, 'interiorKnots', 5),
        degree: num(k, 'degree', 3),
        key: ctx.key,
        ...(k['knots'] === undefined ? {} : { knots: k['knots'] }),
        ...(sample === undefined ? {} : { sample }),
      })
      const lambda = num(k, 'lambda', ctx.lambda)
      const order = num(k, 'order', 2)
      return {
        module: surface,
        kind: 'tensorSpline',
        features,
        label,
        featureDropout: 0,
        smoothness: (params) => (lambda > 0 ? mul(surface.penalty(params, order), lambda) : undefined),
        regularisation: NONE_REGULARISATION,
        edf: NONE_EDF,
      }
    },
  ),
})

/** A term of the list: the columns it reads, an optional label and its configuration in the `additiveTerms` table. */
export interface TermSpec {
  readonly features: readonly number[]
  readonly label?: string
  /** The `additiveTerms` option: `{ kind: 'spline', interiorKnots: 10, ... }`. */
  readonly config: SpaceValue
}

/** Options of {@link splineTerm}. */
export interface SplineTermOptions {
  readonly interiorKnots?: number
  readonly degree?: number
  /** The `splineKnots` option (`{ kind: 'learnable' }` trains the knots). */
  readonly knots?: SpaceValue
  /** The `splineShapes` option (`{ kind: 'increasing' }`). */
  readonly shape?: SpaceValue
  /** `clamped` (default) or `extended` (P-spline knots beyond the range, the basis of `gam`'s `ps`). */
  readonly boundary?: 'clamped' | 'extended'
  /** The smoothness `lambda` (in the `n * loss` convention of `gam`); default the model's. */
  readonly lambda?: number
  /** The difference order of the penalty (default 2). */
  readonly order?: number
  readonly label?: string
}

const defined = (o: Record<string, SpaceValue | undefined>): Record<string, SpaceValue> =>
  Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Record<string, SpaceValue>

/**
 * A learnable spline of one feature, the `s()` of an additive network.
 *
 * @param feature - The column.
 * @param options - Knots (count, placement, learnable), degree, shape constraint, smoothness and label.
 * @returns The term.
 * @concept additive-models
 * @example
 * ```ts
 * import { splineTerm } from 'mdml-engine/nn'
 * splineTerm(0, { interiorKnots: 6, knots: { kind: 'learnable' } }).features // => [0]
 * ```
 */
export function splineTerm(feature: number, options: SplineTermOptions = {}): TermSpec {
  const { label, ...knobs } = options
  return {
    features: [feature],
    ...(label === undefined ? {} : { label }),
    config: { kind: 'spline', ...defined(knobs as Record<string, SpaceValue | undefined>) },
  }
}

/** Options of {@link netTerm}, {@link subnetTerm} and the net form of {@link interactionTerm}. */
export interface NetTermOptions {
  readonly width?: number
  /** The number of hidden layers (default 2). */
  readonly depth?: number
  /** `relu` (default), `exu` (Agarwal et al.'s exp-centred units in the first layer), `gelu`, `silu` or `tanh`. */
  readonly unit?: 'relu' | 'exu' | 'gelu' | 'silu' | 'tanh'
  readonly dropout?: number
  /** The share of rows whose whole contribution is dropped in training. */
  readonly featureDropout?: number
  /** The weight of the output penalty, the mean squared contribution (NAM). */
  readonly outputPenalty?: number
  readonly weightDecay?: number
  readonly label?: string
}

/**
 * A neural additive model's feature net: a small network of one feature.
 *
 * @param feature - The column.
 * @param options - Width, depth, unit (`relu` or `exu`), dropout, feature dropout and output penalty.
 * @returns The term.
 * @concept additive-models
 * @example
 * ```ts
 * import { netTerm } from 'mdml-engine/nn'
 * netTerm(2, { unit: 'exu', featureDropout: 0.1 }).config // => { kind: 'net', unit: 'exu', featureDropout: 0.1 }
 * ```
 */
export function netTerm(feature: number, options: NetTermOptions = {}): TermSpec {
  return subnetTerm([feature], options)
}

/**
 * An arbitrary sub-network over a subset of the features (a hidden interaction of any order).
 *
 * @param features - The columns.
 * @param options - As {@link netTerm}.
 * @returns The term.
 * @concept additive-models
 * @example
 * ```ts
 * import { subnetTerm } from 'mdml-engine/nn'
 * subnetTerm([0, 1, 2], { width: 8 }).features // => [0, 1, 2]
 * ```
 */
export function subnetTerm(features: readonly number[], options: NetTermOptions = {}): TermSpec {
  const { label, ...knobs } = options
  return {
    features: [...features],
    ...(label === undefined ? {} : { label }),
    config: { kind: 'net', ...defined(knobs as Record<string, SpaceValue | undefined>) },
  }
}

/**
 * A linear term in one or several features.
 *
 * @param features - The column, or columns.
 * @param label - The label.
 * @returns The term.
 * @concept additive-models
 * @example
 * ```ts
 * import { linearTerm } from 'mdml-engine/nn'
 * linearTerm(1).config // => { kind: 'linear' }
 * ```
 */
export function linearTerm(features: number | readonly number[], label?: string): TermSpec {
  return {
    features: typeof features === 'number' ? [features] : [...features],
    ...(label === undefined ? {} : { label }),
    config: { kind: 'linear' },
  }
}

/** Options of {@link interactionTerm}. */
export interface InteractionTermOptions extends SplineTermOptions, NetTermOptions {
  /** `spline` (a tensor-product spline, default) or `net` (a small network of the pair, GAMI-Net style). */
  readonly form?: 'spline' | 'net'
}

/**
 * A pairwise interaction term (GA2M, GAMI-Net): a tensor-product spline or a small network of two features.
 *
 * @param pair - The two columns.
 * @param options - The form and its knobs.
 * @returns The term.
 * @concept additive-models
 * @example
 * ```ts
 * import { interactionTerm } from 'mdml-engine/nn'
 * interactionTerm([0, 1]).config // => { kind: 'tensorSpline' }
 * interactionTerm([0, 1], { form: 'net', width: 8 }).config // => { kind: 'net', width: 8 }
 * ```
 */
export function interactionTerm(pair: readonly [number, number], options: InteractionTermOptions = {}): TermSpec {
  const { form, ...rest } = options
  if (form === 'net') return subnetTerm(pair, rest as NetTermOptions)
  const { label, ...knobs } = rest as SplineTermOptions
  return {
    features: [...pair],
    ...(label === undefined ? {} : { label }),
    config: { kind: 'tensorSpline', ...defined(knobs as Record<string, SpaceValue | undefined>) },
  }
}

/**
 * The default label of a term: `s(x0)`, `nn(x0)`, `x0`, `te(x0,x1)`.
 *
 * @param spec - The term.
 * @param names - Feature names (default `x0`, `x1`, ...).
 * @returns The label.
 * @concept additive-models
 * @example
 * ```ts
 * import { splineTerm, termLabel } from 'mdml-engine/nn'
 * termLabel(splineTerm(1), ['age', 'bmi']) // => 's(bmi)'
 * ```
 */
export function termLabel(spec: TermSpec, names?: readonly string[]): string {
  if (spec.label !== undefined) return spec.label
  const f = spec.features.map((i) => names?.[i] ?? `x${i}`).join(',')
  const k = (spec.config as { readonly kind?: string }).kind
  return k === 'spline' ? `s(${f})` : k === 'net' ? `nn(${f})` : k === 'tensorSpline' ? `te(${f})` : f
}
