/**
 * `AdditiveModel`: an intercept plus a sum of terms, through a link (generalised additive models in situ in a network).
 * The terms are the learnable splines, feature nets, linear maps, tensor-product splines and sub-networks of
 * `terms.ts`, chosen from the `additiveTerms` table by their configurations. Each term can be centred over a
 * reference sample (the sum-to-zero constraint of `gam`, so the intercept is identifiable and each shape function is
 * read as an effect), interactions can be required to have their main effects (heredity), and the model reports its
 * penalties, its term values and curves, and its effective degrees of freedom, which is what a trainer logs.
 */

import { MdmlError } from '../../base/index.ts'
import { type Link, type LinkName, linkByName } from '../../dists/families/index.ts'
import { dgesv } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import type { SpaceValue } from '../../space/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  fromStorage,
  mean,
  mul,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { Composite } from '../module/containers.ts'
import { type ApplyContext, initKey, isTraining, type Params } from '../module/module.ts'
import { additiveTerms, type BuiltTerm, type TermSpec, termLabel } from './terms.ts'

/** How an interaction relates to the main effects of its features. */
export type Heredity = 'none' | 'weak' | 'strong'

/** Options of {@link AdditiveModel}. */
export interface AdditiveModelOptions {
  /** The number of feature columns of the input. */
  readonly inFeatures: number
  /** The terms, in order. */
  readonly terms: readonly TermSpec[]
  /** A sample of the inputs `[n, inFeatures]`: ranges, quantile knots and the centring rows. */
  readonly sample?: TensorLike
  /** The link from the linear predictor to the mean (default identity); `mean()` applies its inverse. */
  readonly link?: LinkName | Link
  /** The starting intercept, or `false` for none (default 0). */
  readonly intercept?: number | false
  /** Centre every term over the reference rows, the sum-to-zero constraint (default true). */
  readonly centre?: boolean
  /** The reference rows kept for centring and for the output penalty (default 256, spread over the sample). */
  readonly centreRows?: number
  /** Whether an interaction needs its features' main effects: `strong` all of them, `weak` one (default `none`). */
  readonly heredity?: Heredity
  /** The smoothness `lambda` of spline terms that give none (default 0). */
  readonly lambda?: number
  readonly featureNames?: readonly string[]
  readonly key?: Key
}

const linkOf = (link: LinkName | Link | undefined): Link =>
  typeof link === 'object' ? link : linkByName(link ?? 'identity')

/**
 * An intercept plus a sum of terms through a link, each term a table option (`additiveTerms`). The parameters are the
 * terms' under `t0`, `t1`, ... and the `intercept`; the buffers hold the terms' fixed knots and ranges and the
 * reference rows. `apply` returns the linear predictor `[n, 1]`; {@link AdditiveModel.mean} applies the inverse link.
 *
 * @concept additive-models
 * @example
 * ```ts
 * import { AdditiveModel, linearTerm, splineTerm } from 'mdml-engine/nn'
 * const model = new AdditiveModel({ inFeatures: 2, terms: [splineTerm(0, { interiorKnots: 4 }), linearTerm(1)] })
 * model.forward([[0.2, 1], [0.7, -1]]).shape // => [2, 1]
 * model.labels // => ['s(x0)', 'x1']
 * ```
 */
export class AdditiveModel extends Composite {
  readonly kind = 'AdditiveModel'
  readonly inFeatures: number
  readonly built: readonly BuiltTerm[]
  readonly specs: readonly TermSpec[]
  readonly link: Link
  readonly centre: boolean
  readonly hasIntercept: boolean
  private readonly interceptStart: number
  private readonly reference: Value

  constructor(options: AdditiveModelOptions) {
    const d = options.inFeatures
    const sample = options.sample === undefined ? undefined : (asTensor(options.sample) as Value)
    if (sample !== undefined && (sample.shape.length !== 2 || sample.shape[1] !== d))
      throw new MdmlError('shape', `AdditiveModel: sample must be [n, ${d}]`)
    const base = initKey('AdditiveModel', options.key)
    const names = options.featureNames
    const built = options.terms.map((spec, i) => {
      for (const f of spec.features)
        if (!(Number.isInteger(f) && f >= 0 && f < d))
          throw new MdmlError('value', `AdditiveModel: term ${i} uses column ${f}, of ${d}`)
      const label = termLabel(spec, names)
      return additiveTerms.make(spec.config)(
        spec.features,
        label,
        { key: foldIn(base, i), lambda: options.lambda ?? 0, ...(sample === undefined ? {} : { sample }) },
        knobsOf(spec.config),
      )
    })
    checkHeredity(built, options.heredity ?? 'none')
    super(Object.fromEntries(built.map((t, i) => [`t${i}`, t.module])))
    this.inFeatures = d
    this.built = built
    this.specs = options.terms
    this.link = linkOf(options.link)
    this.centre = options.centre ?? true
    this.hasIntercept = options.intercept !== false
    this.interceptStart = typeof options.intercept === 'number' ? options.intercept : 0
    this.reference = referenceRows(sample, d, options.centreRows ?? 256)
    this.params = this.initParams(base)
    this.buffers = this.initBuffers()
  }

  /** The term labels, in order. */
  get labels(): string[] {
    return this.built.map((t) => t.label)
  }

  override initParams(key: Key): Params {
    const own = super.initParams(key)
    return this.hasIntercept ? { ...own, intercept: asTensor(this.interceptStart) } : own
  }

  override initBuffers(): Params {
    return { ...super.initBuffers(), reference: this.reference }
  }

  private partParams(params: Params, i: number): Params {
    return (params[`t${i}`] ?? {}) as Params
  }

  private partBuffers(buffers: Params, i: number): Params {
    return (buffers[`t${i}`] ?? {}) as Params
  }

  private columns(x: Value, features: readonly number[]): Value {
    return take(x, features, 1)
  }

  /**
   * The mean of every term over the reference rows `[T]`, with the parameters and buffers given: what centring
   * subtracts. Zeros when the model is not centred.
   */
  centres(params: Params, buffers: Params = this.buffers): Value {
    const ref = (buffers['reference'] ?? this.reference) as Value
    if (!this.centre || (ref.shape[0] ?? 0) === 0) return zeros([this.built.length])
    const means = this.built.map((t, i) =>
      mean(
        t.module.apply(this.partParams(params, i), this.columns(ref, t.features), {
          train: false,
          buffers: this.partBuffers(buffers, i),
        }),
      ),
    )
    return reshape(
      concat(
        means.map((m) => reshape(m, [1])),
        0,
      ),
      [this.built.length],
    )
  }

  /**
   * Every term's contribution `[n, T]`, centred if the model is, and with feature dropout applied in training when the
   * context carries a key.
   *
   * @param params - The parameters.
   * @param x - The inputs `[n, inFeatures]`.
   * @param ctx - The apply context (training flag, key, buffers).
   * @returns The contributions.
   */
  contributions(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    if (x.shape.length !== 2 || x.shape[1] !== this.inFeatures)
      throw new MdmlError('shape', `AdditiveModel: expected [n, ${this.inFeatures}], got [${x.shape.join(', ')}]`)
    const buffers = ctx.buffers ?? this.buffers
    const run = this.parts(params, ctx)
    const centres = this.centre ? this.centres(params, buffers) : undefined
    const n = x.shape[0] ?? 0
    const columns = this.built.map((t, i) => {
      let v = run.run<Value, Value>(`t${i}`, this.columns(x, t.features))
      if (centres !== undefined) v = sub(v, reshape(sliceAxis(centres, 0, i, i + 1), [1, 1]))
      if (t.featureDropout > 0 && isTraining(ctx) && ctx.key !== undefined) {
        const keep = uniform(foldIn(ctx.key, 1000 + i), [n, 1])
        const mask = fromStorage(
          Float64Array.from(keep.data, (u) => (u >= t.featureDropout ? 1 / (1 - t.featureDropout) : 0)),
          [n, 1],
          'float64',
        )
        v = mul(v, mask)
      }
      return v
    })
    run.done()
    return columns.length === 0 ? zeros([n, 0]) : concat(columns, 1)
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const n = x.shape[0] ?? 0
    const contributions = this.contributions(params, x, ctx)
    const total = this.built.length === 0 ? zeros([n, 1]) : reshape(sum(contributions, 1), [n, 1])
    return this.hasIntercept ? add(total, params['intercept'] as Value) : total
  }

  /**
   * The mean response: the inverse link of the linear predictor.
   *
   * @param params - The parameters.
   * @param x - The inputs.
   * @param ctx - The apply context.
   * @returns The means `[n, 1]`.
   */
  mean(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    return this.link.inverse(this.apply(params, x, { train: false, ...ctx }))
  }

  /**
   * The penalties of the terms: `smoothness` in the `n * loss` convention of `gam` (a penalised likelihood
   * `-loglik + sum lambda_j beta' S_j beta / 2`-style, so divide by the number of rows to add to a mean loss) and
   * `regularisation` in the mean-loss convention (neural additive models' output penalty and weight decay, evaluated
   * on the reference rows).
   *
   * @param params - The parameters.
   * @param buffers - The buffers (default the module's).
   * @returns The two scalars.
   */
  penalties(
    params: Params,
    buffers: Params = this.buffers,
  ): { readonly smoothness: Value; readonly regularisation: Value } {
    const ref = (buffers['reference'] ?? this.reference) as Value
    let smooth: Value = asTensor(0)
    let regular: Value = asTensor(0)
    this.built.forEach((t, i) => {
      const p = this.partParams(params, i)
      const b = this.partBuffers(buffers, i)
      const s = t.smoothness(p, b)
      if (s !== undefined) smooth = add(smooth, s)
      const r = (ref.shape[0] ?? 0) > 0 ? t.regularisation(p, b, this.columns(ref, t.features)) : undefined
      if (r !== undefined) regular = add(regular, r)
    })
    return { smoothness: smooth, regularisation: regular }
  }

  /**
   * The penalty to add to a mean loss over `rows` training rows: the smoothness over `rows` plus the regularisation.
   *
   * @param params - The parameters.
   * @param rows - The number of training rows.
   * @param buffers - The buffers (default the module's).
   * @returns The scalar.
   */
  penalty(params: Params, rows: number, buffers: Params = this.buffers): Value {
    const { smoothness, regularisation } = this.penalties(params, buffers)
    return add(div(smoothness, rows), regularisation)
  }

  /**
   * One single-feature term on a grid, centred like the model: its shape function.
   *
   * @param params - The parameters.
   * @param term - The index of the term.
   * @param grid - The values of its feature.
   * @param buffers - The buffers (default the module's).
   * @returns The curve `[grid.length]`.
   */
  termCurve(params: Params, term: number, grid: ArrayLike<number>, buffers: Params = this.buffers): Tensor {
    const t = this.termAt(term)
    if (t.features.length !== 1) throw new MdmlError('value', `termCurve: ${t.label} has ${t.features.length} features`)
    const cols = fromStorage(Float64Array.from(grid), [grid.length, 1], 'float64')
    const v = t.module.apply(this.partParams(params, term), cols, {
      train: false,
      buffers: this.partBuffers(buffers, term),
    })
    const centre = this.centre ? (toFlat(this.centres(params, buffers))[term] ?? 0) : 0
    return asTensor(toFlat(v).map((u) => u - centre)) as Tensor
  }

  /**
   * One two-feature term on a grid of both features, centred like the model: row `a` is `gridX[a]`, column `b` is
   * `gridZ[b]`.
   *
   * @param params - The parameters.
   * @param term - The index of the term.
   * @param gridX - Values of its first feature.
   * @param gridZ - Values of its second feature.
   * @param buffers - The buffers (default the module's).
   * @returns The surface `[gridX.length, gridZ.length]`.
   */
  termSurface(
    params: Params,
    term: number,
    gridX: ArrayLike<number>,
    gridZ: ArrayLike<number>,
    buffers: Params = this.buffers,
  ): Tensor {
    const t = this.termAt(term)
    if (t.features.length !== 2)
      throw new MdmlError('value', `termSurface: ${t.label} has ${t.features.length} features`)
    const points = Float64Array.from({ length: gridX.length * gridZ.length * 2 }, (_, i) => {
      const cell = Math.floor(i / 2)
      return i % 2 === 0 ? (gridX[Math.floor(cell / gridZ.length)] as number) : (gridZ[cell % gridZ.length] as number)
    })
    const v = t.module.apply(
      this.partParams(params, term),
      fromStorage(points, [gridX.length * gridZ.length, 2], 'float64'),
      { train: false, buffers: this.partBuffers(buffers, term) },
    )
    const centre = this.centre ? (toFlat(this.centres(params, buffers))[term] ?? 0) : 0
    return fromStorage(
      Float64Array.from(toFlat(v), (u) => u - centre),
      [gridX.length, gridZ.length],
      'float64',
    ) as Tensor
  }

  /**
   * The effective degrees of freedom of every term with a linear-model form (splines and linear terms; `undefined` for
   * networks) at the current knots: the joint penalised least-squares fit of those terms on `rows` with unit weights, each
   * column centred (the sum-to-zero constraint), so the per-term edf are those of `gam`'s Gaussian fit with the same
   * penalties: `tr((X'X + S)^-1 X'X)` summed over the term's block.
   *
   * @param params - The parameters.
   * @param rows - The inputs `[n, inFeatures]`.
   * @param buffers - The buffers (default the module's).
   * @returns The edf of each term, in order.
   */
  termEdf(params: Params, rows: Value, buffers: Params = this.buffers): (number | undefined)[] {
    const n = rows.shape[0] ?? 0
    const blocks = this.built.map((t, i) =>
      t.design(this.partParams(params, i), this.partBuffers(buffers, i), this.columns(rows, t.features)),
    )
    const widths = blocks.map((b) => b?.p ?? 0)
    const P = widths.reduce((a, b) => a + b, 0)
    if (P === 0 || n === 0) return blocks.map(() => undefined)
    const offsets = widths.map((_, i) => widths.slice(0, i).reduce((a, b) => a + b, 0))
    const X = new Float64Array(n * P)
    const S = new Float64Array(P * P)
    blocks.forEach((b, i) => {
      if (b === undefined) return
      for (let c = 0; c < b.p; c++) {
        let mean = 0
        for (let r = 0; r < n; r++) mean += b.X[r * b.p + c] as number
        mean /= n
        for (let r = 0; r < n; r++) X[r * P + (offsets[i] as number) + c] = (b.X[r * b.p + c] as number) - mean
        for (let c2 = 0; c2 < b.p; c2++)
          S[((offsets[i] as number) + c) * P + (offsets[i] as number) + c2] = b.S[c * b.p + c2] as number
      }
    })
    const gram = new Float64Array(P * P)
    for (let r = 0; r < n; r++)
      for (let a = 0; a < P; a++) {
        const xa = X[r * P + a] as number
        if (xa !== 0)
          for (let b = 0; b < P; b++) gram[a * P + b] = (gram[a * P + b] as number) + xa * (X[r * P + b] as number)
      }
    const system = Float64Array.from(gram, (v, i) => v + (S[i] as number) + (i % (P + 1) === 0 ? 1e-9 : 0))
    const diag = new Float64Array(P)
    for (let c = 0; c < P; c++) {
      const col = dgesv(
        system,
        Float64Array.from({ length: P }, (_, r) => gram[r * P + c] as number),
        P,
      )
      diag[c] = col[c] as number
    }
    return blocks.map((b, i) =>
      b === undefined ? undefined : diag.slice(offsets[i], (offsets[i] as number) + b.p).reduce((a, v) => a + v, 0),
    )
  }

  private termAt(i: number): BuiltTerm {
    const t = this.built[i]
    if (t === undefined) throw new MdmlError('value', `AdditiveModel: no term ${i}`)
    return t
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), 1]
  }

  override extraRepr(): string {
    return `terms=[${this.labels.join(', ')}], link=${this.link.name}, centre=${this.centre}`
  }
}

const knobsOf = (config: SpaceValue): Readonly<Record<string, SpaceValue>> =>
  typeof config === 'object' && config !== null && !Array.isArray(config)
    ? (config as Readonly<Record<string, SpaceValue>>)
    : {}

/** The reference rows: up to `count` rows spread evenly over the sample (all of them if it is smaller). */
function referenceRows(sample: Value | undefined, d: number, count: number): Value {
  if (sample === undefined) return zeros([0, d])
  const n = sample.shape[0] ?? 0
  if (n <= count) return sample
  return take(
    sample,
    Array.from({ length: count }, (_, i) => Math.floor((i * n) / count)),
    0,
  )
}

function checkHeredity(built: readonly BuiltTerm[], heredity: Heredity): void {
  if (heredity === 'none') return
  const mains = new Set(built.filter((t) => t.features.length === 1).map((t) => t.features[0] as number))
  for (const t of built) {
    if (t.features.length < 2) continue
    const present = t.features.filter((f) => mains.has(f)).length
    const ok = heredity === 'strong' ? present === t.features.length : present > 0
    if (!ok)
      throw new MdmlError(
        'value',
        `AdditiveModel: ${t.label} breaks ${heredity} heredity (main effects of ${t.features.join(', ')} needed)`,
      )
  }
}
