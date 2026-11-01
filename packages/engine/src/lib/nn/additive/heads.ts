/**
 * Distributional heads (GAMLSS in a network): every parameter of a distribution (location, scale, shape, inflation) has
 * its own additive predictor with its own terms and its own link, and the heads are trained together by the negative
 * log-likelihood. The families are the library's, not copies: a `DistributionalFamily` from `dists/families` (the
 * normal, Student t, gamma, log-normal, negative binomial, Weibull, beta, Poisson and zero-inflated Poisson of
 * GAMLSS) gives its log-likelihood over tensor operations, so the loss is differentiable as it stands; an
 * `ExtendedFamily` (the Gaussian location-scale, beta regression, zero-inflated Poisson, ordered categorical,
 * multinomial, Tweedie) gives the log-likelihood with its derivatives in the predictors, which become the reverse rule of
 * a custom-derivative loss. The links are `dists/families` links too.
 */

import { customVjp } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
import {
  type DistributionalFamily,
  type ExtendedFamily,
  type GamlssName,
  gamlssExtended,
  gamlssGamma,
  gamlssLogNormal,
  gamlssNegativeBinomial,
  gamlssNormal,
  gamlssStudentT,
  identity as identityLink,
  type Link,
} from '../../dists/families/index.ts'
import type { Distribution } from '../../dists/index.ts'
import { crossEntropy } from '../../losses/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  fromStorage,
  item,
  mean,
  neg,
  reshape,
  sliceAxis,
  stack,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { Composite } from '../module/containers.ts'
import { type ApplyContext, initKey, type Params } from '../module/module.ts'
import { AdditiveModel, type AdditiveModelOptions, type Heredity } from './additive-model.ts'
import { splineTerm, type TermSpec } from './terms.ts'

/** A family as the heads need it: its predictors and links, a start for each, and the loss over the predictors. */
export interface HeadFamily {
  readonly name: string
  /** The names of the parameters that have a predictor. */
  readonly predictors: readonly string[]
  /** The link of each predictor: the parameter is the link's inverse of the predictor. */
  readonly links: readonly Link[]
  /** A starting value of each predictor's intercept, from the response. */
  start(y: Float64Array): readonly number[]
  /** The mean negative log-likelihood of responses `[n]` at predictors (`K` values `[n]`), differentiable in them. */
  nll(y: TensorLike, predictors: readonly Value[]): Value
  /** The fitted distribution at predictors, for residuals, intervals and simulation. */
  distribution(predictors: readonly Tensor[]): Distribution | undefined
}

/** The families a head can be made from: a family object, or a GAMLSS family by name. */
export type HeadFamilySpec = DistributionalFamily | ExtendedFamily | GamlssName | HeadFamily

const named: Readonly<Record<GamlssName, () => DistributionalFamily>> = {
  normal: gamlssNormal,
  gamma: gamlssGamma,
  negativeBinomial: gamlssNegativeBinomial,
  studentT: gamlssStudentT,
  logNormal: gamlssLogNormal,
}

const moments = (y: Float64Array): { readonly mean: number; readonly sd: number } => {
  const n = Math.max(1, y.length)
  const m = y.reduce((a, b) => a + b, 0) / n
  const v = y.reduce((a, b) => a + (b - m) ** 2, 0) / n
  return { mean: m, sd: Math.sqrt(Math.max(v, 1e-12)) }
}

/** A starting predictor for a link: the link of the mean for the first parameter, of the spread for the rest. */
function defaultStart(links: readonly Link[], y: Float64Array): number[] {
  const { mean: m, sd } = moments(y)
  return links.map((link, k) => {
    const target = k === 0 ? m : sd
    const safe = link.name === 'log' || link.name === 'logit' ? Math.max(Math.abs(target), 1e-3) : target
    const v = toFlat(link.link(link.name === 'logit' ? Math.min(Math.max(safe, 1e-3), 1 - 1e-3) : safe))[0]
    return Number.isFinite(v) ? (v as number) : 0
  })
}

const isHeadFamily = (f: unknown): f is HeadFamily => typeof f === 'object' && f !== null && 'nll' in f
const isDistributional = (f: unknown): f is DistributionalFamily =>
  typeof f === 'object' && f !== null && 'fromPredictors' in f

/**
 * A head family from a distributional family (log-likelihood by tensor operations), an extended family (log-likelihood
 * with derivatives, through a custom reverse rule) or a GAMLSS family name. The auxiliary parameters of an extended
 * family (a dispersion, cut points) stay at their starting values from the response, which is why the response is given.
 *
 * @param spec - The family.
 * @param y - The training response, for starting values and an extended family's auxiliary parameters.
 * @returns The head family.
 * @throws MdmlError `value` for an extended family without a response to start from.
 * @concept distributional-regression
 * @example
 * ```ts
 * import { headFamily } from 'mdml-engine/nn'
 * headFamily('normal').predictors // => ['mu', 'sigma']
 * ```
 */
export function headFamily(spec: HeadFamilySpec, y?: ArrayLike<number>): HeadFamily {
  if (isHeadFamily(spec)) return spec
  if (typeof spec === 'string') {
    const f = named[spec]
    if (f === undefined) throw new MdmlError('value', `headFamily: unknown family ${spec}`)
    const start = gamlssExtended(spec).start
    return distributional(f(), (yy) => start(yy, new Float64Array(yy.length).fill(1)))
  }
  if (isDistributional(spec)) return distributional(spec)
  return extended(spec, y)
}

function distributional(family: DistributionalFamily, start?: (y: Float64Array) => readonly number[]): HeadFamily {
  const links = family.parameters.map((p) => p.link)
  return {
    name: family.name,
    predictors: family.parameters.map((p) => p.name),
    links,
    start: start ?? ((y) => defaultStart(links, y)),
    nll: (y, predictors) => neg(mean(family.fromPredictors(predictors).logProb(y))),
    distribution: (predictors) => family.fromPredictors(predictors),
  }
}

function extended(family: ExtendedFamily, y: ArrayLike<number> | undefined): HeadFamily {
  const responses = y === undefined ? undefined : Float64Array.from(y)
  const aux =
    responses === undefined ? [] : family.aux(responses, new Float64Array(responses.length).fill(1)).map((a) => a.start)
  const K = family.predictors.length
  return {
    name: family.name,
    predictors: family.predictors,
    links: family.links,
    start: (yy) => family.start(yy, new Float64Array(yy.length).fill(1)),
    nll: (target, predictors) => {
      const yy = Float64Array.from(toFlat(asTensor(target)))
      const n = yy.length
      const w = new Float64Array(n).fill(1)
      const at = (etas: readonly Value[]): ReturnType<ExtendedFamily['derivatives']> =>
        family.derivatives(
          yy,
          w,
          etas.map((e) => Float64Array.from(toFlat(e))),
          aux,
          2,
        )
      const loss = customVjp({
        name: `${family.name}/nll`,
        forward: (...etas) => asTensor(-at(etas).logLik / n),
        vjp: (g, etas) => {
          const d = at(etas)
          const scale = item(g) / n
          return Array.from({ length: K }, (_, k) =>
            fromStorage(
              Float64Array.from(d.grad[k] as Float64Array, (v) => -scale * v),
              [n],
              'float64',
            ),
          )
        },
      })
      return loss(...predictors)
    },
    distribution: (predictors) =>
      family.distribution?.(
        predictors.map((p) => Float64Array.from(toFlat(p))),
        aux,
      ),
  }
}

/**
 * The categorical family as a head family: one logit predictor per class and the cross-entropy loss. With a smoothness
 * penalty on each class's terms this is the multinomial additive model; two classes are the binary logistic case.
 *
 * @param classes - The number of classes (at least 2).
 * @returns The head family, predictors `class0 .. class(K-1)`, identity links; targets are class indices.
 * @throws MdmlError `value` for fewer than two classes.
 * @concept distributional-regression
 * @example
 * ```ts
 * import { classLogitFamily } from 'mdml-engine/nn'
 * import { item, tensor } from 'mdml-engine/tensor'
 * Math.abs(item(classLogitFamily(2).nll([0, 1], [tensor([0, 0]), tensor([0, 0])])) - Math.LN2) < 1e-12 // => true
 * ```
 */
export function classLogitFamily(classes: number): HeadFamily {
  if (!(Number.isInteger(classes) && classes >= 2))
    throw new MdmlError('value', 'classLogitFamily: needs at least two classes')
  return {
    name: `categorical(${classes})`,
    predictors: Array.from({ length: classes }, (_, k) => `class${k}`),
    links: Array.from({ length: classes }, () => identityLink()),
    start: () => new Array<number>(classes).fill(0),
    nll: (y, predictors) => crossEntropy(stack(predictors, 1), y),
    distribution: () => undefined,
  }
}

/** Options of {@link DistributionalAdditive}. */
export interface DistributionalAdditiveOptions {
  readonly inFeatures: number
  readonly family: HeadFamily
  /**
   * For each predictor its term list: `undefined` is one spline per feature (as the GAMs of `gam` default to), an empty
   * list is a constant parameter.
   */
  readonly terms?: readonly (readonly TermSpec[] | undefined)[]
  readonly sample?: TensorLike
  /** The training response: the intercepts start at the family's starting predictors. */
  readonly y?: ArrayLike<number>
  readonly centre?: boolean
  readonly centreRows?: number
  readonly heredity?: Heredity
  readonly lambda?: number
  readonly featureNames?: readonly string[]
  readonly key?: Key
}

/**
 * Additive predictors for every parameter of a family: part `p<k>` is the {@link AdditiveModel} of predictor `k` (its
 * own terms, centring, penalties and link), and `apply` returns the predictors side by side, `[n, K]`.
 * {@link DistributionalAdditive.nll} is the loss, {@link DistributionalAdditive.naturalParameters} the parameters on their
 * natural scale, {@link DistributionalAdditive.distribution} the fitted distribution of each row.
 *
 * @concept distributional-regression
 * @example
 * ```ts
 * import { DistributionalAdditive, headFamily } from 'mdml-engine/nn'
 * const net = new DistributionalAdditive({ inFeatures: 1, family: headFamily('normal') })
 * net.forward([[0.2], [0.9]]).shape // => [2, 2]
 * net.predictors // => ['mu', 'sigma']
 * ```
 */
export class DistributionalAdditive extends Composite {
  readonly kind = 'DistributionalAdditive'
  readonly family: HeadFamily
  readonly heads: readonly AdditiveModel[]

  constructor(options: DistributionalAdditiveOptions) {
    const family = options.family
    const base = initKey('DistributionalAdditive', options.key)
    const start = options.y === undefined ? family.links.map(() => 0) : family.start(Float64Array.from(options.y))
    const heads = family.predictors.map((_, k) => {
      const given = options.terms?.[k]
      const terms = given ?? Array.from({ length: options.inFeatures }, (_, j) => splineTerm(j, { interiorKnots: 6 }))
      const model: AdditiveModelOptions = {
        inFeatures: options.inFeatures,
        terms,
        intercept: start[k] ?? 0,
        link: family.links[k] as Link,
        key: foldIn(base, k),
        ...(options.sample === undefined ? {} : { sample: options.sample }),
        ...(options.centre === undefined ? {} : { centre: options.centre }),
        ...(options.centreRows === undefined ? {} : { centreRows: options.centreRows }),
        ...(options.heredity === undefined ? {} : { heredity: options.heredity }),
        ...(options.lambda === undefined ? {} : { lambda: options.lambda }),
        ...(options.featureNames === undefined ? {} : { featureNames: options.featureNames }),
      }
      return new AdditiveModel(model)
    })
    super(Object.fromEntries(heads.map((h, k) => [`p${k}`, h])))
    this.family = family
    this.heads = heads
  }

  /** The names of the parameters with a predictor. */
  get predictors(): readonly string[] {
    return this.family.predictors
  }

  /** The predictors `[n, K]` (linear predictors, on the link scale). */
  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const run = this.parts(params, ctx)
    const etas = this.heads.map((_, k) => run.run<Value, Value>(`p${k}`, x))
    run.done()
    return concat(etas, 1)
  }

  /** The `K` predictors of an output `[n, K]`, each `[n]`. */
  split(output: Value): Value[] {
    const n = output.shape[0] ?? 0
    return this.heads.map((_, k) => reshape(sliceAxis(output, 1, k, k + 1), [n]))
  }

  /**
   * The mean negative log-likelihood of responses at predictors: the training objective.
   *
   * @param output - The predictors `[n, K]`.
   * @param y - The responses `[n]`.
   * @returns The scalar loss.
   */
  nll(output: Value, y: TensorLike): Value {
    const n = output.shape[0] ?? 0
    return this.family.nll(reshape(asTensor(y), [n]), this.split(output))
  }

  /**
   * The parameters on their natural scales (the inverse links of the predictors), `[n, K]`.
   *
   * @param params - The parameters.
   * @param x - The inputs.
   * @returns The parameters.
   */
  naturalParameters(params: Params, x: Value): Value {
    const etas = this.split(this.apply(params, x, { train: false }))
    const n = x.shape[0] ?? 0
    return concat(
      etas.map((e, k) => reshape((this.family.links[k] as Link).inverse(e), [n, 1])),
      1,
    )
  }

  /**
   * The fitted distribution of every row.
   *
   * @param params - The parameters.
   * @param x - The inputs.
   * @returns The distribution, or undefined when the family has none.
   */
  distribution(params: Params, x: Value): Distribution | undefined {
    return this.family.distribution(this.split(this.apply(params, x, { train: false })) as Tensor[])
  }

  /**
   * The penalty to add to a mean loss over `rows` training rows: the sum over the predictors' penalties.
   *
   * @param params - The parameters.
   * @param rows - The number of training rows.
   * @param buffers - The buffers (default the module's).
   * @returns The scalar.
   */
  penalty(params: Params, rows: number, buffers: Params = this.buffers): Value {
    let total: Value = asTensor(0)
    this.heads.forEach((h, k) => {
      total = add(total, h.penalty((params[`p${k}`] ?? {}) as Params, rows, (buffers[`p${k}`] ?? {}) as Params))
    })
    return total
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.heads.length]
  }

  override extraRepr(): string {
    return `family=${this.family.name}, predictors=[${this.predictors.join(', ')}]`
  }
}
