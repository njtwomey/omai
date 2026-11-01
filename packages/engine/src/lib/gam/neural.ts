/**
 * Additive networks as estimators on the trainer base (`learn/bases`): GAMs and GAMLSS whose terms are learnable splines
 * (knots that train with the weights), neural additive model feature nets, interactions and sub-networks, trained by
 * minibatch gradient descent with the optimiser, its schedule and the batch size as slots, and the smoothness penalty
 * inside the loss. `additiveRegressor` (Gaussian, identity link), `additiveClassifier` (a logit predictor per class) and
 * `additiveDistribution` (every parameter of a family has its own additive predictor and link, trained by the negative
 * log-likelihood) give the base's `fit`, `method` and `session`, and their contract adds what a view of the shape
 * functions needs: each term's fitted curve or surface on a grid at any step, the knots, the effective degrees of
 * freedom, the penalty, and all the shape functions as one `dependence` record. They live here, beside `gamlss.ts`,
 * because the trainer base is in `learn`, which `nn` cannot import; the layers, terms and heads are in `nn/additive`.
 */

import type { Json } from '../base/index.ts'
import { families } from '../dists/index.ts'
import { type Dependence, dependence, type Exposition } from '../exposition/index.ts'
import { defineTrainer, type TrainerEstimator, type TrainerFitted } from '../learn/bases/index.ts'
import type { LearnData } from '../learn/index.ts'
import { crossEntropy, mse } from '../losses/index.ts'
import { r2Score } from '../metrics/index.ts'
import {
  AdditiveModel,
  classLogitFamily,
  DistributionalAdditive,
  type HeadFamily,
  type Heredity,
  headFamily,
  type Module,
  type Params,
  splineTerm,
  type TermSpec,
  type TrainState,
} from '../nn/index.ts'
import { optimisers } from '../optim/index.ts'
import { type Key, key } from '../random/index.ts'
import { dict, real, type SpaceValue } from '../space/index.ts'
import {
  asTensor,
  item,
  reshape,
  sliceAxis,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Contract, type DerivedSource, derived } from '../trace/index.ts'

/** The paths an additive network's session adds to the trainer base's. */
export type AdditiveNetPath =
  | 'additive/penalty'
  | 'additive/edf'
  | 'additive/knots'
  | 'additive/shape-functions'
  | `predictor/${number}/term/${number}/curve`
  | `predictor/${number}/term/${number}/surface`

/** The hyperparameters of the additive networks: the terms, the penalty and the training slots. */
export interface AdditiveNetHyper {
  /** The terms of every predictor (default one spline per feature, made from `interiorKnots`, `degree` and `knots`). */
  readonly terms?: readonly TermSpec[]
  /** Interior knots of the default spline terms. */
  readonly interiorKnots: number
  readonly degree: number
  /** The `splineKnots` option of the default spline terms: `{ kind: 'learnable' }` trains the knots. */
  readonly knots: SpaceValue
  /** `clamped` (default) or `extended` (the P-spline knots of `gam`'s `ps` basis) for the default spline terms. */
  readonly boundary: 'clamped' | 'extended'
  /**
   * The smoothness `lambda` of spline terms that give none. The objective is `n * loss + lambda * sum of squared second
   * differences of the coefficients`: for the Gaussian regressor exactly `gam`'s penalised least squares at fixed
   * smoothing parameters; for likelihood heads `lambda` penalises the negative log-likelihood (half `gam`'s deviance scale).
   */
  readonly lambda: number
  /** Centre every term over the training rows (the sum-to-zero constraint; default true). */
  readonly centre: boolean
  /** The training rows kept as the reference for centring and the nets' output penalty (default 64, spread over the data). */
  readonly centreRows: number
  readonly heredity: Heredity
  /** The `optimisers` option (`{ kind: 'adam' }`); the learning rate and schedule are separate. */
  readonly optimiser: SpaceValue
  readonly learningRate: number
  /** A `schedules` option shaping the learning rate (default constant). */
  readonly schedule?: SpaceValue
  readonly epochs: number
  /** Minibatches in all, instead of epochs. */
  readonly steps?: number
  /** Minibatch size (default the whole data). */
  readonly batchSize?: number
  readonly validationFraction?: number
  /** Points of the grid each curve is evaluated on. */
  readonly gridSize: number
  /** Points per axis of the grid each pairwise surface is evaluated on. */
  readonly surfaceSize: number
}

/** The default hyperparameters of the additive networks. */
export const ADDITIVE_NET_DEFAULTS: AdditiveNetHyper = {
  interiorKnots: 8,
  degree: 3,
  knots: { kind: 'uniform' },
  boundary: 'clamped',
  lambda: 0,
  centre: true,
  centreRows: 64,
  heredity: 'none',
  optimiser: { kind: 'adam' },
  learningRate: 0.05,
  epochs: 100,
  gridSize: 60,
  surfaceSize: 20,
}

/** The families `additiveDistribution` knows by name. */
export type AdditiveFamilyName =
  | 'normal'
  | 'gamma'
  | 'negativeBinomial'
  | 'studentT'
  | 'logNormal'
  | 'weibull'
  | 'beta'
  | 'poisson'
  | 'zeroInflatedPoisson'
  | 'skewNormal'
  | 'gaussianLocationScale'
  | 'betaRegression'
  | 'negativeBinomialTheta'
  | 'zeroInflatedPoissonLogit'
  | 'tweedie'

/** The hyperparameters of `additiveDistribution`: the family and, optionally, the terms of each predictor. */
export interface AdditiveDistributionHyper extends AdditiveNetHyper {
  readonly family: AdditiveFamilyName
  /** For each predictor of the family its terms (default the shared `terms`, else a spline per feature). */
  readonly predictorTerms?: readonly (readonly TermSpec[])[]
}

function familyByName(name: AdditiveFamilyName, y: Float64Array): HeadFamily {
  switch (name) {
    case 'weibull':
      return headFamily(families.gamlssWeibull(), y)
    case 'beta':
      return headFamily(families.gamlssBeta(), y)
    case 'poisson':
      return headFamily(families.gamlssPoisson(), y)
    case 'zeroInflatedPoisson':
      return headFamily(families.gamlssZeroInflatedPoisson(), y)
    case 'skewNormal':
      return headFamily(families.gamlssSkewNormal(), y)
    case 'gaussianLocationScale':
      return headFamily(families.gaussianLocationScale(), y)
    case 'betaRegression':
      return headFamily(families.betaRegression(), y)
    case 'negativeBinomialTheta':
      return headFamily(families.negativeBinomialTheta(), y)
    case 'zeroInflatedPoissonLogit':
      return headFamily(families.zeroInflatedPoissonLogit(), y)
    case 'tweedie':
      return headFamily(families.tweedieEstimated(), y)
    default:
      return headFamily(name, y)
  }
}

/** The additive heads of a model, whether it is one predictor or several, and how to reach a head's parameters. */
interface Heads {
  readonly heads: readonly AdditiveModel[]
  readonly names: readonly string[]
  params(tree: Params, k: number): Params
  buffers(tree: Params, k: number): Params
}

function headsOf(model: Module): Heads {
  if (model instanceof DistributionalAdditive) {
    return {
      heads: model.heads,
      names: model.predictors,
      params: (tree, k) => (tree[`p${k}`] ?? {}) as Params,
      buffers: (tree, k) => (tree[`p${k}`] ?? {}) as Params,
    }
  }
  return { heads: [model as AdditiveModel], names: ['mean'], params: (tree) => tree, buffers: (tree) => tree }
}

const rowsOf = (data: LearnData): number => asTensor(data.x).shape[0] ?? 0
const featuresOf = (data: LearnData): number => asTensor(data.x).shape[1] ?? 1
const responseOf = (data: LearnData): Float64Array =>
  Float64Array.from(toFlat(asTensor(data.y ?? data.x)).slice(0, rowsOf(data)))

const defaultTerms = (data: LearnData, h: AdditiveNetHyper): readonly TermSpec[] =>
  h.terms ??
  Array.from({ length: featuresOf(data) }, (_, j) =>
    splineTerm(j, { interiorKnots: h.interiorKnots, degree: h.degree, knots: h.knots, boundary: h.boundary }),
  )

const namesOf = (data: LearnData): readonly string[] | undefined =>
  (data as { readonly featureNames?: readonly string[] }).featureNames

const sampleOf = (data: LearnData): Tensor => asTensor(data.x) as Tensor

const columnRange = (data: LearnData, j: number): readonly [number, number] => {
  const x = sampleOf(data)
  const d = x.shape[1] ?? 1
  const flat = toFlat(x)
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (let i = 0; i < flat.length / d; i++) {
    const v = flat[i * d + j] as number
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  return hi > lo ? [lo, hi] : [lo - 0.5, lo + 0.5]
}

const gridOf = (range: readonly [number, number], n: number): number[] =>
  Array.from({ length: n }, (_, i) => range[0] + ((range[1] - range[0]) * i) / Math.max(1, n - 1))

/** One term's fitted curve or surface at a state, as plain data. */
function termShape(
  structure: Heads,
  data: LearnData,
  h: AdditiveNetHyper,
  state: TrainState,
  k: number,
  i: number,
): Json {
  const head = structure.heads[k] as AdditiveModel
  const term = head.built[i]
  if (term === undefined) return null
  const p = structure.params(state.params, k)
  const b = structure.buffers(state.buffers, k)
  if (term.features.length === 1) {
    const grid = gridOf(columnRange(data, term.features[0] as number), h.gridSize)
    return {
      label: term.label,
      kind: term.kind,
      feature: term.features[0] as number,
      grid,
      values: toFlat(head.termCurve(p, i, grid, b)),
    }
  }
  if (term.features.length === 2) {
    const gx = gridOf(columnRange(data, term.features[0] as number), h.surfaceSize)
    const gz = gridOf(columnRange(data, term.features[1] as number), h.surfaceSize)
    return {
      label: term.label,
      kind: term.kind,
      features: [...term.features],
      gridX: gx,
      gridZ: gz,
      values: toFlat(head.termSurface(p, i, gx, gz, b)),
    }
  }
  return null
}

/** The shape functions of a state as `dependence` records, one per predictor. */
function shapeFunctions(structure: Heads, data: LearnData, h: AdditiveNetHyper, state: TrainState): Dependence[] {
  const d = featuresOf(data)
  const flat = toFlat(sampleOf(data))
  const n = rowsOf(data)
  const names = namesOf(data) ?? Array.from({ length: d }, (_, j) => `x${j}`)
  return structure.heads.map((head, k) => {
    const curves: Dependence['curves'][number][] = []
    const surfaces: NonNullable<Dependence['surfaces']>[number][] = []
    head.built.forEach((_, i) => {
      const s = termShape(structure, data, h, state, k, i) as Record<string, unknown> | null
      if (s === null) return
      if (s['grid'] !== undefined) {
        const feature = s['feature'] as number
        curves.push({
          feature,
          grid: s['grid'] as number[],
          average: s['values'] as number[],
          rug: Array.from({ length: n }, (_, r) => flat[r * d + feature] as number),
        })
      } else {
        const gx = s['gridX'] as number[]
        const gz = s['gridZ'] as number[]
        const v = s['values'] as number[]
        surfaces.push({
          features: s['features'] as [number, number],
          gridX: gx,
          gridY: gz,
          values: gx.map((_, a) => gz.map((__, b) => v[a * gz.length + b] as number)),
        })
      }
    })
    return dependence({
      method: 'shape function',
      features: names,
      output: structure.names[k] as string,
      curves,
      ...(surfaces.length === 0 ? {} : { surfaces }),
      transform: head.centre ? 'centred' : 'none',
    })
  })
}

/** What a fitted additive network offers beyond the verbs of its kind. */
export interface AdditiveVerbs {
  /** The partial effect of every term of the first predictor at new rows `[m, terms]` (centred; the intercept excluded). */
  readonly predictTerms: (x: TensorLike) => Tensor
  /** The linear predictors `[m, predictors]`. */
  readonly linearPredictor: (x: TensorLike) => Tensor
  /** A single-feature term of a predictor on a grid. */
  readonly termCurve: (term: number, grid: ArrayLike<number>, predictor?: number) => Tensor
  /** The effective degrees of freedom of each term by label (spline terms; unit weights), per predictor. */
  readonly edfTerms: Readonly<Record<string, number>>
  /** The shape functions at the trained parameters, one `dependence` record per predictor. */
  readonly shapeFunctions: () => readonly Dependence[]
  /** The term labels of each predictor. */
  readonly termLabels: readonly (readonly string[])[]
  /** The parameters on their natural scale `[m, predictors]` (a distributional model's location, scale, ...). */
  readonly predictParameters?: (x: TensorLike) => Tensor
}

/** A fitted additive network: the trainer base's fitted value and the additive verbs. */
export type AdditiveNetFitted = TrainerFitted & AdditiveVerbs

/** An additive-network estimator: the trainer base's, with the additive verbs on what it fits. */
export interface AdditiveNetEstimator<H extends AdditiveNetHyper>
  extends Omit<TrainerEstimator<H, AdditiveNetPath>, 'fit' | 'with' | 'revive'> {
  fit(...args: Parameters<TrainerEstimator<H, AdditiveNetPath>['fit']>): AdditiveNetFitted
  revive(params: Json): AdditiveNetFitted
  with(changes: Partial<H>): AdditiveNetEstimator<H>
}

interface Variant<H extends AdditiveNetHyper> {
  readonly name: string
  readonly kind: 'regressor' | 'classifier'
  readonly defaults: H
  model(data: LearnData, h: H, key: Key): Module
  objective(h: H, data: LearnData): (prediction: Value, target: Value) => Value
  metrics?: boolean
  /** Verbs of this variant over the structure and a state. */
  extra?(
    model: Module,
    state: { readonly params: Params; readonly buffers: Params },
    data: LearnData,
    h: H,
  ): Record<string, unknown>
}

function additiveEstimator<H extends AdditiveNetHyper>(v: Variant<H>): AdditiveNetEstimator<H> {
  const structures = new WeakMap<Module, Heads>()
  const structureOf = (model: Module): Heads => {
    const found = structures.get(model) ?? headsOf(model)
    structures.set(model, found)
    return found
  }
  const base = defineTrainer<H, AdditiveNetPath>({
    name: v.name,
    kind: v.kind,
    defaults: v.defaults,
    hyperSpace: (h) =>
      dict({
        learningRate: real(1e-4, 1, { log: true, default: h.learningRate, label: 'learning rate' }),
        lambda: real(0, 1e6, { default: h.lambda, label: 'smoothness λ' }),
      }),
    model: v.model,
    objective: v.objective,
    penalty: (_h, data, model) => {
      const n = rowsOf(data)
      return (params) => (model as AdditiveModel).penalty(params, n)
    },
    optimiser: (lr, h) => optimisers.make(h.optimiser)(lr),
    settings: (h) => ({
      learningRate: h.learningRate,
      epochs: h.epochs,
      ...(h.schedule === undefined ? {} : { schedule: h.schedule }),
      ...(h.steps === undefined ? {} : { steps: h.steps }),
      ...(h.batchSize === undefined ? {} : { batchSize: h.batchSize }),
      ...(h.validationFraction === undefined ? {} : { validationFraction: h.validationFraction }),
    }),
    ...(v.metrics === false ? { metrics: () => ({}) } : {}),
    verbs: (model, state, data, h) => {
      const s = structureOf(model)
      const buffers = state.buffers
      const rows = (x: TensorLike): Value => asTensor(x) as Value
      const edfRows = sampleOf(data) as Value
      const edfTerms: Record<string, number> = {}
      s.heads.forEach((head, k) => {
        const edf = head.termEdf(s.params(state.params, k), edfRows, s.buffers(buffers, k))
        head.built.forEach((t, i) => {
          const e = edf[i]
          if (e !== undefined) edfTerms[s.heads.length > 1 ? `${s.names[k]}:${t.label}` : t.label] = e
        })
      })
      const verbs: AdditiveVerbs = {
        predictTerms: (x) =>
          (s.heads[0] as AdditiveModel).contributions(s.params(state.params, 0), rows(x), {
            train: false,
            buffers: s.buffers(buffers, 0),
          }) as Tensor,
        linearPredictor: (x) => model.apply(state.params, rows(x), { train: false, buffers }) as Tensor,
        termCurve: (term, grid, predictor = 0) =>
          (s.heads[predictor] as AdditiveModel).termCurve(
            s.params(state.params, predictor),
            term,
            grid,
            s.buffers(buffers, predictor),
          ),
        edfTerms,
        shapeFunctions: () => shapeFunctions(s, data, h, { params: state.params, buffers } as TrainState),
        termLabels: s.heads.map((head) => head.labels),
      }
      return { ...verbs, ...(v.extra?.(model, state, data, h) ?? {}) }
    },
    contract: (data, h) => {
      const model = v.model(data, h, key(0))
      const s = structureOf(model)
      const n = rowsOf(data)
      const out: Record<string, DerivedSource<TrainState>> = {
        'additive/penalty': derived<TrainState>({
          description: 'the penalty added to the loss: smoothness over the rows plus the nets’ regularisation',
          compute: (st) => item((model as AdditiveModel).penalty(st.params, n, st.buffers)),
        }),
        'additive/edf': derived<TrainState>({
          description: 'effective degrees of freedom of each spline term (unit weights)',
          compute: (st) =>
            s.heads.map((head, k) => ({
              predictor: s.names[k] as string,
              terms: head.labels.map((label, i) => ({
                label,
                edf: head.termEdf(s.params(st.params, k), sampleOf(data) as Value, s.buffers(st.buffers, k))[i] ?? null,
              })),
            })) as Json,
        }),
        'additive/knots': derived<TrainState>({
          description: 'the interior knots of each spline term, as trained',
          compute: (st) =>
            s.heads.map((head, k) => ({
              predictor: s.names[k] as string,
              terms: head.built.flatMap((t, i) => {
                const layer = t.module as { interior?: (p: Params, b?: Params) => Value }
                if (t.kind !== 'spline' || layer.interior === undefined) return []
                const part = (tree: Params): Params => (tree[`t${i}`] ?? {}) as Params
                return [
                  {
                    label: t.label,
                    knots: toFlat(layer.interior(part(s.params(st.params, k)), part(s.buffers(st.buffers, k)))),
                  },
                ]
              }),
            })) as Json,
        }),
        'additive/shape-functions': derived<TrainState>({
          description: 'every term’s fitted curve or surface on its grid, as `dependence` records (one per predictor)',
          cost: 'dear',
          compute: (st) => shapeFunctions(s, data, h, st) as unknown as Json,
        }),
      }
      s.heads.forEach((head, k) => {
        head.built.forEach((t, i) => {
          if (t.features.length === 1)
            out[`predictor/${k}/term/${i}/curve`] = derived<TrainState>({
              description: `${t.label} on a grid`,
              compute: (st) => termShape(s, data, h, st, k, i),
            })
          else if (t.features.length === 2)
            out[`predictor/${k}/term/${i}/surface`] = derived<TrainState>({
              description: `${t.label} on a grid`,
              cost: 'dear',
              compute: (st) => termShape(s, data, h, st, k, i),
            })
        })
      })
      return out as unknown as Contract<TrainState, AdditiveNetPath>
    },
    records: (state, data, h): readonly Exposition[] => {
      const model = v.model(data, h, key(0))
      return shapeFunctions(structureOf(model), data, h, state)
    },
  })
  const wrap = (e: TrainerEstimator<H, AdditiveNetPath>): AdditiveNetEstimator<H> => ({
    ...e,
    fit: (data, options) => e.fit(data, options) as AdditiveNetFitted,
    revive: (params) => e.revive(params) as AdditiveNetFitted,
    with: (changes) => wrap(e.with(changes)),
  })
  return wrap(base)
}

const targetLike = (t: Value, shapeOf: Value): Value => reshape(t, shapeOf.shape)

/**
 * A Gaussian additive network (a GAM trained by gradient descent): an intercept plus learnable splines, feature nets and
 * interactions, squared error plus the P-spline penalty. With fixed knots, `lambda` and enough steps it reaches the
 * penalised least-squares fit of `gam`; with `{ kind: 'learnable' }` knots the knots move to where the curve needs them.
 *
 * @concept additive-models
 * @example
 * ```ts
 * import { additiveRegressor } from 'mdml-engine/gam'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map(([t]) => Math.sin(3 * (t ?? 0)))
 * const fit = additiveRegressor.with({ epochs: 150, interiorKnots: 4 }).fit({ x, y }, { key: key(1) })
 * fit.termLabels // => [['s(x0)']]
 * ```
 */
export const additiveRegressor: AdditiveNetEstimator<AdditiveNetHyper> = additiveEstimator<AdditiveNetHyper>({
  name: 'additive-regressor',
  kind: 'regressor',
  defaults: ADDITIVE_NET_DEFAULTS,
  model: (data, h, key) => {
    const y = responseOf(data)
    return new AdditiveModel({
      inFeatures: featuresOf(data),
      terms: defaultTerms(data, h),
      sample: sampleOf(data),
      intercept: y.reduce((a, b) => a + b, 0) / Math.max(1, y.length),
      centre: h.centre,
      centreRows: h.centreRows,
      heredity: h.heredity,
      lambda: h.lambda,
      ...(namesOf(data) === undefined ? {} : { featureNames: namesOf(data) as readonly string[] }),
      key,
    })
  },
  objective: () => (p, t) => mse(p, targetLike(t, p)),
})

/**
 * A multinomial additive network: one additive logit predictor per class (two for a binary problem), trained by
 * cross-entropy with the smoothness penalty on every class's splines.
 *
 * @concept additive-models
 * @example
 * ```ts
 * import { additiveClassifier } from 'mdml-engine/gam'
 * additiveClassifier.name // => 'additive-classifier'
 * ```
 */
export const additiveClassifier: AdditiveNetEstimator<AdditiveNetHyper> = additiveEstimator<AdditiveNetHyper>({
  name: 'additive-classifier',
  kind: 'classifier',
  defaults: ADDITIVE_NET_DEFAULTS,
  model: (data, h, key) => {
    const labels = [...new Set(toFlat(asTensor(data.y ?? [0, 1])))]
    return new DistributionalAdditive({
      inFeatures: featuresOf(data),
      family: classLogitFamily(Math.max(2, labels.length)),
      terms: Array.from({ length: Math.max(2, labels.length) }, () => defaultTerms(data, h)),
      sample: sampleOf(data),
      centre: h.centre,
      centreRows: h.centreRows,
      heredity: h.heredity,
      lambda: h.lambda,
      key,
    })
  },
  objective: () => (p, t) => crossEntropy(p, t),
})

/**
 * Additive GAMLSS in a network: every parameter of a family (location, scale, shape, inflation) has its own additive
 * predictor with its own terms and link, trained together by the negative log-likelihood. The families are `dists`'
 * (`family: 'normal' | 'gamma' | 'negativeBinomial' | 'studentT' | 'logNormal'`, the ones `gamlss` fits, and more). The
 * fitted value predicts the mean, and `predictDistribution` the distribution of each row.
 *
 * @concept distributional-regression
 * @example
 * ```ts
 * import { additiveDistribution } from 'mdml-engine/gam'
 * additiveDistribution.hyper.family // => 'normal'
 * ```
 */
export const additiveDistribution: AdditiveNetEstimator<AdditiveDistributionHyper> =
  additiveEstimator<AdditiveDistributionHyper>({
    name: 'additive-distribution',
    kind: 'regressor',
    defaults: { ...ADDITIVE_NET_DEFAULTS, family: 'normal' },
    metrics: false,
    model: (data, h, key) => {
      const y = responseOf(data)
      const family = familyByName(h.family, y)
      return new DistributionalAdditive({
        inFeatures: featuresOf(data),
        family,
        terms: family.predictors.map((_, k) => h.predictorTerms?.[k] ?? defaultTerms(data, h)),
        sample: sampleOf(data),
        y,
        centre: h.centre,
        centreRows: h.centreRows,
        heredity: h.heredity,
        lambda: h.lambda,
        key,
      })
    },
    objective: (h, data) => {
      const family = familyByName(h.family, responseOf(data))
      return (p, t) => {
        const n = p.shape[0] ?? 0
        const etas = family.predictors.map((_, k) => sliceColumn(p, k))
        return family.nll(reshape(t, [n]), etas)
      }
    },
    extra: (model, state, _data, _h) => {
      const net = model as DistributionalAdditive
      const mean = (x: TensorLike): Tensor => {
        const dist = net.distribution(state.params, asTensor(x) as Value)
        const mu = dist === undefined ? net.naturalParameters(state.params, asTensor(x) as Value) : dist.mean()
        const t = mu as Tensor
        return dist === undefined ? (sliceColumn(t, 0) as Tensor) : t
      }
      return {
        predict: mean,
        predictDistribution: (x: TensorLike) => net.distribution(state.params, asTensor(x) as Value),
        predictParameters: (x: TensorLike) => net.naturalParameters(state.params, asTensor(x) as Value) as Tensor,
        score: (d: LearnData) => r2Score(asTensor(d.y ?? []), mean(d.x)).estimate.value,
      }
    },
  })

/** Column `k` of `[n, K]` as `[n]`, keeping the graph. */
const sliceColumn = (t: Value, k: number): Value => reshape(sliceAxis(t, 1, k, k + 1), [t.shape[0] ?? 0])
