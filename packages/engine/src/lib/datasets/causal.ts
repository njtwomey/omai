/**
 * Generators for causal inference with known answers: random linear structural causal models over a random DAG (the
 * causal-discovery benchmark), observational treatment-effect data with confounding and both potential outcomes, and
 * instrumental-variable data with an unobserved confounder.
 */

import { type DatasetGenerator, type RegressionData, type Truth, xy } from '../data/index.ts'
import { foldIn, type Key, laplace, normal, permutation, uniform } from '../random/index.ts'
import { structure } from '../scene/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import {
  add,
  cos,
  div,
  log,
  mul,
  sigmoid,
  sin,
  sliceAxis,
  square,
  sub,
  type Tensor,
  tanh,
  type Value,
} from '../tensor/index.ts'
import { column, matrix, stamp, vector } from './_util.ts'

/** Noise draws for structural equations: Gaussian, uniform or Laplace with unit variance. */
function noise(k: Key, n: number, kind: string): Float64Array {
  if (kind === 'uniform') return Float64Array.from(uniform(k, [n], { low: -Math.sqrt(3), high: Math.sqrt(3) }).data)
  if (kind === 'laplace') return Float64Array.from(laplace(k, [n], 0, 1 / Math.SQRT2).data)
  return Float64Array.from(normal(k, [n]).data)
}

/** `(I - B)^-1 - I` for a strictly lower-triangular (in causal order) weight matrix: the total causal effects. */
function totalEffects(b: readonly (readonly number[])[], order: readonly number[]): number[][] {
  const p = b.length
  // T[i][j] = sum over directed paths j -> ... -> i of the product of weights; accumulate in causal order.
  const t = Array.from({ length: p }, () => new Array<number>(p).fill(0))
  for (const i of order) {
    for (let j = 0; j < p; j++) {
      const direct = b[i]?.[j] ?? 0
      if (direct === 0) continue
      const ti = t[i] as number[]
      ti[j] = (ti[j] ?? 0) + direct
      for (let a = 0; a < p; a++) ti[a] = (ti[a] ?? 0) + direct * (t[j]?.[a] ?? 0)
    }
  }
  return t
}

/**
 * A random linear structural causal model (the standard causal-discovery benchmark): a random DAG over `p` variables
 * (edges with probability `edgeProbability` along a random causal order), weights of magnitude in [0.5, 2] with random
 * signs, and independent Gaussian, uniform or Laplace noise (non-Gaussian noise makes the DAG identifiable, as in
 * LiNGAM). All variables are the features; the target repeats the last variable in causal order. The truth has the
 * DAG (as a `Structure`), the weights, the total effects and interventional means `E[X | do(X_j = v)]`.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { linearScm } from 'mdml-engine/datasets'
 * generate(linearScm, { knobs: { variables: 4 } }).truth.structure?.nodes.length // => 4
 * ```
 */
export const linearScm: DatasetGenerator<RegressionData> = {
  name: 'linearScm',
  kind: 'regression',
  description: 'Samples from a random linear structural causal model over a random DAG.',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'samples', description: 'The number of rows.', default: 500 }),
    variables: int(2, 100, {
      label: 'variables',
      description: 'The number of variables (nodes of the DAG).',
      default: 5,
    }),
    edgeProbability: real(0, 1, {
      label: 'edge probability',
      description: 'The chance of an edge between each ordered pair of variables.',
      default: 0.4,
    }),
    noise: choice(['gaussian', 'uniform', 'laplace'], {
      label: 'noise distribution',
      description: 'The distribution of the exogenous noise (non-Gaussian noise makes the DAG identifiable).',
      default: 'gaussian',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['variables'] as number
    const order = Array.from(permutation(foldIn(k, 0), p).data)
    const u = uniform(foldIn(k, 1), [p * p]).data
    const w = uniform(foldIn(k, 2), [p * p]).data
    const b = Array.from({ length: p }, () => new Array<number>(p).fill(0))
    for (let a = 0; a < p; a++)
      for (let c = 0; c < a; c++) {
        // order[c] precedes order[a]: maybe an edge order[c] -> order[a].
        if ((u[a * p + c] ?? 1) < (knobs['edgeProbability'] as number)) {
          const v = w[a * p + c] ?? 0.5
          const mag = 0.5 + 1.5 * Math.abs(2 * v - 1)
          ;(b[order[a] ?? 0] as number[])[order[c] ?? 0] = v < 0.5 ? -mag : mag
        }
      }
    const e = Array.from({ length: p }, (_, j) => noise(foldIn(foldIn(k, 3), j), n, knobs['noise'] as string))
    const x = new Float64Array(n * p)
    for (let i = 0; i < n; i++)
      for (const j of order) {
        let s = e[j]?.[i] ?? 0
        for (let a = 0; a < p; a++) s += (b[j]?.[a] ?? 0) * (x[i * p + a] ?? 0)
        x[i * p + j] = s
      }
    const sink = order[p - 1] ?? 0
    const total = totalEffects(b, order)
    const names = Array.from({ length: p }, (_, j) => `X${j}`)
    // With zero-mean noise, E[X | do(X_j = v)] = v times the total effects of j (and v itself at j).
    const interventionalMean = (j: number, v: number): number[] =>
      names.map((_, i) => (i === j ? v : v * (total[i]?.[j] ?? 0)))
    const parents = (i: number) => names.flatMap((_, a) => ((b[i]?.[a] ?? 0) !== 0 ? [a] : []))
    const truth: Truth = {
      structure: structure(
        'causal-dag',
        names.map((nm) => ({ id: nm, label: nm, role: 'observed' as const })),
        b.flatMap((row, i) =>
          row.flatMap((v, a) =>
            v !== 0 ? [{ from: names[a] ?? '', to: names[i] ?? '', directed: true, label: v.toFixed(2) }] : [],
          ),
        ),
      ),
      relevantFeatures: parents(sink),
      extra: {
        weights: b,
        totalEffects: total,
        causalOrder: order,
        interventionalMean,
        parents: names.map((_, i) => parents(i)),
      },
    }
    return {
      data: xy(matrix(x, n, p), vector(Array.from({ length: n }, (_, i) => x[i * p + sink] ?? 0)), {
        ...stamp('linearScm', linearScm.description, knobs),
        featureNames: names,
        targetNames: [names[sink] ?? 'X'],
      }),
      truth,
    }
  },
}

/**
 * Observational treatment-effect data with known potential outcomes: covariates `x ~ N(0, I)`, a confounded
 * treatment `T ~ Bernoulli(e(x))` with propensity `e(x) = clip(sigmoid(confounding (x0 + x1) / √2), overlap,
 * 1 - overlap)`, baseline outcome `mu0(x)` (linear or nonlinear), and effect `tau(x) = ate + heterogeneity × x0`
 * (or `× tanh(2 x1)` in the nonlinear setting). Features are the covariates and then the treatment; the target is the
 * observed outcome. The truth has the ATE, ATT, CATE, every unit's individual effect and both potential outcomes.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { treatmentEffects } from 'mdml-engine/datasets'
 * generate(treatmentEffects, { knobs: { ate: 2 } }).truth.effects?.ate // => 2
 * ```
 */
export const treatmentEffects: DatasetGenerator<RegressionData> = {
  name: 'treatmentEffects',
  kind: 'regression',
  description: 'Confounded binary treatment, covariates and outcomes with known heterogeneous effects.',
  knobs: dict({
    n: int(2, 1_000_000, { label: 'samples', description: 'The number of units.', default: 1000 }),
    covariates: int(2, 100, { label: 'covariates', description: 'The number of covariates.', default: 5 }),
    setting: choice(['linear', 'nonlinear'], {
      label: 'setting',
      description: 'Linear or nonlinear outcome and propensity models.',
      default: 'linear',
    }),
    ate: real(-100, 100, { label: 'average effect', description: 'The average treatment effect.', default: 1 }),
    heterogeneity: real(0, 10, {
      label: 'heterogeneity',
      description: 'How much the effect varies with the covariates.',
      default: 0.5,
    }),
    confounding: real(0, 10, {
      description: 'How strongly the covariates drive the treatment propensity.',
      default: 1,
      label: 'strength of confounding in the propensity',
    }),
    overlap: real(0, 0.49, {
      description: 'Propensities are clipped to [overlap, 1 - overlap].',
      default: 0.05,
      label: 'propensity clipping',
    }),
    noise: real(0, 10, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the outcome noise.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['covariates'] as number
    const linear = knobs['setting'] === 'linear'
    const ate = knobs['ate'] as number
    const het = knobs['heterogeneity'] as number
    const conf = knobs['confounding'] as number
    const clipLo = knobs['overlap'] as number
    const x = normal(foldIn(k, 0), [n, d])
    const propensity = (v: Value): Value => {
      const raw = sigmoid(mul(add(column(v, 0), column(v, 1)), conf / Math.SQRT2))
      return add(mul(raw, 1 - 2 * clipLo), clipLo)
    }
    const mu0 = (v: Value): Value =>
      linear
        ? add(add(column(v, 0), mul(column(v, 1), 0.5)), mul(column(v, 2), -0.5))
        : add(
            add(sin(mul(mul(column(v, 0), column(v, 1)), Math.PI)), square(sub(column(v, 2), 0.5))),
            mul(cos(column(v, 1)), 0.5),
          )
    const cate = (v: Value): Value => add(mul(linear ? column(v, 0) : tanh(mul(column(v, 1), 2)), het), ate)
    const e = Float64Array.from((propensity(x) as Tensor).data as ArrayLike<number>)
    const u = uniform(foldIn(k, 1), [n]).data
    const t = Float64Array.from(e, (p, i) => ((u[i] ?? 1) < p ? 1 : 0))
    const eps = normal(foldIn(k, 2), [n]).data
    const base = Float64Array.from((mu0(x) as Tensor).data as ArrayLike<number>)
    const tau = Float64Array.from((cate(x) as Tensor).data as ArrayLike<number>)
    const s = knobs['noise'] as number
    const y0 = Float64Array.from(base, (m, i) => m + s * (eps[i] ?? 0))
    const y1 = Float64Array.from(y0, (v, i) => v + (tau[i] ?? 0))
    const y = Float64Array.from(t, (ti, i) => (ti === 1 ? (y1[i] ?? 0) : (y0[i] ?? 0)))
    const features = new Float64Array(n * (d + 1))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) features[i * (d + 1) + j] = x.data[i * d + j] ?? 0
      features[i * (d + 1) + d] = t[i] ?? 0
    }
    const treated = Array.from(t).flatMap((ti, i) => (ti === 1 ? [tau[i] ?? 0] : []))
    const regression = (v: Value): Value => {
      const cov = matrixSlice(v, d)
      const tr = column(v, d)
      return add(mu0(cov), mul(tr, cate(cov)))
    }
    return {
      data: xy(matrix(features, n, d + 1), vector(y), {
        ...stamp('treatmentEffects', treatmentEffects.description, knobs),
        featureNames: [...Array.from({ length: d }, (_, j) => `x${j}`), 'treatment'],
        targetNames: ['outcome'],
      }),
      truth: {
        regressionFunction: regression,
        noiseScale: s,
        effects: {
          ate,
          att: treated.reduce((a, b) => a + b, 0) / Math.max(1, treated.length),
          cate: (v) => cate(matrixSlice(v, d)),
          ite: vector(tau),
          potentialOutcomes: { control: vector(y0), treated: vector(y1) },
        },
        extra: {
          propensity: (v: Value) => propensity(matrixSlice(v, d)),
          treatmentColumn: d,
          sampleAte: tau.reduce((a, b) => a + b, 0) / n,
        },
      },
    }
  },
}

/** The first `d` columns of a matrix value (the covariates). */
function matrixSlice(v: Value, d: number): Value {
  return (v.shape[1] ?? d) === d ? v : sliceAxis(v, 1, 0, d)
}

/**
 * Instrumental-variable data with an unobserved confounder: instrument `Z ~ N(0, 1)`, confounder `U ~ N(0, 1)`,
 * treatment `T = strength Z + confounding U + noise`, outcome `Y = effect T + confounding U + noise`. Features are
 * `[Z, T]`; ordinary least squares of Y on T is biased, two-stage least squares is not. The truth gives the effect and
 * the OLS bias.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { instrumentalVariables } from 'mdml-engine/datasets'
 * generate(instrumentalVariables, { knobs: { effect: 1.5 } }).truth.effects?.ate // => 1.5
 * ```
 */
export const instrumentalVariables: DatasetGenerator<RegressionData> = {
  name: 'instrumentalVariables',
  kind: 'regression',
  description: 'An instrument, a confounded treatment and an outcome, with the true effect.',
  knobs: dict({
    n: int(2, 1_000_000, { label: 'samples', description: 'The number of rows.', default: 1000 }),
    effect: real(-100, 100, {
      label: 'treatment effect',
      description: 'The true causal effect of the treatment on the outcome.',
      default: 1,
    }),
    strength: real(0, 10, {
      description: 'How strongly the instrument moves the treatment (weak instruments near 0).',
      default: 1,
      label: 'instrument strength',
    }),
    confounding: real(0, 10, {
      label: 'confounding',
      description: 'The strength of the unobserved confounder of treatment and outcome.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const beta = knobs['effect'] as number
    const pi = knobs['strength'] as number
    const c = knobs['confounding'] as number
    const z = normal(foldIn(k, 0), [n]).data
    const uu = normal(foldIn(k, 1), [n]).data
    const v = normal(foldIn(k, 2), [n]).data
    const e = normal(foldIn(k, 3), [n]).data
    const t = Float64Array.from(z, (zi, i) => pi * zi + c * (uu[i] ?? 0) + (v[i] ?? 0))
    const y = Float64Array.from(t, (ti, i) => beta * ti + c * (uu[i] ?? 0) + (e[i] ?? 0))
    const features = Float64Array.from({ length: 2 * n }, (_, j) =>
      j % 2 === 0 ? (z[j / 2] ?? 0) : (t[(j - 1) / 2] ?? 0),
    )
    // OLS slope of Y on T converges to beta + cov(T, cU) / var(T) = beta + c^2 / (pi^2 + c^2 + 1).
    const bias = (c * c) / (pi * pi + c * c + 1)
    return {
      data: xy(matrix(features, n, 2), vector(y), {
        ...stamp('instrumentalVariables', instrumentalVariables.description, knobs),
        featureNames: ['instrument', 'treatment'],
        targetNames: ['outcome'],
      }),
      truth: {
        effects: { ate: beta },
        latents: { confounder: vector(uu) },
        extra: { olsBias: bias, instrumentColumn: 0, treatmentColumn: 1, firstStage: pi },
      },
    }
  },
}

/**
 * Panel data for difference-in-differences and event studies, in long format (one row per unit and period): unit and
 * period effects, a share of units that adopt a treatment either all at once or at staggered times, an optional
 * differential pre-trend for the adopters (which breaks parallel trends by a known slope), selection of units with
 * higher baselines into adoption, and an effect that is constant or grows with time since adoption. Features are
 * `[unit, period, treated, adopter]`; the outcome is `y`. The truth has the unit and period effects, every unit's
 * adoption period (-1 for never), each row's individual effect, both potential outcomes, the effect by event time and
 * the average effect on the treated.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { panelEffects } from 'mdml-engine/datasets'
 * generate(panelEffects, { knobs: { units: 10, periods: 6 } }).data.x.shape // => [60, 4]
 * ```
 */
export const panelEffects: DatasetGenerator<RegressionData> = {
  name: 'panelEffects',
  kind: 'regression',
  description: 'Unit-by-period panel with staggered or simultaneous adoption, a pre-trend and dynamic effects.',
  knobs: dict({
    units: int(2, 100_000, { label: 'units', description: 'The number of units.', default: 100 }),
    periods: int(3, 1000, { label: 'periods', description: 'The number of time periods.', default: 10 }),
    adopters: real(0, 1, { label: 'adopter share', description: 'The share of units that ever adopt.', default: 0.5 }),
    timing: choice(['simultaneous', 'staggered'], {
      label: 'timing',
      description: 'All adopters adopt at the middle period, or at times spread over the middle half of the periods.',
      default: 'simultaneous',
    }),
    effect: real(-100, 100, { label: 'effect', description: 'The effect in the period of adoption.', default: 2 }),
    growth: real(-1, 5, {
      label: 'effect growth',
      description: 'The effect at event time e is `effect (1 + growth e)`; 0 is a constant effect.',
      default: 0,
    }),
    trend: real(-5, 5, {
      label: 'differential trend',
      description:
        'Adopters’ outcomes drift by this much more per period than the others’ even untreated (parallel trends fail).',
      default: 0,
    }),
    selection: real(0, 5, {
      label: 'selection',
      description: 'Adopters’ unit effects are higher by this many standard deviations.',
      default: 0.5,
    }),
    noise: real(0, 10, { label: 'noise standard deviation', description: 'The outcome noise.', default: 0.5 }),
  }),
  generate: (k, knobs) => {
    const nu = knobs['units'] as number
    const nt = knobs['periods'] as number
    const effect = knobs['effect'] as number
    const growth = knobs['growth'] as number
    const trend = knobs['trend'] as number
    const noiseSd = knobs['noise'] as number
    const adopter = Array.from(uniform(foldIn(k, 0), [nu]).data, (u) => (u < (knobs['adopters'] as number) ? 1 : 0))
    const mid = Math.floor(nt / 2)
    const spread = uniform(foldIn(k, 1), [nu]).data
    const adoption = adopter.map((a, i) =>
      a === 0
        ? -1
        : knobs['timing'] === 'staggered'
          ? Math.floor(nt / 4) + Math.floor((spread[i] ?? 0) * Math.max(1, nt - 2 * Math.floor(nt / 4) - 1)) + 1
          : mid,
    )
    const unitEffect = Array.from(
      normal(foldIn(k, 2), [nu]).data,
      (z, i) => z + (adopter[i] === 1 ? (knobs['selection'] as number) : 0),
    )
    const periodEffect = Array.from(normal(foldIn(k, 3), [nt]).data, (z, t) => 0.5 * z + 0.1 * t)
    const eps = normal(foldIn(k, 4), [nu * nt]).data
    const n = nu * nt
    const x = new Float64Array(n * 4)
    const y = new Float64Array(n)
    const y0 = new Float64Array(n)
    const y1 = new Float64Array(n)
    const ite = new Float64Array(n)
    let treatedSum = 0
    let treatedCount = 0
    const byEvent = new Map<number, number>()
    for (let i = 0; i < nu; i++)
      for (let t = 0; t < nt; t++) {
        const r = i * nt + t
        const treated = adoption[i] !== -1 && t >= (adoption[i] ?? 0) ? 1 : 0
        const event = t - (adoption[i] ?? 0)
        const tau = adoption[i] === -1 ? 0 : effect * (1 + growth * Math.max(0, event))
        const base =
          (unitEffect[i] ?? 0) + (periodEffect[t] ?? 0) + (adopter[i] === 1 ? trend * t : 0) + noiseSd * (eps[r] ?? 0)
        y0[r] = base
        y1[r] = base + tau
        ite[r] = tau
        y[r] = treated === 1 ? base + tau : base
        x.set([i, t, treated, adopter[i] ?? 0], r * 4)
        if (treated === 1) {
          treatedSum += tau
          treatedCount++
          byEvent.set(event, tau)
        }
      }
    return {
      data: xy(matrix(x, n, 4), vector(y), {
        ...stamp('panelEffects', panelEffects.description, knobs),
        featureNames: ['unit', 'period', 'treated', 'adopter'],
        targetNames: ['outcome'],
      }),
      truth: {
        noiseScale: noiseSd,
        effects: {
          ate: treatedCount === 0 ? 0 : treatedSum / treatedCount,
          att: treatedCount === 0 ? 0 : treatedSum / treatedCount,
          ite: vector(ite),
          potentialOutcomes: { control: vector(y0), treated: vector(y1) },
        },
        latents: { unitEffects: vector(unitEffect), periodEffects: vector(periodEffect) },
        extra: {
          adoptionPeriod: adoption,
          eventTimeEffects: Object.fromEntries([...byEvent.entries()].sort((a, b) => a[0] - b[0])),
          differentialTrend: trend,
          units: nu,
          periods: nt,
        },
      },
    }
  },
}

/**
 * Regression-discontinuity data: a running variable, a cutoff at which treatment switches on (sharply, or with a
 * jump in the probability of treatment for a fuzzy design), and an outcome that is smooth in the running variable
 * apart from the treatment effect. Features are `[running, treated]` (for the fuzzy design the instrument is whether
 * the running variable is at or above the cutoff). The truth has the effect at the cutoff, the first-stage jump and
 * the outcome's jump (the effect times the first stage), and the smooth outcome function without the effect.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { discontinuityDesign } from 'mdml-engine/datasets'
 * generate(discontinuityDesign, { knobs: { n: 50, effect: 3 } }).truth.effects?.ate // => 3
 * ```
 */
export const discontinuityDesign: DatasetGenerator<RegressionData> = {
  name: 'discontinuityDesign',
  kind: 'regression',
  description: 'A running variable with treatment switching on at a cutoff, sharp or fuzzy, with the jump as truth.',
  knobs: dict({
    n: int(10, 10_000_000, { label: 'samples', description: 'The number of units.', default: 2000 }),
    cutoff: real(-10, 10, {
      label: 'cutoff',
      description: 'The value of the running variable at which treatment switches.',
      default: 0,
    }),
    effect: real(-100, 100, { label: 'effect', description: 'The treatment effect at the cutoff.', default: 2 }),
    design: choice(['sharp', 'fuzzy'], {
      label: 'design',
      description: 'Everyone above the cutoff is treated, or the chance of treatment jumps there.',
      default: 'sharp',
    }),
    compliance: real(0, 1, {
      label: 'fuzzy first stage',
      description: 'For a fuzzy design: the chance of treatment is 0.1 below the cutoff and 0.1 plus this above.',
      default: 0.7,
    }),
    slope: real(-10, 10, {
      label: 'slope',
      description: 'The slope of the smooth outcome function below the cutoff.',
      default: 1,
    }),
    slopeChange: real(-10, 10, {
      label: 'slope change',
      description: 'How much the slope changes at the cutoff (a kink that is not an effect).',
      default: 0,
    }),
    curvature: real(-10, 10, {
      label: 'curvature',
      description: 'The coefficient of the squared running variable.',
      default: 0.3,
    }),
    noise: real(0, 10, { label: 'noise standard deviation', description: 'The outcome noise.', default: 1 }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const c = knobs['cutoff'] as number
    const effect = knobs['effect'] as number
    const fuzzy = knobs['design'] === 'fuzzy'
    const compliance = knobs['compliance'] as number
    const slope = knobs['slope'] as number
    const kink = knobs['slopeChange'] as number
    const curve = knobs['curvature'] as number
    const sd = knobs['noise'] as number
    const r = Array.from(normal(foldIn(k, 0), [n]).data, (z) => c + 2 * z)
    const u = uniform(foldIn(k, 1), [n]).data
    const pAbove = fuzzy ? 0.1 + compliance : 1
    const pBelow = fuzzy ? 0.1 : 0
    const smooth = (v: number): number => 1 + slope * (v - c) + kink * Math.max(0, v - c) + curve * (v - c) ** 2
    const treated = r.map((v, i) => ((u[i] ?? 1) < (v >= c ? pAbove : pBelow) ? 1 : 0))
    const eps = normal(foldIn(k, 2), [n]).data
    const y = r.map((v, i) => smooth(v) + effect * (treated[i] ?? 0) + sd * (eps[i] ?? 0))
    const features = new Float64Array(n * 2)
    for (let i = 0; i < n; i++) features.set([r[i] ?? 0, treated[i] ?? 0], i * 2)
    return {
      data: xy(matrix(features, n, 2), vector(y), {
        ...stamp('discontinuityDesign', discontinuityDesign.description, knobs),
        featureNames: ['running', 'treated'],
        targetNames: ['outcome'],
      }),
      truth: {
        noiseScale: sd,
        effects: { ate: effect },
        extra: {
          cutoff: c,
          design: fuzzy ? 'fuzzy' : 'sharp',
          firstStageJump: pAbove - pBelow,
          outcomeJump: effect * (pAbove - pBelow),
          smoothFunction: (v: number): number => smooth(v),
        },
      },
    }
  },
}

/**
 * Synthetic-control data: one treated unit and a pool of donors observed over periods, from a factor model
 * `y_it = delta_t + lambda_t . mu_i + alpha_i + noise`. The treated unit's loadings and unit effect are an exact convex
 * combination of a few donors' (so a perfect synthetic control exists), and from `start` its outcome gains an effect
 * that is constant or grows with time. Features are `[unit, period, treated]` in long format (unit 0 is the treated
 * unit); the truth has the donor weights, the effect path, the treated unit's counterfactual and its mean effect.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { syntheticControlPanel } from 'mdml-engine/datasets'
 * generate(syntheticControlPanel, { knobs: { donors: 9, periods: 12 } }).data.x.shape // => [120, 3]
 * ```
 */
export const syntheticControlPanel: DatasetGenerator<RegressionData> = {
  name: 'syntheticControlPanel',
  kind: 'regression',
  description: 'One treated unit whose counterfactual is a convex combination of donors, with a planted effect.',
  knobs: dict({
    donors: int(3, 1000, { label: 'donors', description: 'The number of donor units.', default: 20 }),
    periods: int(4, 1000, { label: 'periods', description: 'The number of time periods.', default: 30 }),
    start: int(2, 999, { label: 'start', description: 'The first treated period.', default: 20 }),
    factors: int(1, 20, { label: 'factors', description: 'The number of common factors.', default: 3 }),
    support: int(1, 20, {
      label: 'support',
      description: 'How many donors the true weights are spread over.',
      default: 3,
    }),
    effect: real(-100, 100, { label: 'effect', description: 'The effect in the first treated period.', default: 3 }),
    growth: real(-1, 5, {
      label: 'effect growth',
      description: 'The effect at s periods after the start is `effect (1 + growth s)`.',
      default: 0,
    }),
    noise: real(0, 10, { label: 'noise standard deviation', description: 'The idiosyncratic noise.', default: 0.2 }),
  }),
  generate: (k, knobs) => {
    const nd = knobs['donors'] as number
    const nt = knobs['periods'] as number
    const start = Math.min(knobs['start'] as number, nt - 1)
    const nf = knobs['factors'] as number
    const support = Math.min(knobs['support'] as number, nd)
    const effect = knobs['effect'] as number
    const growth = knobs['growth'] as number
    const sd = knobs['noise'] as number
    const nu = nd + 1
    // Smooth common factors (random walks) and period effects.
    const raw = normal(foldIn(k, 0), [nt * nf]).data
    const lambda = new Float64Array(nt * nf)
    for (let f = 0; f < nf; f++) {
      let level = 0
      for (let t = 0; t < nt; t++) {
        level += 0.5 * (raw[t * nf + f] ?? 0)
        lambda[t * nf + f] = level
      }
    }
    const delta = Array.from(normal(foldIn(k, 1), [nt]).data, (z, t) => 0.3 * z + 0.1 * t)
    const mu = normal(foldIn(k, 2), [nu * nf]).data
    const alpha = normal(foldIn(k, 3), [nu]).data
    // Sparse simplex weights on `support` donors.
    const chosen = Array.from(permutation(foldIn(k, 4), nd).data).slice(0, support)
    const u = uniform(foldIn(k, 5), [support]).data
    const total = Array.from(u).reduce((a, b) => a + b + 0.2, 0)
    const weights = new Float64Array(nd)
    chosen.forEach((j, i) => {
      weights[j] = ((u[i] ?? 0) + 0.2) / total
    })
    // Unit 0 is treated; donors are units 1..nd.
    const loadings = new Float64Array(nu * nf)
    const unitEffect = new Float64Array(nu)
    for (let i = 1; i < nu; i++) {
      for (let f = 0; f < nf; f++) loadings[i * nf + f] = mu[i * nf + f] ?? 0
      unitEffect[i] = alpha[i] ?? 0
    }
    for (let j = 0; j < nd; j++) {
      for (let f = 0; f < nf; f++)
        loadings[f] = (loadings[f] ?? 0) + (weights[j] ?? 0) * (loadings[(j + 1) * nf + f] ?? 0)
      unitEffect[0] = (unitEffect[0] ?? 0) + (weights[j] ?? 0) * (unitEffect[j + 1] ?? 0)
    }
    const eps = normal(foldIn(k, 6), [nu * nt]).data
    const n = nu * nt
    const x = new Float64Array(n * 3)
    const y = new Float64Array(n)
    const counterfactual = new Float64Array(nt)
    const effectPath = new Float64Array(nt)
    for (let i = 0; i < nu; i++)
      for (let t = 0; t < nt; t++) {
        let v = (delta[t] ?? 0) + (unitEffect[i] ?? 0)
        for (let f = 0; f < nf; f++) v += (lambda[t * nf + f] ?? 0) * (loadings[i * nf + f] ?? 0)
        v += sd * (eps[i * nt + t] ?? 0)
        const tau = i === 0 && t >= start ? effect * (1 + growth * (t - start)) : 0
        if (i === 0) {
          counterfactual[t] = v
          effectPath[t] = tau
        }
        const r = i * nt + t
        y[r] = v + tau
        x.set([i, t, i === 0 && t >= start ? 1 : 0], r * 3)
      }
    const post = Array.from(effectPath).slice(start)
    return {
      data: xy(matrix(x, n, 3), vector(y), {
        ...stamp('syntheticControlPanel', syntheticControlPanel.description, knobs),
        featureNames: ['unit', 'period', 'treated'],
        targetNames: ['outcome'],
      }),
      truth: {
        noiseScale: sd,
        effects: {
          ate: post.reduce((a, b) => a + b, 0) / post.length,
          att: post.reduce((a, b) => a + b, 0) / post.length,
        },
        extra: {
          treatedUnit: 0,
          start,
          donorWeights: Array.from(weights),
          effectPath: Array.from(effectPath),
          counterfactual: Array.from(counterfactual),
          units: nu,
          periods: nt,
        },
      },
    }
  },
}

/**
 * Randomised-experiment data for uplift modelling: covariates `x ~ N(0, I)`, a binary outcome whose probability without
 * treatment is `p0(x)` and with treatment `p1(x)`, where the log-odds move by `strength (x0 - shift)` so the treatment
 * helps units with a large first covariate, harms those with a small one (the sleeping dogs) and does nothing on
 * average near `shift`. Treatment is assigned with probability `share` independently of `x`. The two potential outcomes
 * share one uniform draw (so the effect on a unit is -1, 0 or 1). Features are the covariates then the treatment; the
 * truth has the uplift function `p1 - p0`, the potential outcomes and the sample average effect.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { upliftData } from 'mdml-engine/datasets'
 * generate(upliftData, { knobs: { n: 50, covariates: 3 } }).data.x.shape // => [50, 4]
 * ```
 */
export const upliftData: DatasetGenerator<RegressionData> = {
  name: 'upliftData',
  kind: 'regression',
  description:
    'A randomised experiment with a binary outcome and a covariate-dependent uplift, including harmed units.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of units.', default: 5000 }),
    covariates: int(1, 100, { label: 'covariates', description: 'The number of covariates.', default: 4 }),
    baseline: real(-5, 5, {
      label: 'baseline log-odds',
      description: 'The log-odds of the outcome without treatment at x = 0.',
      default: -1,
    }),
    strength: real(0, 10, {
      label: 'uplift strength',
      description: 'How fast the log-odds of the effect rise with the first covariate.',
      default: 1.2,
    }),
    shift: real(-5, 5, {
      label: 'shift',
      description: 'The first covariate at which the treatment has no effect.',
      default: 0.3,
    }),
    share: real(0.01, 0.99, {
      label: 'treated share',
      description: 'The probability of being assigned to treatment.',
      default: 0.5,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['covariates'] as number
    const base = knobs['baseline'] as number
    const strength = knobs['strength'] as number
    const shift = knobs['shift'] as number
    const x = normal(foldIn(k, 0), [n, d])
    const p0 = (v: Value): Value =>
      sigmoid(add(add(mul(column(v, 0), 0.4), mul(d > 1 ? column(v, 1) : mul(column(v, 0), 0), -0.3)), base))
    const p1 = (v: Value): Value => sigmoid(add(logitOf(p0(v)), mul(sub(column(v, 0), shift), strength)))
    const a0 = Float64Array.from((p0(x) as Tensor).data as ArrayLike<number>)
    const a1 = Float64Array.from((p1(x) as Tensor).data as ArrayLike<number>)
    const u = uniform(foldIn(k, 1), [n]).data
    const assign = uniform(foldIn(k, 2), [n]).data
    const share = knobs['share'] as number
    const t = Float64Array.from(assign, (a) => (a < share ? 1 : 0))
    const y0 = Float64Array.from(a0, (p, i) => ((u[i] ?? 1) < p ? 1 : 0))
    const y1 = Float64Array.from(a1, (p, i) => ((u[i] ?? 1) < p ? 1 : 0))
    const y = Float64Array.from(t, (ti, i) => (ti === 1 ? (y1[i] ?? 0) : (y0[i] ?? 0)))
    const features = new Float64Array(n * (d + 1))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) features[i * (d + 1) + j] = x.data[i * d + j] ?? 0
      features[i * (d + 1) + d] = t[i] ?? 0
    }
    const ite = Float64Array.from(y1, (v, i) => v - (y0[i] ?? 0))
    const uplift = (v: Value): Value => sub(p1(matrixSlice(v, d)), p0(matrixSlice(v, d)))
    const treatedIte = Array.from(ite).filter((_, i) => t[i] === 1)
    return {
      data: xy(matrix(features, n, d + 1), vector(y), {
        ...stamp('upliftData', upliftData.description, knobs),
        featureNames: [...Array.from({ length: d }, (_, j) => `x${j}`), 'treatment'],
        targetNames: ['outcome'],
      }),
      truth: {
        effects: {
          ate: Array.from(a1).reduce((s, v, i) => s + v - (a0[i] ?? 0), 0) / n,
          att: treatedIte.reduce((s, v) => s + v, 0) / Math.max(1, treatedIte.length),
          cate: uplift,
          ite: vector(ite),
          potentialOutcomes: { control: vector(y0), treated: vector(y1) },
        },
        extra: {
          treatmentColumn: d,
          share,
          upliftAt: (v: Value): Value => uplift(v),
          sampleUplift: vector(Array.from(a1, (v, i) => v - (a0[i] ?? 0))),
        },
      },
    }
  },
}

function logitOf(p: Value): Value {
  return log(div(p, sub(1, p)))
}
