import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { survivalData } from '../datasets/index.ts'
import { smoothSolvers } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { type DictSpace, defaults } from '../space/index.ts'
import { sliceAxis, type TensorLike, toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import * as S from './index.ts'

const near = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs((a[i] ?? 0) - (b[i] ?? 0)), `entry ${i}`).toBeLessThan(tol)
}

/** Weibull proportional-hazards data with the known coefficients of the library's generator. */
function planted(n: number, seed: number, model: 'weibull' | 'lognormal' = 'weibull', features = 2) {
  const { data, truth } = generate(survivalData, {
    knobs: { n, features, model, shape: model === 'weibull' ? 1.5 : 0.7 },
    key: key(seed),
  })
  const y = toFlat(data.y)
  const time = Array.from({ length: n }, (_, i) => y[2 * i] ?? 0)
  const event = Array.from({ length: n }, (_, i) => y[2 * i + 1] ?? 0)
  const beta = (truth.extra as { coefficients: number[] }).coefficients
  return { time, event, x: data.x, beta }
}

describe('Kaplan-Meier and Nelson-Aalen', () => {
  const time = [3, 5, 5, 8, 10, 12, 12, 15]
  const event = [1, 1, 0, 1, 1, 1, 0, 1]

  it('is the product-limit estimate with Greenwood variance, by hand', () => {
    // Event times 3, 5, 8, 10, 12, 15 with 8, 7, 5, 4, 3, 1 at risk.
    const km = S.kaplanMeier(time, event)
    expect(km.time).toEqual([3, 5, 8, 10, 12, 15])
    expect(km.atRisk).toEqual([8, 7, 5, 4, 3, 1])
    expect(km.censored).toEqual([0, 1, 0, 0, 1, 0])
    const expected = [7 / 8, (7 / 8) * (6 / 7), (7 / 8) * (6 / 7) * (4 / 5), 0, 0, 0]
    expected[3] = (expected[2] ?? 0) * (3 / 4)
    expected[4] = (expected[3] ?? 0) * (2 / 3)
    expected[5] = 0
    near(km.survival, expected, 1e-12)
    let g = 0
    const atRisk = [8, 7, 5, 4, 3]
    const se = atRisk.map((n, k) => {
      g += 1 / (n * (n - 1))
      return (expected[k] ?? 0) * Math.sqrt(g)
    })
    near(km.standardError.slice(0, 5), se, 1e-12)
    expect(km.median).toBe(10)
  })

  it('has the log-log, plain and log intervals of survfit', () => {
    const level = 0.95
    const z = 1.959963984540054
    const km = S.kaplanMeier(time, event, { level })
    const plain = S.kaplanMeier(time, event, { interval: 'plain' })
    const logI = S.kaplanMeier(time, event, { interval: 'log' })
    const k = 2
    const s = km.survival[k] ?? 0
    const gw = (km.standardError[k] ?? 0) ** 2 / s ** 2
    const llse = Math.sqrt(gw) / Math.abs(Math.log(s))
    expect(km.lower[k]).toBeCloseTo(s ** Math.exp(z * llse), 12)
    expect(km.upper[k]).toBeCloseTo(s ** Math.exp(-z * llse), 12)
    expect(plain.lower[k]).toBeCloseTo(s - z * s * Math.sqrt(gw), 12)
    expect(logI.lower[k]).toBeCloseTo(Math.exp(Math.log(s) - z * Math.sqrt(gw)), 12)
    for (let i = 0; i < km.time.length - 1; i++) {
      expect(km.lower[i] ?? 0).toBeLessThanOrEqual(km.survival[i] ?? 0)
      expect(km.upper[i] ?? 0).toBeGreaterThanOrEqual(km.survival[i] ?? 0)
    }
  })

  it('answers survival, quantile and restricted-mean questions of the step function', () => {
    const km = S.kaplanMeier(time, event)
    expect(S.survivalAt(km, 2)).toBe(1)
    expect(S.survivalAt(km, 5)).toBeCloseTo(0.75, 12)
    expect(S.survivalQuantile(km, 0.25)).toBe(5)
    expect(S.survivalQuantile(S.kaplanMeier([1, 2, 3], [1, 0, 0]), 0.5)).toBe(Number.POSITIVE_INFINITY)
    // Area: 1 * 3 + 7/8 * 2 + 3/4 * 3 on [0, 8].
    expect(S.restrictedMeanSurvival(km, 8)).toBeCloseTo(3 + 1.75 + 2.25, 12)
  })

  it('is formed one subject per step by the algorithm, with a state that serialises', () => {
    const km = S.kaplanMeier(time, event)
    const algorithm = S.kaplanMeierAlgorithm(time, event)
    const rec = recorder()
    const run = runAlgorithm(algorithm, { seed: 0, maxSteps: 20, tracer: rec })
    expect(run.steps).toBe(8)
    expect(run.status).toBe('completed')
    expect(run.state.survival).toBeCloseTo(km.survival[km.survival.length - 1] ?? Number.NaN, 12)
    // Half way: the state round trips through JSON and the run resumes to the same end.
    const half = runAlgorithm(algorithm, { seed: 0, maxSteps: 4 })
    const restored = algorithm.deserialise?.(algorithm.serialise?.(half.state) ?? null)
    expect(restored?.survival).toBeCloseTo(half.state.survival, 15)
    const rest = runAlgorithm(algorithm, { seed: 0, maxSteps: 4, start: { state: restored, step: 4 } })
    expect(rest.state.survival).toBeCloseTo(run.state.survival, 15)
    expect(rest.state.greenwood).toBeCloseTo(run.state.greenwood, 12)
  })

  it('accumulates Nelson-Aalen d / n and relates to Kaplan-Meier through exp(-H)', () => {
    const na = S.nelsonAalen(time, event)
    near(na.cumulativeHazard.slice(0, 3), [1 / 8, 1 / 8 + 1 / 7, 1 / 8 + 1 / 7 + 1 / 5], 1e-12)
    expect(na.cumulativeHazard[5]).toBeCloseTo(1 / 8 + 1 / 7 + 1 / 5 + 1 / 4 + 1 / 3 + 1, 12)
    expect(na.standardError[1]).toBeCloseTo(Math.sqrt(1 / 64 + 1 / 49), 12)
    const km = S.kaplanMeier(time, event)
    // exp(-H) is close to the product limit estimate where few have failed.
    expect(Math.abs((na.survival[0] ?? 0) - (km.survival[0] ?? 0))).toBeLessThan(0.01)
  })
})

describe('log-rank tests', () => {
  // Two groups, hand-computed at each event time: observed minus expected and the hypergeometric variance.
  const time = [1, 2, 3, 4, 5, 6]
  const group = [0, 1, 0, 1, 0, 1]

  it('sums observed minus expected over the event times', () => {
    const r = S.logRankTest(time, undefined, group)
    let o = 0
    let e = 0
    let v = 0
    const risk = [
      [3, 3],
      [2, 3],
      [2, 2],
      [1, 2],
      [1, 1],
      [0, 1],
    ]
    for (let t = 0; t < 6; t++) {
      const [n0 = 0, n1 = 0] = risk[t] ?? []
      const n = n0 + n1
      const died0 = group[t] === 0 ? 1 : 0
      o += died0
      e += n0 / n
      v += n > 1 ? ((n0 / n) * (n1 / n) * (n - 1)) / (n - 1) : 0
    }
    expect(r.observed[0]).toBe(o)
    expect(r.expected[0]).toBeCloseTo(e, 12)
    expect(r.score[0]).toBeCloseTo(o - e, 12)
    expect(r.covariance[0]).toBeCloseTo(v, 12)
    expect(r.statistic).toBeCloseTo((o - e) ** 2 / v, 12)
    expect(r.df).toBe(1)
    expect(S.logRankZ(r).z ** 2).toBeCloseTo(r.statistic, 10)
  })

  it('is zero when the groups are the same and invariant to relabelling', () => {
    const same = S.logRankTest([1, 1, 2, 2, 3, 3], undefined, [0, 1, 0, 1, 0, 1])
    expect(same.statistic).toBeCloseTo(0, 12)
    const a = S.logRankTest(time, [1, 1, 0, 1, 1, 1], group)
    const b = S.logRankTest(
      time,
      [1, 1, 0, 1, 1, 1],
      group.map((g) => 10 - 3 * g),
    )
    expect(b.statistic).toBeCloseTo(a.statistic, 12)
    expect((a.observed[0] ?? 0) + (a.observed[1] ?? 0)).toBe(5)
  })

  it('weights early times more with Gehan-Breslow and sums strata', () => {
    const t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    const g = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    const lr = S.logRankTest(t, undefined, g)
    const gb = S.logRankTest(t, undefined, g, { weight: 'gehan-breslow' })
    expect(gb.statistic).not.toBeCloseTo(lr.statistic, 3)
    expect(S.logRankTest(t, undefined, g, { weight: 'wilcoxon' }).statistic).toBeCloseTo(gb.statistic, 12)
    // Two copies of the same data as two strata: the scores and variances add, so the statistic doubles.
    const doubled = S.logRankTest([...t, ...t], undefined, [...g, ...g], {
      strata: [...t.map(() => 0), ...t.map(() => 1)],
    })
    expect(doubled.statistic).toBeCloseTo(2 * lr.statistic, 10)
    expect(() =>
      S.logRankTest(
        t,
        undefined,
        t.map(() => 0),
      ),
    ).toThrow(/two groups/)
  })
})

describe('Cox proportional hazards', () => {
  const { time, event, x, beta } = planted(400, 3)

  it('recovers the planted coefficients and reports the usual inference', () => {
    const fit = S.coxPh(S.coxData(time, event, x))
    expect(fit.converged).toBe(true)
    fit.beta.forEach((b, j) => {
      const se = fit.coefficients[j]?.standardError ?? 1
      expect(Math.abs(b - (beta[j] ?? 0)), `beta ${j}`).toBeLessThan(4 * se)
    })
    expect(fit.nEvents).toBeGreaterThan(100)
    expect(fit.likelihoodRatio.pValue).toBeLessThan(1e-3)
    // The three omnibus tests agree to first order in a sample this large.
    expect(Math.abs(fit.wald.statistic / fit.likelihoodRatio.statistic - 1)).toBeLessThan(0.15)
    expect(Math.abs(fit.score.statistic / fit.likelihoodRatio.statistic - 1)).toBeLessThan(0.15)
    expect(fit.concordance).toBeGreaterThan(0.55)
    const c = fit.coefficients[0]
    expect(c?.hazardRatio).toBeCloseTo(Math.exp(c?.coefficient ?? 0), 12)
    expect(c?.lower).toBeCloseTo((c?.coefficient ?? 0) - 1.959963984540054 * (c?.standardError ?? 0), 10)
  })

  it('has the solver as a slot: every smooth solver reaches the Newton optimum', () => {
    const data = S.coxData(time.slice(0, 120), event.slice(0, 120), sliceRows(x, 120))
    const reference = S.coxPh(data, { ties: 'efron' })
    const base = (name: string): object => ({
      ...(defaults(smoothSolvers.space.options[name] as DictSpace) as object),
      kind: name,
    })
    for (const config of [
      { ...base('lbfgs'), gradTol: 1e-9 },
      { ...base('bfgs'), gradTol: 1e-9 },
      { ...base('conjugateGradient'), gradTol: 1e-9 },
    ]) {
      const fit = S.coxPh(data, { solver: config as never, maxIter: 500 })
      near(fit.beta, reference.beta, 1e-5)
      expect(fit.logLikelihood).toBeCloseTo(reference.logLikelihood, 8)
    }
    // The stochastic gradient mode (Adam over the whole set) gets there more slowly, to a looser tolerance.
    const adam = S.coxPh(data, {
      solver: { ...base('gradient'), learningRate: 0.05, optimiser: { kind: 'adam' }, gradTol: 1e-6 } as never,
      maxIter: 4000,
    })
    near(adam.beta, reference.beta, 2e-2)
  })

  it('logs the solver steps into the tape', () => {
    const data = S.coxData(time.slice(0, 60), event.slice(0, 60), sliceRows(x, 60))
    const rec = recorder()
    S.coxPh(data, {}, { tracer: rec })
    const run = rec.run()
    expect(run.events.length).toBeGreaterThan(2)
  })

  it('treats ties by Breslow or Efron, which agree without ties and differ with them', () => {
    const t = [1, 2, 3, 4, 5, 6, 7, 8]
    const e = [1, 1, 1, 0, 1, 1, 0, 1]
    const covariate = [0.3, -0.2, 1.1, 0.4, -0.7, 0.9, 0.1, -0.4]
    const noTies = (ties: S.CoxTies): number => S.coxPh(S.coxData(t, e, covariate), { ties }).beta[0] ?? 0
    expect(noTies('efron')).toBeCloseTo(noTies('breslow'), 10)
    const tied = [1, 1, 2, 2, 3, 3, 4, 4]
    const withTies = (ties: S.CoxTies): number => S.coxPh(S.coxData(tied, e, covariate), { ties }).beta[0] ?? 0
    expect(Math.abs(withTies('efron') - withTies('breslow'))).toBeGreaterThan(1e-3)
  })

  it('has an objective whose gradient vanishes at the fit and a ridge that shrinks it', () => {
    const data = S.coxData(time.slice(0, 100), event.slice(0, 100), sliceRows(x, 100))
    const problem = S.coxProblem(data)
    const fit = S.coxPh(data)
    near(problem.score(fit.beta), [0, 0], 1e-6)
    const ridge = S.coxPh(data, { ridge: 0.5 })
    expect(Math.abs(ridge.beta[0] ?? 0)).toBeLessThan(Math.abs(fit.beta[0] ?? 0))
    // The information is the negative Hessian of the log-likelihood: finite differences of the score.
    const info = problem.information(fit.beta)
    const h = 1e-5
    const up = problem.score([(fit.beta[0] ?? 0) + h, fit.beta[1] ?? 0])
    const dn = problem.score([(fit.beta[0] ?? 0) - h, fit.beta[1] ?? 0])
    expect(info[0]).toBeCloseTo(-((up[0] ?? 0) - (dn[0] ?? 0)) / (2 * h), 4)
  })

  it('gives a monotone baseline hazard and survival curves ordered by risk', () => {
    const data = S.coxData(time, event, x)
    const fit = S.coxPh(data)
    const b = fit.baseline[0]
    for (let k = 1; k < (b?.cumulativeHazard.length ?? 0); k++)
      expect(b?.cumulativeHazard[k] ?? 0).toBeGreaterThanOrEqual(b?.cumulativeHazard[k - 1] ?? 0)
    const times = [1, 5, 10, 20]
    const low = S.coxPredictSurvival(fit, [-1, 1], times)
    const high = S.coxPredictSurvival(fit, [1, -1], times)
    for (let k = 0; k < times.length; k++) expect(high[k] ?? 1).toBeLessThan(low[k] ?? 0)
    expect(S.baselineHazardAt(fit, 0)).toBe(0)
  })

  it('rejects data with no event', () => {
    expect(() => S.coxPh(S.coxData([1, 2, 3], [0, 0, 0], [1, 2, 3]))).toThrow(/event/)
  })
})

describe('accelerated failure time models', () => {
  const { time, event, x, beta } = planted(500, 5)

  it('recovers the planted Weibull model: slopes -beta / k and scale 1 / k', () => {
    const fit = S.aft(S.aftData(time, event, x), { distribution: 'weibull' })
    expect(fit.converged).toBe(true)
    expect(fit.scale).toBeGreaterThan(0)
    fit.beta.forEach((b, j) => {
      const se = fit.coefficients[j + 1]?.standardError ?? 1
      expect(Math.abs(b - -(beta[j] ?? 0) / 1.5), `slope ${j}`).toBeLessThan(4 * se)
    })
    expect(Math.abs(fit.scale - 1 / 1.5)).toBeLessThan(4 * fit.scaleStandardError)
    expect(fit.aic).toBeCloseTo(2 * fit.theta.length - 2 * fit.logLikelihood, 10)
    expect(fit.concordance).toBeGreaterThan(0.55)
  })

  it('recovers a planted log-normal model', () => {
    const p = planted(500, 6, 'lognormal')
    const fit = S.aft(S.aftData(p.time, p.event, p.x), { distribution: 'lognormal' })
    fit.beta.forEach((b, j) => {
      const se = fit.coefficients[j + 1]?.standardError ?? 1
      expect(Math.abs(b - -(p.beta[j] ?? 0)), `slope ${j}`).toBeLessThan(4 * se)
    })
    expect(Math.abs(fit.scale - 0.7)).toBeLessThan(4 * fit.scaleStandardError)
  })

  it('has the solver as a slot: L-BFGS, BFGS and Adam reach the Newton optimum', () => {
    const data = S.aftData(time.slice(0, 150), event.slice(0, 150), sliceRows(x, 150))
    const reference = S.aft(data, { distribution: 'loglogistic' })
    const base = (name: string): object => ({
      ...(defaults(smoothSolvers.space.options[name] as DictSpace) as object),
      kind: name,
    })
    for (const config of [
      { ...base('lbfgs'), gradTol: 1e-9 },
      { ...base('bfgs'), gradTol: 1e-9 },
    ]) {
      const fit = S.aft(data, { distribution: 'loglogistic', solver: config as never, maxIter: 1000 })
      near(fit.theta, reference.theta, 1e-4)
      expect(fit.logLikelihood).toBeCloseTo(reference.logLikelihood, 6)
    }
    const adam = S.aft(data, {
      distribution: 'loglogistic',
      solver: { ...base('gradient'), learningRate: 0.02, optimiser: { kind: 'adam' }, gradTol: 1e-6 } as never,
      maxIter: 6000,
    })
    near(adam.theta, reference.theta, 5e-2)
  })

  it('predicts survival and quantiles that agree, and fits the exponential as scale 1', () => {
    const fit = S.aft(S.aftData(time, event, x), { distribution: 'weibull' })
    const row = [0.2, -0.3]
    for (const p of [0.25, 0.5, 0.9]) {
      const t = S.aftPredictQuantile(fit, row, p)
      expect(S.aftPredictSurvival(fit, row, [t])[0]).toBeCloseTo(1 - p, 10)
    }
    const expo = S.aft(S.aftData(time, event, x), { distribution: 'exponential' })
    expect(expo.scale).toBe(1)
    expect(expo.theta.length).toBe(3)
    // The exponential is nested in the Weibull, so its likelihood is no higher.
    expect(expo.logLikelihood).toBeLessThanOrEqual(fit.logLikelihood + 1e-9)
  })

  it('rejects times that are not positive', () => {
    expect(() => S.aftData([0, 1], [1, 1])).toThrow(/> 0/)
  })
})

describe('concordance index', () => {
  it('counts concordant, discordant and tied comparable pairs', () => {
    const c = S.concordanceIndex([1, 2, 3, 4], [1, 1, 0, 1], [4, 3, 3, 1])
    // Comparable: (0,1),(0,2),(0,3),(1,2),(1,3). Scores 4>3,4>3,4>1,3=3 (tied),3>1.
    expect(c.comparable).toBe(5)
    expect(c.concordant).toBe(4)
    expect(c.tied).toBe(1)
    expect(c.concordance).toBeCloseTo(4.5 / 5, 12)
    expect(S.concordanceIndex([1, 2, 3, 4], [1, 1, 0, 1], [1, 2, 3, 4], { score: 'time' }).concordance).toBe(1)
  })

  it('does not compare a censored subject with a later one, nor two events at the same time', () => {
    const c = S.concordanceIndex([1, 1, 2], [1, 1, 0], [3, 1, 2])
    expect(c.comparable).toBe(2)
    expect(() => S.concordanceIndex([1, 2], [0, 0], [1, 2])).toThrow(/comparable/)
  })
})

/** The first `n` rows of a covariate matrix. */
function sliceRows(x: TensorLike, n: number): TensorLike {
  return sliceAxis(x, 0, 0, n)
}
